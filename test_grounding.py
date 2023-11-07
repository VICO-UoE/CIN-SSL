import argparse
import os
import random
import warnings
from tqdm import tqdm
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
from data import localized_narratives_pretrain_loader
from data import data_utils

from models.mcr import BertPretrain

from utils.evaluator import Evaluator
from utils.utils import (
    union_target,
    get_world_size,
    get_rank,
    init_distributed_mode,
)
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.serialization import SourceChangeWarning
from scipy import spatial

import warnings

warnings.filterwarnings("ignore", category=SourceChangeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)




list_of_pronouns = [
    "them",
    "they",
    "their",
    "this",
    "that",
    "which",
    "those",
    "it",
    "who",
    "he",
    "she",
    "her",
    "him",
    "its",
    "his",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--eval", action="store_true", help="evaluation mode")

    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--gpus", type=int, default=2, help="training epochs")
    parser.add_argument(
        "--save_name",
        type=str,
        default="models_E/model_E",
        help="name for saved model",
    )
    parser.add_argument(
        "--adaptor-layers", action="store_true", help="train with adaptor layers"
    )
    parser.add_argument(
        "--matching",
        type=str,
        default="maximum",
        help="hungarian / maximum matching",
    )
    parser.add_argument(
        "--use-prior", action="store_true", help="use prior loss in unsupervised loss"
    )
    parser.add_argument(
        "--label-prop", action="store_true", help="use label propagation."
    )
    parser.add_argument(
        "--sentence-patch-sim",
        action="store_true",
        help="Sentence patch similarity using clip.",
    )
    parser.add_argument("--use-ssl", action="store_true", help="use SSL.")

    parser.add_argument(
        "--noisy-mouse-labels", action="store_true", help="use label propagation."
    )
    parser.add_argument(
        "--test_set", action="store_true", help="use test set for evaluation"
    )
    parser.add_argument(
        "--model_type", type=str, default="transformer", help="use faster rcnn features"
    )
    parser.add_argument("--ssl_loss", type=str, default="fro", help="fro/contrastive")
    parser.add_argument(
        "--image_features_type",
        type=str,
        default="faster_rcnn",
        help="use faster rcnn features",
    )

    parser.add_argument("--use-ema", action="store_true", help="use EMA model")
    parser.add_argument(
        "--bbox-reg", action="store_true", help="use bbox regression loss"
    )
    parser.add_argument(
        "--trans-func",
        action="store_true",
        help="use transformation function to learn hm",
    )
    parser.add_argument(
        "--mouse-trace-as-inputs",
        action="store_true",
        help="use transformation function to learn hm",
    )
    parser.add_argument("--ema-decay", default=0.999, type=float, help="EMA decay rate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--image_feature_dim", default=2048, type=int)
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="url used to set up distributed training",
    )
    parser.add_argument("--distributed", default=True, type=bool)
    # TODO
    parser.add_argument(
        "--unsupervised", action="store_true", help="test unsupervised accuracy"
    )
    parser.add_argument(
        "--model_config", type=str, help="path to model structure config json"
    )
    parser.add_argument("--sched", type=str, default="cosine", help="scheduler")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument("--decay_rate", type=float, default=1, help="decay rate")
    parser.add_argument("--warmup_lr", type=float, default=1e-5, help="warmup lr")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="warmup epochs")
    parser.add_argument(
        "--cooldown_epochs", type=int, default=0, help="cooldown epochs"
    )
    parser.add_argument(
        "--encoding-type",
        type=str,
        default="bert",
        help="which type of encodig to use",
    )
    parser.add_argument(
        "--use-phrase-mask", action="store_true", help="use phrase specific masking."
    )

   
    args = parser.parse_args()
    return args


eval_mlm = False


def evaluate(test_loader, model, device, args):
    use_gpu = torch.cuda.is_available()

    correct_preds = 0
    all_preds = 0

    model = model.float()

    pred_bboxes_list = []
    target_bboxes_list = []
    num_query_list = []
    preds = {}
    for (
        idx,
        phrase_queries,
        image_features,
        object_label_input_ids,
        object_label_attention_mask,
        object_regions,
        phrase_queries_input_ids,
        phrase_queries_attention_mask,
        caption_input_ids,
        caption_attn_masks,
        sense2vec_feats,
        query_start_end,
        num_objects,
        num_query,
        target_bboxes,
        _,
        mouse_trace_feats,
        _,
        _,
        _,
    ) in tqdm(test_loader):
        if use_gpu:
            (
                idx,
                image_features,
                object_label_input_ids,
                object_label_attention_mask,
                object_regions,
                phrase_queries_input_ids,
                phrase_queries_attention_mask,
                caption_input_ids,
                caption_attn_masks,
                sense2vec_feats,
                query_start_end,
                num_objects,
                num_query,
                target_bboxes,
                mouse_trace_feats,
            ) = (
                idx.to(device),
                image_features.to(device),
                object_label_input_ids.to(device),
                object_label_attention_mask.to(device),
                object_regions.to(device),
                phrase_queries_input_ids.to(device),
                phrase_queries_attention_mask.to(device),
                caption_input_ids.to(device),
                caption_attn_masks.to(device),
                sense2vec_feats.to(device),
                query_start_end.to(device),
                num_objects.to(device),
                num_query.to(device),
                target_bboxes.to(device),
                mouse_trace_feats.to(device),
            )
        dict = {}
        model.eval()
        caption_input_ids = caption_input_ids.squeeze(1)
        caption_attn_masks = caption_attn_masks.squeeze(1)

        phrase_queries_input_ids = phrase_queries_input_ids.squeeze(2)
        phrase_queries_attention_mask = phrase_queries_attention_mask.squeeze(2)
        if args.encoding_type == "glove":
            object_label_input_ids = object_label_input_ids.unsqueeze(-1)
            object_label_attention_mask = object_label_attention_mask.unsqueeze(-1)
        else:
            object_label_input_ids = object_label_input_ids.squeeze(-2)
            object_label_attention_mask = object_label_attention_mask.squeeze(-2)
            object_label_input_ids = object_label_input_ids.view(-1, 16)
            object_label_attention_mask = object_label_attention_mask.view(-1, 16)
        if int(num_query[0]) > 0:
            (
                _,
                grounding_matrix,
                pred_coref_matrix,
                _,
                _,
                _,
                _,
                target,
                probs,
                text_embeds,
                _,
                query_box_max_id,
                pred_bboxes,
                modified_pred_bboxes,
            ) = model.forward(
                image_features,
                object_label_input_ids,
                object_label_attention_mask,
                object_regions,
                caption_input_ids,
                caption_attn_masks,
                sense2vec_feats,
                query_start_end,
                num_objects,
                num_query,
                mouse_trace_feats,
                phrase_queries_input_ids=phrase_queries_input_ids,
            )

            # sup acc

            # pred_bboxes = modified_pred_bboxes.unsqueeze(0)
            target_pred = torch.argmax(target, dim=1)  # [B]
            prediction = torch.argmax(probs, dim=1)  # [all_querys]
            correct_preds += int(prediction.eq(target_pred).sum())
            all_preds += len(prediction)

            
            pred_bboxes_list = pred_bboxes.cpu().tolist()
            target_bboxes_list = target_bboxes.cpu().tolist()
            num_query_list = num_query.cpu().tolist()
            _, pred_list, gtbox_list, query_word_list, _, _ = evaluate_helper(
                pred_bboxes_list,
                target_bboxes_list,
                num_query_list,
                phrase_queries,
            )
            dict["gtbox"] = gtbox_list
            dict["pred_box"] = pred_list
            dict["queries"] = query_word_list

            if str(idx.cpu().tolist()[0]) not in preds:
                preds[str(idx.cpu().tolist()[0])] = [dict]
            else:
                preds[str(idx.cpu().tolist()[0])].append(dict)
            # preds.append(dict)
        else:
            continue

    out_file = open(os.path.join("saved/model_unsup_predictions.json"), "w")
    data = json.dump(preds, out_file, indent=4)
    out_file.close()
    overall_score, _, _, _, noun_score, pronoun_score = evaluate_helper(
        pred_bboxes_list,
        target_bboxes_list,
        num_query_list,
        phrase_queries,
    )
    supacc = correct_preds / all_preds

    return noun_score, pronoun_score, overall_score, supacc


def evaluate_helper(pred_bboxes, target_bboxes, num_query, phrase_queries):
    evaluator = Evaluator()
    noun_pred_list = []
    noun_gtbox_list = []
    pronoun_pred_list = []
    pronoun_gtbox_list = []
    pred_list = []
    gtbox_list = []
    query_list = []
    for pred, targ, nq in zip(pred_bboxes, target_bboxes, num_query):
        # ipred: [query, 5]
        # itarget: [query, 12, 4]
        if nq > 0:
            noun_nq = 0
            pronoun_nq = 0
            noun_target_bboxes = []
            noun_pred_boxes = []
            pronoun_pred_boxes = []
            pronoun_target_bboxes = []
            pronoun_pred_boxes = []
            pred_list += pred[:nq]
            gtbox_list += union_target(targ[:nq])  # [query, 4]
            phrase_query = phrase_queries[:nq]
            for i, phrase in enumerate(phrase_query):
                if str(phrase[0]) not in list_of_pronouns:
                    noun_nq += 1
                    noun_pred_boxes.append(pred[i])
                    noun_target_bboxes += [targ[i]]

                else:
                    pronoun_nq += 1
                    pronoun_pred_boxes.append(pred[i])
                    pronoun_target_bboxes += [targ[i]]
            noun_pred_list += noun_pred_boxes
            pronoun_pred_list += pronoun_pred_boxes
            noun_gtbox_list += union_target(noun_target_bboxes)  # [query, 4]
            pronoun_gtbox_list += union_target(pronoun_target_bboxes)  # [query, 4]
    if len(gtbox_list) > 0:
        accuracy, _ = evaluator.evaluate(pred_list, gtbox_list)  # [query, 4]

        noun_accuracy, _ = evaluator.evaluate(
            noun_pred_list, noun_gtbox_list
        )  # [query, 4]
        pronoun_accuracy = 0.0
        
    else:
        accuracy = 0.0
        noun_accuracy = 0.0
        pronoun_accuracy = 0.0
    query_list += phrase_queries[:nq]
    return accuracy, pred_list, gtbox_list, query_list, noun_accuracy, pronoun_accuracy


if __name__ == "__main__":
    args = parse_args()
    print(args)
    init_distributed_mode(args)

    device = torch.device(args.device)
    config = json.load(open(args.model_config, "rb"))
    # set random seed
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if torch.cuda.device_count() >= 1:
        print("Use {} gpus".format(torch.cuda.device_count()))
    wordEmbedding = data_utils.load_vocabulary("datasets/glove/glove.6B.300d.txt")
    test_dset = localized_narratives_pretrain_loader.LocalizedNarrativesFlickr30dataset(
        dataroot="datasets/",
        image_features_type=args.image_features_type,
        word_embedding=wordEmbedding,
        split="test",
        eval_grounding=True,
    )
    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        test_sampler = torch.utils.data.DistributedSampler(
            test_dset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        test_sampler = None

    test_loader = DataLoader(
        test_dset,
        batch_size=args.batch,
        num_workers=4,
        drop_last=True,
        sampler=test_sampler,
    )


    model = BertPretrain(
        text_encoder="bert-base-uncased",
        config=config,
        args=args,
    )

    checkpoint = torch.load(
        str(args.save_name),
        map_location=device,
    )

    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        print("CUDA available")
        model = model.to(device)
    score = evaluate(test_loader, model, device, args)
    print("untrained eval score:", score)
