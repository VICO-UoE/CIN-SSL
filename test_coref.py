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

from utils.utils import (
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
import jsonlines

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
ref_dir = "coref/modelrefs"


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
        "--label-prop", action="store_true", help="use label propagation."
    )
    parser.add_argument(
        "--sentence-patch-sim",
        action="store_true",
        help="Sentence patch similarity using clip.",
    )
    parser.add_argument("--use-ssl", action="store_true", help="use SSL.")

   
    parser.add_argument(
        "--test_set", action="store_true", help="use test set for evaluation"
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
        "--use-phrase-mask", action="store_true", help="use phrase specific masking."
    )
    parser.add_argument(
        "--encoding-type",
        type=str,
        default="bert",
        help="which type of encodig to use",
    )
  
    args = parser.parse_args()
    return args



def evaluate(test_loader, model, device, args):
    use_gpu = torch.cuda.is_available()

    correct_preds = 0
    all_preds = 0

    model = model.float()

    pred_bboxes_list = []
    target_bboxes_list = []
    num_query_list = []
    final_output_data = []
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
        mouse_trace_for_phrases,
        mouse_trace_feats,
        _,
        rule_coref_matrix,
        max_assignments,
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
                mouse_trace_for_phrases,
                mouse_trace_feats,
                rule_coref_matrix,
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
                mouse_trace_for_phrases.to(device),
                mouse_trace_feats.to(device),
                rule_coref_matrix.to(device),
            )

        model.eval()
        caption_input_ids = caption_input_ids.squeeze(1)
        caption_attn_masks = caption_attn_masks.squeeze(1)
        mouse_trace_for_phrases = mouse_trace_for_phrases.squeeze(-2)
        phrase_queries_input_ids = phrase_queries_input_ids.squeeze(2)
        phrase_queries_attention_mask = phrase_queries_attention_mask.squeeze(2)
        
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
                matrix_logits_reg,
                target,
                probs,
                text_embeds,
                _,
                query_box_max_id,
                _,
                modefied_mouse_trace_for_phrases,
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
                mouse_trace_feats=mouse_trace_feats,
                phrase_queries_input_ids=phrase_queries_input_ids,
            )

            cost_matrix = matrix_logits_reg[0]
            argmin_cos_dist = []
            cost_matrix = cost_matrix[: int(num_query), : int(num_query)]
            cost_matrix = cost_matrix.cpu().detach().numpy()


            for k in range(int(num_query)):
                cos_dist = np.where(cost_matrix[k] > 0.8)[0].tolist()
                argmin_cos_dist.append(cos_dist)
            
            argmin_cos_dist = np.unique(np.array(argmin_cos_dist)).tolist()
            
            y = []
            if not isinstance(argmin_cos_dist[0], list):
                argmin_cos_dist = np.expand_dims(argmin_cos_dist, axis=1).tolist()
            for sublist in argmin_cos_dist:
                found = False
                for i, y_sublist in enumerate(y):
                    if any(num in y_sublist for num in sublist):
                        y[i].extend(sublist)
                        found = True
                        break
                if not found:
                    y.append(sublist)

            # Remove duplicates in each sublist
            for i, y_sublist in enumerate(y):
                y[i] = list(set(y_sublist))
            final_list = y

            gold_data = evaluate_helper(
                idx.cpu().tolist(),
                final_list,
                phrase_queries,
                num_query.cpu().tolist(),
            )
            final_output_data.append(gold_data)


def evaluate_helper(image_id, query_box_max_id, phrase_queries, num_queries):
    for img_id, n_query in zip(image_id, num_queries):
        similar_clusters = {}
        count = 0
        max_box_id = query_box_max_id[:n_query]
        phrase_query = phrase_queries[:n_query]
        phrases_with_count = []
        for c in range(len(phrase_query)):
            phrases_with_count.append(phrase_query[c][0] + str(c))

        for _, cluster in enumerate(max_box_id):
            # if len(cluster) > 1:
            similar_clusters[str(count)] = []
            if isinstance(cluster, list):
                for j in cluster:
                    similar_clusters[str(count)].append(str(phrases_with_count[j]))
            else:
                similar_clusters[str(count)].append(str(phrases_with_count[cluster]))
            count = count + 1

        gold = {"name": str(img_id), "type": "clusters", "clusters": similar_clusters}
        out_file = open(os.path.join(ref_dir, "test", str(img_id) + ".json"), "w")
        data = json.dump(gold, out_file, indent=4)
        out_file.close()

    return gold



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
   
    checkpoint = torch.load(str(args.save_name),
        map_location=device,
    )
    

    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        print("CUDA available")
        model = model.to(device)
    score = evaluate(test_loader, model, device, args)
    print("untrained eval score:", score)
