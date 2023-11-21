import argparse
import os
import random
import warnings

import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import localized_narratives_pretrain_loader
from data import data_utils
from train import train_model
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

import warnings

warnings.filterwarnings("ignore", category=SourceChangeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--eval", action="store_true", help="evaluation mode")

    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--gpus", type=int, default=2, help="training epochs")
    parser.add_argument(
        "--save_name",
        type=str,
        default="models/model",
        help="name for saved model",
    )

    parser.add_argument(
        "--label-prop", action="store_true", help="use label propagation."
    )
    parser.add_argument("--use-ssl", action="store_true", help="use SSL.")
    parser.add_argument(
        "--use-phrase-mask", action="store_true", help="use phrase specific masking."
    )
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
    parser.add_argument(
        "--encoding-type",
        type=str,
        default="bert",
        help="which type of encodig to use",
    )

    parser.add_argument("--use-ema", action="store_true", help="use EMA model")
    parser.add_argument(
        "--bbox-reg", action="store_true", help="use bbox regression loss"
    )
    parser.add_argument(
        "--grounding", action="store_true", help="use grounding alignment loss"
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
        "--supervised", action="store_true", help="supervised training only"
    )
    parser.add_argument(
        "--adaptor-layers", action="store_true", help="train with adaptor layers"
    )
    parser.add_argument("--denoise", action="store_true", help="mouse denoising")
    parser.add_argument(
        "--unsupervised", action="store_true", help="test unsupervised accuracy"
    )
    parser.add_argument(
        "--model_config", type=str, help="path to model structure config json"
    )
    parser.add_argument("--sched", type=str, default="cosine", help="scheduler")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument("--decay_rate", type=float, default=1.0, help="decay rate")
    parser.add_argument("--warmup_lr", type=float, default=1e-5, help="warmup lr")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="warmup epochs")
    parser.add_argument(
        "--cooldown_epochs", type=int, default=0, help="cooldown epochs"
    )

    args = parser.parse_args()
    return args


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
    
    model = BertPretrain(
        text_encoder="bert-base-uncased",
        config=config,
        args=args,
    )
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    
    ## If loading weights from ALBEF
    
    # checkpoint = torch.load("saved/albef_pretrained/ALBEF_4M.pth", map_location="cpu")
    # state_dict = checkpoint["model"]
    # msg = model.load_state_dict(state_dict, strict=False)

    # print("load checkpoint from ALBEF")
    # print(msg)

    if torch.cuda.is_available():
        print("CUDA available")
        model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

        print(model)

    if args.test_set:
        test_dset = (
            localized_narratives_pretrain_loader.LocalizedNarrativesFlickr30dataset(
                dataroot="datasets/",
                image_features_type=args.image_features_type,
                word_embedding=wordEmbedding,
                split="test",
            )
        )
    else:
        test_dset = (
            localized_narratives_pretrain_loader.LocalizedNarrativesFlickr30dataset(
                dataroot="datasets/",
                image_features_type=args.image_features_type,
                word_embedding=wordEmbedding,
                split="val",
            )
        )

    train_dset = (
        localized_narratives_pretrain_loader.LocalizedNarrativesFlickr30dataset(
            dataroot="datasets/",
            image_features_type=args.image_features_type,
            word_embedding=wordEmbedding,
            split="train",
            ssl=False,
            sentence_patch_sim=False,
        )
    )
    ssl_dset = localized_narratives_pretrain_loader.LocalizedNarrativesFlickr30dataset(
        dataroot="datasets/",
        image_features_type=args.image_features_type,
        word_embedding=wordEmbedding,
        split="train",
        ssl=True,
        sentence_patch_sim=False,
    )

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        test_sampler = torch.utils.data.DistributedSampler(
            test_dset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        train_sampler = torch.utils.data.DistributedSampler(
            train_dset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        ssl_sampler = torch.utils.data.DistributedSampler(
            ssl_dset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        train_sampler = None
        test_sampler = None
        ssl_sampler = None

    test_loader = DataLoader(
        test_dset,
        batch_size=args.batch,
        num_workers=16,
        drop_last=True,
        sampler=test_sampler,
    )

    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch,
        num_workers=0,
        drop_last=True,
        sampler=train_sampler,
    )

    ssl_loader = DataLoader(
        ssl_dset,
        batch_size=args.batch,
        num_workers=0,
        drop_last=True,
        sampler=ssl_sampler,
    )

    train_model(
        model,
        ema_model,
        train_loader,
        test_loader,
        ssl_loader,
        device,
        args,
        lr=args.lr,
        epochs=args.epochs,
    )
    model_without_ddp = model.module if hasattr(model, "module") else model
    if args.use_ema:
        model_without_ddp = (
            ema_model.ema.module
            if hasattr(ema_model.ema, "module")
            else ema_model.ema
        )


    save_path = os.path.join("saved", args.save_name, "models" + ".pt")
    torch.save(model_without_ddp.state_dict(), save_path)
    print("save model to", save_path)
