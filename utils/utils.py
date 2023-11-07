from __future__ import print_function
from cProfile import label
import torch.distributed as dist

import functools
import operator
import os
import sys
import re
import json
import numpy as np
import torch
import torch.nn as nn
import string


def largest(bboxes):
    maxS = 0
    use_gpu = torch.cuda.is_available()
    maxBox = torch.tensor([0, 0, 0, 0])
    if use_gpu:
        maxBox = maxBox.cuda()
    for box in bboxes:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        s = (right - left) * (bottom - top)
        if s > maxS:
            maxS = s
            maxBox = box
    return maxBox


def confidence(score, bboxes):
    maxIdx = np.argmax(score)
    return bboxes[maxIdx]


def union(bboxes):
    leftmin, topmin, rightmax, bottommax = 999, 999, 0, 0
    for box in bboxes:
        left, top, right, bottom = box
        if left == 0 and top == 0:
            continue
        leftmin, topmin, rightmax, bottommax = (
            min(left, leftmin),
            min(top, topmin),
            max(right, rightmax),
            max(bottom, bottommax),
        )

    return [leftmin, topmin, rightmax, bottommax]


def union_target(bboxes_list):
    target_box_list = []
    for boxes in bboxes_list:
        # boxes: [12, 5]
        target_box = union(boxes)  # target_box: [4]
        target_box_list.append(target_box)
    return target_box_list  # [query, 4]


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, "jpg")
    img_ids = set()
    for img in images:
        img_id = int(img.split("/")[-1].split(".")[0].split("_")[-1])
        img_ids.add(img_id)
    return img_ids


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print("%s is not initialized." % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def print_model(model, logger):
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    if logger:
        logger.write("nParams=\t" + str(nParams))


def save_model(path, model, epoch, optimizer=None):
    model_dict = {"epoch": epoch, "model_state": model.state_dict()}
    if optimizer is not None:
        model_dict["optimizer_state"] = optimizer.state_dict()

    torch.save(model_dict, path)


# Remove Flickr30K Entity annotations in a string
def remove_annotations(s):
    return re.sub(r"\[[^ ]+ ", "", s).replace("]", "")


def calculate_iou(obj1, obj2):
    EPS = 1e-6
    area1 = calculate_area(obj1)
    area2 = calculate_area(obj2)
    intersection = get_intersection(obj1, obj2)
    area_int = calculate_area(intersection)
    return area_int / ((area1 + area2 - area_int) + EPS)


def calculate_area(obj):
    return (obj[2] - obj[0]) * (obj[3] - obj[1])


def get_intersection(obj1, obj2):
    left = obj1[0] if obj1[0] > obj2[0] else obj2[0]
    top = obj1[1] if obj1[1] > obj2[1] else obj2[1]
    right = obj1[2] if obj1[2] < obj2[2] else obj2[2]
    bottom = obj1[3] if obj1[3] < obj2[3] else obj2[3]
    if left > right or top > bottom:
        return [0, 0, 0, 0]
    return [left, top, right, bottom]


def get_match_index(src_bboxes, dst_bboxes):
    indices = set()
    for src_bbox in src_bboxes:
        for i, dst_bbox in enumerate(dst_bboxes):
            iou = calculate_iou(src_bbox, dst_bbox)
            if iou >= 0.5:
                indices.add(i)  # match iou>0.5!!
    return list(indices)


def get_grounding_alignment(src_bboxes, dst_bboxes):
    alignment = torch.zeros((len(src_bboxes), len(dst_bboxes)))
    for i, src_bbox in enumerate(src_bboxes):
        for j, dst_bbox in enumerate(dst_bboxes):
            iou = calculate_iou(src_bbox, dst_bbox)
            # if iou >= 0.9:
            alignment[i][j] = iou  # match iou>0.5!!

    maxval, _ = alignment.max(dim=-1, keepdim=True)
    predictions = alignment == maxval  # [B, querys, K]
    return predictions


def bbox_is_match(src_bbox, dst_bboxes):
    for i, dst_bbox in enumerate(dst_bboxes):
        iou = calculate_iou(src_bbox, dst_bbox)
        if iou >= 0.5:
            return True
    return False


def unsupervised_get_match_index(src_bboxes, dst_bboxes):
    """
    src_bboxes: dict (for all entities)
    """
    indices = set()
    for entity, src_bboxes_list in src_bboxes.items():
        for src_bbox in src_bboxes_list:
            for i, dst_bbox in enumerate(dst_bboxes):
                iou = calculate_iou(src_bbox, dst_bbox)
                if iou >= 0.5:
                    indices.add(i)
    return list(indices)


# code for Parallel Processing


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args