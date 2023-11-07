from ast import Num
import time
import warnings
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import losses
from utils import utils
from utils.evaluator import Evaluator
from utils.utils import union_target, AttrDict
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from scheduler.warmup_scheduler import GradualWarmupScheduler
import pickle

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

ceLoss = nn.CrossEntropyLoss(reduction="mean")
contrastive_loss = losses.ContrastiveLoss(1.0)
sup_contrastive_loss = losses.SupContrastiveLoss(0.1)
sup_contrastive_loss_unlabeled = losses.TripletLoss(1.0)
cluster_triplet_loss = losses.ClusteringTripletLoss(1.0)
hard_triplet_loss = losses.HardTripletLoss(1.0)

triplet_loss_grounding = losses.TripletLossGrounding(1.0)


def ssl_train(model, ssl_loader, ssl_train_iter, device, optimizer, args, lr=1e-4):
    use_gpu = torch.cuda.is_available()
    model = model.float()
    total_loss = 0
    n_batches = 0

    (
        image_id,
        _,
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
        gt_coref_matrix,
        _,
    ) = next(ssl_train_iter)
    if use_gpu:
        (
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
            gt_coref_matrix,
        ) = (
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
            gt_coref_matrix.to(device),
        )

    n_batches += 1
    
    caption_input_ids = caption_input_ids.squeeze(1)
    caption_attn_masks = caption_attn_masks.squeeze(1)
    mouse_trace_for_phrases = mouse_trace_for_phrases.squeeze(-2)
    phrase_queries_input_ids = phrase_queries_input_ids.squeeze(2)
    phrase_queries_attention_mask = phrase_queries_attention_mask.squeeze(2)
    object_label_input_ids = object_label_input_ids.squeeze(-2)
    object_label_attention_mask = object_label_attention_mask.squeeze(-2)
    object_label_input_ids = object_label_input_ids.view(-1, 16)
    object_label_attention_mask = object_label_attention_mask.view(-1, 16)
    
    (
        loss_mlm,
        grounding_matrix,
        pred_coref_matrix,
        weighted_phrase_embedding,
        _,
        _,
        _,
        target,
        probs,
        _,
        _,
        _,
        _,
        modified_pred_bboxes,
    ) = model.module.forward(
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
        max_assignments=None,
        train=False,
    )
    target_pred = torch.argmax(target, dim=1)  # [B]
    
    if args.ssl_loss == "fro":
        batch_reg_loss = [
            losses.fro_norm(pred_coref_matrix[b], gt_coref_matrix[b])
            for b in range(len(gt_coref_matrix))
        ]
        reg_loss = torch.stack(batch_reg_loss).mean()
    elif args.ssl_loss == "sigmoid":
        batch_reg_loss = [
            losses.bce_loss(
                weighted_phrase_embedding[b][: num_query[b], :],
                gt_coref_matrix[b][: num_query[b], : num_query[b]],
            )
            for b in range(len(gt_coref_matrix))
        ]
        reg_loss = torch.stack(batch_reg_loss).mean()
    else:
        batch_reg_loss = [
            sup_contrastive_loss(
                weighted_phrase_embedding[b][: num_query[b], :],
                gt_coref_matrix[b][: num_query[b], : num_query[b]],
            )
            for b in range(len(gt_coref_matrix))
        ]
        reg_loss = torch.stack(batch_reg_loss).mean()

    if args.bbox_reg:
        bbox_reg_loss = losses.smooth_l1_loss(
            modified_pred_bboxes, target_bboxes.view(-1, 4).squeeze(-2)
        )
        ssl_loss = reg_loss + bbox_reg_loss + loss_mlm + ceLoss(probs, target_pred)
    if args.grounding:
        target_bboxes = target_bboxes.squeeze(-2)
        alignment_loss = []
        for b in range(len(object_regions)):
            gnd_alignment = utils.get_grounding_alignment(
                target_bboxes[b][: num_query[b], :],
                object_regions[b][: num_objects[b], :4],
            )
           
            batch_alignment = (gnd_alignment).long().to(device)
            alignment_loss.append(
                losses.ce_loss(
                    grounding_matrix[b][: num_query[b], : num_objects[b]],
                    batch_alignment,
                )
            )
        alignment_loss = torch.stack(alignment_loss).mean()
        if args.bbox_reg:
            ssl_loss = (
                reg_loss
                + bbox_reg_loss
                + loss_mlm
                + ceLoss(probs, target_pred)
                + alignment_loss
            )
        else:
            ssl_loss = reg_loss + loss_mlm + ceLoss(probs, target_pred) + alignment_loss

    else:
        ssl_loss = reg_loss + loss_mlm + ceLoss(probs, target_pred)

    total_loss += ssl_loss
    return ssl_loss


def pretrain(
    model,
    ema_model,
    train_loader,
    test_loader,
    ssl_loader,
    device,
    args,
    lr=1e-4,
    epochs=25,
):
    use_gpu = torch.cuda.is_available()
    model = model.float()
   

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
   
    scheduler_steplr = StepLR(optimizer, step_size=10, gamma=0.95)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=2, after_scheduler=scheduler_steplr
    )
    
    for epoch in range(epochs):
        scheduler_warmup.step(epoch)
        t = time.time()
        correct_preds = 0
        all_preds = 0

        total_loss = 0
        n_batches = 0
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            ssl_loader.sampler.set_epoch(epoch)

        if args.use_ssl:
            ssl_iter = iter(ssl_loader)

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
            gt_coref_matrix,
            rule_coref_matrix,
        ) in tqdm(train_loader):
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
                    gt_coref_matrix,
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
                    gt_coref_matrix.to(device),
                    rule_coref_matrix.to(device),
                )
            model.train(True)
            optimizer.zero_grad()
            mouse_trace_for_phrases = mouse_trace_for_phrases.squeeze(-2)

            if args.use_ssl:
                try:
                    ssl_loss = ssl_train(
                        model, ssl_loader, ssl_iter, device, optimizer, args
                    )
                except:
                    ssl_iter = iter(ssl_loader)
                    ssl_loss = ssl_train(
                        model, ssl_loader, ssl_iter, device, optimizer, args
                    )
            n_batches += 1

            caption_input_ids = caption_input_ids.squeeze(1)
            caption_attn_masks = caption_attn_masks.squeeze(1)
            phrase_queries_input_ids = phrase_queries_input_ids.squeeze(2)
            phrase_queries_attention_mask = phrase_queries_attention_mask.squeeze(2)
          
            object_label_input_ids = object_label_input_ids.squeeze(-2)
            object_label_attention_mask = object_label_attention_mask.squeeze(-2)
            object_label_input_ids = object_label_input_ids.view(-1, 16)
            object_label_attention_mask = object_label_attention_mask.view(-1, 16)
            if args.label_prop:
                with torch.no_grad():
                    (
                        _,
                        gt_grounding_matrix,
                        pred_gt_coref_matrix,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        gt_predictions,
                        _,
                        _,
                        modified_pred_boxes,
                    ) = model.module.forward(
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
                        max_assignments=None,
                        train=False,
                    )
            (
                loss_mlm,
                grounding_matrix,
                pred_coref_matrix,
                weighted_phrase_embedding,
                image_encoding,
                phrase_embeddings,
                matrix_logits_reg,
                target,
                probs,
                _,
                predictions,
                _,
                _,
                _,
            ) = model.module.forward(
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
            target_pred = torch.argmax(target, dim=1)  # [B]
            prediction = torch.argmax(probs, dim=1)  # [all_querys]
            correct_preds += int(prediction.eq(target_pred).sum())
            all_preds += len(prediction)
            
            if args.label_prop:
                if args.ssl_loss == "sigmoid":
                    gt_coref_matrix = (pred_gt_coref_matrix > 0.5).float()
                    batch_reg_loss = [
                        losses.bce_loss(
                            weighted_phrase_embedding[b][: num_query[b], :],
                            gt_coref_matrix[b][: num_query[b], : num_query[b]],
                        )
                        for b in range(len(gt_coref_matrix))
                    ]
                    reg_loss = torch.stack(batch_reg_loss).mean()
                else:
                    gt_coref_matrix = (pred_gt_coref_matrix > 0.9).float()
                    batch_reg_loss = [
                        sup_contrastive_loss_unlabeled(
                            weighted_phrase_embedding[b][: num_query[b], :],
                            gt_coref_matrix[b][: num_query[b], : num_query[b]],
                        )
                        for b in range(len(gt_coref_matrix))
                    ]
                    
                    batch_reg_loss = torch.stack(batch_reg_loss)
                    reg_loss = batch_reg_loss.mean()
                if args.grounding:
                    gt_grounding_matrix = gt_grounding_matrix.float()
                    maxval, _ = gt_grounding_matrix.max(dim=-1, keepdim=True)

                    gt_grounding_matrix = (
                        gt_grounding_matrix == maxval
                    )  # [B, querys, K]
                    gt_grounding_matrix = gt_grounding_matrix.long()
                    batch_reg_g_loss = []
                    for b in range(len(gt_grounding_matrix)):
                        b_gt_grounding_matrix = []
                        b_grounding_matrix = []
                        for q in range(num_query[b]):
                            if maxval[b][q][0] > 0.5:
                                b_gt_grounding_matrix.append(
                                    gt_grounding_matrix[b][q]
                                )
                                b_grounding_matrix.append(
                                    grounding_matrix[b][q]
                                )
                        if len(b_grounding_matrix) > 0:
                            b_grounding_matrix = torch.stack(
                                b_grounding_matrix
                            ).to(device)
                            b_gt_grounding_matrix = torch.stack(
                                b_gt_grounding_matrix
                            ).to(device)
                            reg_g_loss = losses.ce_loss(
                                b_grounding_matrix, b_gt_grounding_matrix
                            )
                            batch_reg_g_loss.append(reg_g_loss)
                    if len(batch_reg_g_loss) > 0:
                        batch_reg_g_loss = torch.stack(batch_reg_g_loss)
                        reg_loss = (
                            batch_reg_loss.mean() + (batch_reg_g_loss).mean()
                        )
                loss = ceLoss(probs, target_pred) + reg_loss

            else:
                loss = ceLoss(probs, target_pred)
            if args.use_ssl:
                loss = loss + ssl_loss + loss_mlm
            total_loss += loss

            loss.backward()
            optimizer.step()

            if args.use_ema:
                ema_model.update(model)
            
        
        total_loss = total_loss.item() / n_batches
        t1 = time.time()
        print("--- EPOCH", epoch)
        print("--- LR:", optimizer.param_groups[0]["lr"])
        print("     time:", t1 - t)
        print("     total loss:", total_loss)
        print(
            "     supervised accuracy on training set: ",
            correct_preds / all_preds,
        )

        save_path = os.path.join(
            "saved", args.save_name, "models" + "_" + str(epoch) + ".pt"
        )
        model_to_save = model.module if hasattr(model, "module") else model
        if args.use_ema:
            model_to_save = (
                ema_model.ema.module
                if hasattr(ema_model.ema, "module")
                else ema_model.ema
            )
        torch.save(model_to_save.state_dict(), save_path)
        
