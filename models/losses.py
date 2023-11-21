import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def sq_euclidean_dist(x1, x2):
    diff = x1 - x2
    diff_sq = diff * diff
    diff_sum = torch.sum(diff_sq, axis=-1)
    return diff_sum


def fro_norm(x1, x2):
    diff = x1 - x2
    fro_loss = torch.linalg.norm(diff, dim=1).mean()
    return fro_loss


def fro_norm_pos(x1, x2):
    pos_mask = (x2 > 0.0).float()
    diff = x1 - x2
    fro_loss = (torch.linalg.norm(diff, dim=1) * pos_mask).mean()
    return fro_loss


def smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
    """
    beta = 1
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

    # loss = loss * weight.unsqueeze(1)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features=None, labels=None, dist_matrix=None):
        # features: [B, N, D]
        # labels: [B, N, N]
        # calculate pairwise distance matrix [B, 1, N, D] [B, N, 1, D]
        # dist_matrix = torch.sum((features.unsqueeze(1) - features.unsqueeze(2)) ** 2, dim=3)
        if dist_matrix is None:
            dist_matrix = torch.cdist(features, features)

        # create mask of positive and negative pairs
        mask_pos = labels.eq(1).float()
        mask_neg = labels.eq(0).float()

        non_zero_mask = (torch.sum(labels, dim=1) > 0.0).float()

        # calculate loss for positive and negative pairs
        pos_loss = mask_pos * torch.clamp(self.margin - torch.sqrt(dist_matrix), min=0)

        pos_loss = torch.mean(pos_loss * non_zero_mask[:, None])
        neg_loss = mask_neg * torch.clamp(torch.sqrt(dist_matrix) - self.margin, min=0)
        neg_loss = torch.mean(neg_loss * non_zero_mask[:, None])
        # return total loss
        return pos_loss + neg_loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, features=None, labels=None, perm_indexes=None):
        features = F.normalize(features, p=2.0)
        if perm_indexes is not None:
            anchor = torch.index_select(
                features, 0, torch.tensor([perm_indexes]).to(features.device)
            )
            labels = torch.index_select(
                labels, 0, torch.tensor([perm_indexes]).to(features.device)
            )
            labels = labels.squeeze(0)
            positive_chains = torch.mean((features * labels[:, None]), dim=0)
            neg_labels = (~labels.bool()).float()

            negative_chains = torch.mean((features * neg_labels[:, None]), dim=0)

            distance_positive = self.calc_euclidean(anchor, positive_chains)
            distance_negative = self.calc_euclidean(anchor, negative_chains)
            losses = torch.relu(distance_positive - distance_negative + self.margin)

        else:
            anchor = features.to(features.device)
            labels = labels.to(features.device)
            features = features.unsqueeze(1).repeat(1, len(labels), 1)
            positive_chains = torch.mean((features * labels[:, :, None]), dim=1)
            neg_labels = (~labels.bool()).float()

            negative_chains = torch.mean((features * neg_labels[:, :, None]), dim=1)

            distance_positive = self.calc_euclidean(anchor, positive_chains)
            distance_negative = self.calc_euclidean(anchor, negative_chains)
            losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class HardTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(HardTripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, features=None, labels=None, perm_indexes=None):
        features = F.normalize(features, p=2.0)
        if perm_indexes is not None:
            anchor = torch.index_select(
                features, 0, torch.tensor([perm_indexes]).to(features.device)
            )
            labels = torch.index_select(
                labels, 0, torch.tensor([perm_indexes]).to(features.device)
            )
        else:
            anchor = features
        random_positives_idx = []

        pos_indices = [k for k in range(len(labels[0])) if labels[0][k] == 1.0]
        try:
            random_positives_idx.append(random.choice(pos_indices))
        except:
            print(pos_indices)
            random_positives_idx.append(0)

        positives = torch.index_select(
            features, 0, torch.tensor(random_positives_idx).to(features.device)
        )

        neg_labels = (~labels.bool()).float()
        random_negatives_idx = []
        neg_indices = [k for k in range(len(neg_labels[0])) if neg_labels[0][k] == 1.0]
        try:
            random_negatives_idx.append(random.choice(neg_indices))
        except:
            print(neg_indices)
            random_negatives_idx.append(0)

        negatives = torch.index_select(
            features, 0, torch.tensor(random_negatives_idx).to(features.device)
        )

        distance_positive = self.calc_euclidean(anchor, positives)
        distance_negative = self.calc_euclidean(anchor, negatives)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


def bce_loss(features, target):
    features = F.normalize(features, p=2.0)
    input = torch.matmul(features, features.t())
    max_val = (-input).clamp(min=0)
    loss = (
        input
        - input * target
        + max_val
        + ((-max_val).exp() + (-input - max_val).exp()).log()
    )

    return loss.mean()


def ce_loss(pred, target):
    # softmax pred
    loss = -(target * (pred + 1e-9).log()).mean()

    return loss



class TripletLossGrounding(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLossGrounding, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, phrase_features=None, image_features=None, labels=None):
        phrase_features = F.normalize(phrase_features, p=2.0)
        image_features = F.normalize(image_features, p=2.0)

        anchor = phrase_features
        image_features = image_features.repeat(len(phrase_features), 1, 1)
        positives = torch.mean((image_features * labels[:, :, None]), dim=-2)
        neg_labels = (~labels.bool()).float()
        random_negatives_idx = []
        for b in range(len(phrase_features)):
            neg_indices = [
                k for k in range(len(neg_labels[b])) if neg_labels[b][k] == 1.0
            ]
            try:
                random_negatives_idx.append(random.choice(neg_indices))
            except:
                print(neg_indices)
                random_negatives_idx.append(0)

        negatives = [
            torch.index_select(
                image_features[b],
                0,
                torch.tensor([random_negatives_idx[b]]).to(image_features.device),
            )
            for b in range(len(phrase_features))
        ]
        negatives = torch.stack(negatives).squeeze(-2)
        distance_positive = self.calc_euclidean(anchor, positives)
        distance_negative = self.calc_euclidean(anchor, negatives)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class SupContrastiveLoss(nn.Module):
    def __init__(self, temp):
        super(SupContrastiveLoss, self).__init__()
        self.temperature = temp

    def small_val(self, dtype):
        return torch.finfo(dtype).tiny

    def neg_inf(self, dtype):
        return torch.finfo(dtype).min

    def logsumexp(self, x, keep_mask=None, add_one=True, dim=1):
        if keep_mask is not None:
            x = x.masked_fill(~keep_mask, self.neg_inf(x.dtype))
        if add_one:
            zeros = torch.zeros(
                x.size(dim - 1), dtype=x.dtype, device=x.device
            ).unsqueeze(dim)
            x = torch.cat([x, zeros], dim=dim)

        output = torch.logsumexp(x, dim=dim, keepdim=True)
        if keep_mask is not None:
            output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        return output

    def forward(self, features=None, labels=None, perm_indexes=None):
        # features: [B, N, D]
        # labels: [B, N, N]
        # calculate pairwise distance matrix [B, 1, N, D] [B, N, 1, D]
        features = F.normalize(features, p=2.0)
        mat = torch.matmul(features, features.t())
        if perm_indexes is not None:
            mat = torch.index_select(
                mat, 0, torch.tensor([perm_indexes]).to(mat.device)
            )
            labels = torch.index_select(
                labels, 0, torch.tensor([perm_indexes]).to(mat.device)
            )
        # create mask of positive and negative pairs
        pos_mask = labels.eq(1).float()
        neg_mask = labels.eq(0).float()
        mat = mat / self.temperature
        mat_max, _ = mat.max(dim=1, keepdim=True)
        mat = mat - mat_max.detach()  # for numerical stability

        denominator = self.logsumexp(
            mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
        )
        log_prob = mat - denominator
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
            pos_mask.sum(dim=1) + self.small_val(mat.dtype)
        )

        # return total loss
        return (-mean_log_prob_pos).mean()
