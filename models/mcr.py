"""
Image Text Tranformer Model
"""

from functools import partial
from models.xbert import BertConfig, BertForMaskedLM
from transformers import BertTokenizer

from models.bert_layers import BertLayer
from models import bbox_regression
from models import hungarian

from models.losses import ContrastiveLoss, fro_norm
import torch
import torch.nn.functional as F
from torch import nn
import copy
import numpy as np
import random
import math



class ImageEmbedder(nn.Module):
    def __init__(self, config, img_dim, obj_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config["hidden_size"])
        self.obj_label_linear = nn.Linear(obj_dim, config["hidden_size"])
        self.img_layer_norm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.obj_layer_norm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.pos_layer_norm = nn.LayerNorm(config["hidden_size"], eps=1e-12)

        self.image_xmin = nn.Linear(1, 16)
        self.image_ymin = nn.Linear(1, 16)
        self.image_xmax = nn.Linear(1, 16)
        self.image_ymax = nn.Linear(1, 16)
        self.image_area = nn.Linear(1, 16)
        self.pos_linear = nn.Sequential(
            nn.Linear(16 * 5, 128),
            # nn.LayerNorm(512, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config["hidden_size"]),
        )
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, img_feat, img_pos_feat, object_label_embeds, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_obj = self.obj_layer_norm(
            self.obj_label_linear(object_label_embeds)
        )
        img_pos_feat = img_pos_feat.float()
        loc_emb_xmin = self.image_xmin(img_pos_feat[:, :, 0].unsqueeze(-1))
        loc_emb_ymin = self.image_ymin(img_pos_feat[:, :, 1].unsqueeze(-1))
        loc_emb_xmax = self.image_xmax(img_pos_feat[:, :, 2].unsqueeze(-1))
        loc_emb_ymax = self.image_ymax(img_pos_feat[:, :, 3].unsqueeze(-1))
        loc_emb_area = self.image_area(img_pos_feat[:, :, 4].unsqueeze(-1))
        loc_emb_cat = torch.cat(
            (loc_emb_xmin, loc_emb_ymin, loc_emb_xmax, loc_emb_ymax, loc_emb_area), -1
        )
        transformed_pos = self.pos_layer_norm(self.pos_linear(loc_emb_cat))
        embeddings = transformed_im + transformed_obj + transformed_pos
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config["num_hidden_layers"])]
        )
        self.pos_encoder = PositionalEncoding(config["embed_dim"], max_len=100)

    def forward(self, input_, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        input_ = self.pos_encoder(input_)
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPretrain(nn.Module):
    def __init__(self, text_encoder=None, config=None, args=None, temp=0.07):
        super().__init__()
        self.config = config
        self.mlm_probability = config["mlm_probability"]
        embed_dim = config["embed_dim"]
        self.max_num_queries = config["max_num_queries"]
        self.max_phrase_length = config["max_phrase_length"]
        bert_config = BertConfig.from_json_file(config["bert_config"])
        self.tokenizer = BertTokenizer.from_pretrained(text_encoder)

        self.text_encoder = BertForMaskedLM.from_pretrained(
            text_encoder, config=bert_config
        )
        text_width = self.text_encoder.config.hidden_size
        self.softmax = nn.Softmax(dim=-1)
        self.visual_embedder = ImageEmbedder(config, config["img_dim"], text_width)
        vision_width = config["vision_width"]
        self.s2v = True
        if self.s2v:
            self.new_text_proj = nn.Linear(text_width + 128, embed_dim)
        else:
            self.new_text_proj = nn.Linear(text_width, embed_dim)

        self.visual_encoder = ImageEncoder(config)

        self.bbox_reg = args.bbox_reg
        self.transformation_func = args.trans_func
        self.use_ssl = args.use_ssl
        self.use_phrase_mask = args.use_phrase_mask
        self.adaptor_layers = args.adaptor_layers
        self.ssl_loss = args.ssl_loss
        if self.adaptor_layers:
            self.linear_phrase_adaptor = nn.Linear(
                config["hidden_size"], config["hidden_size"]
            )
            self.linear_multimodal_adaptor = nn.Linear(
                config["hidden_size"], config["hidden_size"]
            )
        if self.bbox_reg:
            self.weights = (10.0, 10.0, 5.0, 5.0)
            self.bbox_reg_fc = nn.Sequential(
                nn.Linear(embed_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 4)
            )
            self.bbox_transform = bbox_regression.Box2BoxTransform(self.weights)

    def get_phrase_embeddings(self, q_start_ind, sentence_embedding, num_query):
        batch_len = q_start_ind.size(0)
        phrase_word_embs = torch.zeros(
            q_start_ind.size(0),
            self.max_num_queries,
            self.max_phrase_length,
            sentence_embedding.shape[-1],
        ).cuda()
        # create a mask for the phrases embeddings
        phrase_mask = (
            torch.zeros(
                q_start_ind.size(0), self.max_num_queries, self.max_phrase_length
            )
            .cuda()
            .float()
        )

        for batch in range(batch_len):
            num_queries = num_query[batch]

            for n_q in range(int(num_queries)):
                phr_len = (
                    int((q_start_ind[batch][n_q][1]))
                    - int((q_start_ind[batch][n_q][0]))
                    + 1
                )
                start_ind = int(q_start_ind[batch][n_q][0])
                end_ind = int(q_start_ind[batch][n_q][1])
                if phr_len <= self.max_phrase_length:
                    phrase_word_embs[batch, n_q, :phr_len, :] += sentence_embedding[
                        batch
                    ][start_ind : end_ind + 1]
                    phrase_mask[batch, n_q, :phr_len] = 1.0
                else:
                    print("skipping. phrase length of {} too long".format(phr_len))

        phrase_word_embs = phrase_word_embs * phrase_mask[:, :, :, None]

        return phrase_word_embs

    def cosine_similarity_matrix(self, matrix):
        # Normalize the matrix along the last dimension
        norm = torch.norm(matrix, p=2, dim=-1, keepdim=True)
        matrix = matrix / norm
        # Compute the dot product of the matrix with its transpose
        dot_prod = torch.matmul(matrix, matrix.transpose(-1, -2))
        return dot_prod

    def sigmoid_similarity_matrix(self, matrix):
        # Compute the dot product of the matrix with its transpose
        dot_prod = torch.matmul(matrix, matrix.transpose(-1, -2))
        return F.sigmoid(dot_prod)

    def random_mask(
        self,
        input_ids,
        vocab_size,
        device,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            # We only compute loss on masked tokens
            targets[~masked_indices] = -100
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(
            device
        )
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def forward(
        self,
        image_features,
        object_label_input_ids,
        object_label_attention_mask,
        object_regions,
        text_input_ids,
        text_attention_mask,
        sense2vec_feats,
        q_start_ind,
        num_objects=None,
        num_query=None,
        mouse_trace_feats=None,
        num_sentences=None,
        clip_img_embeds=None,
        sentence_input_ids=None,
        sentence_attn_mask=None,
        sense2vec_sentence_feats=None,
        patch_sentence_sim=None,
        mouse_trace_for_phrases=None,
        phrase_queries_input_ids=None,
        max_assignments=None,
        train=False,
        alpha=0,
    ):
        text_output = self.text_encoder.bert(
            text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state

        ##================= MLM ========================##
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        input_ids, labels = self.random_mask(
            input_ids,
            self.text_encoder.config.vocab_size,
            text_input_ids.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )
        mlm_output = self.text_encoder(
            input_ids,
            attention_mask=text_attention_mask,
            return_dict=True,
            labels=labels,
            mode="text",
            alpha=alpha,
        )
        loss_mlm = mlm_output.loss

        if self.s2v:
            text_embeds = self.new_text_proj(
                torch.cat((text_embeds, sense2vec_feats), dim=-1)
            )
        else:
            text_embeds = self.new_text_proj(text_embeds)

        # Encode the test embedings using text encoder

        phrase_embeddings = self.get_phrase_embeddings(
            q_start_ind, text_embeds, num_query
        )

        ########################################################
        ##================= Image-Text Contrastive Loss ========================##

        object_label_output = self.text_encoder.bert(
            object_label_input_ids,
            attention_mask=object_label_attention_mask,
            return_dict=True,
            mode="text",
        )
        object_label_embeds = object_label_output.last_hidden_state
        object_label_embeds = torch.mean(object_label_embeds, dim=-2)
        object_label_embeds = object_label_embeds.view(
            image_features.shape[0], image_features.shape[1], -1
        )

        image_embeds = self.visual_embedder(
            image_features, object_regions, object_label_embeds
        )

        image_mask = torch.zeros(image_embeds.shape[0], image_embeds.shape[1]).cuda()
        for b in range(len(image_mask)):
            image_mask[b, : int(num_objects[b])] = 1
        mm_image_attention_mask = image_mask

        query_mask = torch.zeros(
            phrase_embeddings.shape[0], phrase_embeddings.shape[1]
        ).cuda()
        for b in range(len(query_mask)):
            query_mask[b, : int(num_query[b])] = 1
        query_attention_mask = query_mask
        image_mask = torch.zeros(image_embeds.shape[0], image_embeds.shape[1]).cuda()
        for b in range(len(image_mask)):
            image_mask[b, : int(num_objects[b])] = 1
        image_attention_mask = image_mask.unsqueeze(1).unsqueeze(-1)
        image_attention_mask = image_attention_mask.repeat(
            1, self.config["num_attention_heads"], 1, image_mask.shape[-1]
        )
        image_attention_mask = image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        image_attention_mask = (1.0 - image_attention_mask) * -10000.0
        # Encode the image features + location using a sequence of encoder layers
        image_encoding = self.visual_encoder(
            image_embeds,
            image_attention_mask,
            output_all_encoded_layers=False,
        )
        image_encoding = image_encoding[-1]

        phrase_embeddings = torch.mean(phrase_embeddings, dim=-2)
        if self.adaptor_layers:
            phrase_embeddings = self.linear_phrase_adaptor(phrase_embeddings)
        # [B, K, dim] x [B, querys, dim] => [B, querys, K]
        i_att = torch.einsum("bkd, byd -> byk", image_encoding, phrase_embeddings)

        output_pos = self.text_encoder.bert(
            encoder_embeds=phrase_embeddings,
            attention_mask=query_attention_mask,
            encoder_hidden_states=image_encoding,
            encoder_attention_mask=mm_image_attention_mask,
            output_attentions=True,
            return_dict=True,
            mode="fusion",
        )
        prediction = torch.argmax(i_att, dim=-1)  # [B, querys]
        attmap = torch.einsum("avd, bqd -> baqv", image_encoding, phrase_embeddings)  #
        grounding_matrix = output_pos.cross_attentions[-1][
            :, -1, :, :
        ]  # [B, querys, K]
        grounding_matrix = grounding_matrix.view(-1, self.max_num_queries, 100)
        # [B1, B2, querys]: B1th sentence to B2th image
        maxatt, _ = attmap.max(dim=-1)
        logits = torch.sum(maxatt, dim=-1).div(
            num_query.unsqueeze(1).expand(maxatt.size(0), maxatt.size(1))
        )  # [B1, B2]: B1th sentence to B2th image

        n_obj = int(q_start_ind.size(0))
        target = torch.eye(n_obj).cuda()  # [b, b]

        # return predicted boxes for grounding
        maxval, _ = i_att.max(dim=2, keepdim=True)
        predictions = i_att == maxval  # [B, querys, K]
        query_mask = torch.zeros(image_encoding.shape[0], self.max_num_queries).cuda()
        query_mask_reg = torch.zeros(
            image_encoding.shape[0], self.max_num_queries, self.max_num_queries
        ).cuda()
        if self.adaptor_layers:
            weighted_phrase_embedding = self.linear_multimodal_adaptor(
                output_pos.last_hidden_state
            )
        else:
            weighted_phrase_embedding = output_pos.last_hidden_state
            # weighted_phrase_embedding = phrase_embeddings
        weighted_phrase_embedding = weighted_phrase_embedding.view(
            -1, self.max_num_queries, weighted_phrase_embedding.shape[-1]
        )
        if self.ssl_loss == "sigmoid":
            pred_coref_matrix = self.sigmoid_similarity_matrix(
                weighted_phrase_embedding
            )
        else:
            pred_coref_matrix = self.cosine_similarity_matrix(weighted_phrase_embedding)
        if train:
            if max_assignments is None:
                max_assignments = []
                gt_coref_matrix = (
                    (pred_coref_matrix > 0.99).float().clone().cpu().detach().numpy()
                )
                for i in range(len(gt_coref_matrix)):
                    gt_batch = gt_coref_matrix[i]
                    num_tasks = []
                    for k in range(len(gt_batch)):
                        num_tasks.append(np.count_nonzero(gt_batch[k]))
                    max_assignments.append(num_tasks)
            predictions = self.get_hungarian_assignment(i_att, max_assignments)
        else:
            predictions = i_att == maxval  # [B, querys, K]
        for b in range(len(query_mask_reg)):
            query_mask[b, : int(num_query[b])] = 1.0
            query_mask_reg[b, : int(num_query[b]), : int(num_query[b])] = 1.0
        matrix_logits_reg = torch.matmul(
            predictions.float(), predictions.permute(0, 2, 1).float()
        )
        matrix_logits_reg = pred_coref_matrix * query_mask_reg

        pred_bboxes = []
        for i in range(predictions.size(0)):
            # predictions[i]:querys x K
            select_box = (
                object_regions[i]
                .unsqueeze(0)
                .expand(predictions.size(1), predictions.size(2), 5)
                .long()
            )  # querys x K x 5
            select_mask = predictions[i].unsqueeze(-1).long()  # querys x K x 1

            avail_box = select_box * select_mask  # # querys x K x 5
            _, maxidx = avail_box.max(dim=1)  # querys x 5

            # maxidx[:, -1] -- [querys] location of the max bbox
            bbox = select_box[torch.arange(select_box.size(0)), maxidx[:, -1]]
            pred_bboxes.append(bbox[:, :4])

       
        pred_bboxes = torch.stack(pred_bboxes)  # [B, querys, 4]
        if self.bbox_reg:
            pred_bbox_deltas = self.bbox_reg_fc(weighted_phrase_embedding)

            modified_pred_bboxes = self.bbox_transform.apply_deltas(
                pred_bbox_deltas.view(-1, 4), pred_bboxes.view(-1, 4)
            )
            modified_pred_bboxes = modified_pred_bboxes * query_mask.view(-1)[:, None]
        else:
            modified_pred_bboxes = pred_bboxes


        return (
            loss_mlm,
            grounding_matrix,
            pred_coref_matrix,
            weighted_phrase_embedding,
            image_encoding,
            phrase_embeddings,
            matrix_logits_reg,
            target,
            logits,
            text_embeds,
            predictions,
            prediction,
            pred_bboxes,
            modified_pred_bboxes,
        )
