from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import spacy


import json
import h5py
import os
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import re
from PIL import Image
from data import data_utils

from data import patch_sentence_similarity_clip
from utils import utils
from transformers import BertTokenizer

NoneType = type(None)
bert_version = "bert-base-uncased"

nlp = spacy.load("en_core_web_sm")
s2v = nlp.add_pipe("sense2vec")
s2v.from_disk(
    "/disk/nfs/gazinasvolume1/s1985335/coreference-and-grounding/datasets/s2v_old"
)
## load pretrained CLIP model
# from transformers import CLIPProcessor, CLIPModel

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

stop_words = [
    "we",
    "front",
    "the foreground",
    "the front",
    "the image",
    "the right",
    "the bottom",
    "the right side",
    "something",
    "the middle",
    "the left side",
    "the left",
    "the center",
    "something",
    "this picture",
    "this image",
    "the background",
    "i",
    "bottom",
    "left",
    "right",
]


def get_captions(ix, label_start_ix, label_end_ix, label, seq_per_img=5):
    # fetch the sequence labels
    ix1 = label_start_ix[ix] - 1  # label_start_ix starts from 1
    ix2 = label_end_ix[ix] - 1
    ncap = ix2 - ix1 + 1  # number of captions available for this image
    seq_length = 256
    if ncap == 0:
        seq = np.zeros([seq_per_img, seq_length], dtype="int")
    # assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'
    else:
        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, seq_length], dtype="int")
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                cap_len = label.shape[1]
                seq[q, :cap_len] = label[ixl, :cap_len]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = label[ixl : ixl + seq_per_img, :]

    return seq


class LocalizedNarrativesFlickr30dataset(Dataset):
    def __init__(
        self,
        dataroot="datasets/",
        image_features_type="faster_rcnn",
        word_embedding=None,
        split="train",
        ssl=False,
        sentence_patch_sim=False,
        eval_grounding=False,
    ):
        self.split = split
        self.ssl = ssl
        self.max_detected_boxes = 100
        self.max_queries = 128
        self.max_query_length = 16
        self.max_caption_length = 512
        self.dataroot = dataroot
        self.sentence_patch_sim = sentence_patch_sim
        self.eval_grounding= eval_grounding
        self.encoding_type = "bert"
        self.coco_data = False
        # Load image id to idx mapping for loading visual features
        self.imgid2idx = pickle.load(
            open(
                os.path.join(
                    dataroot,
                    "%s_image_features" % image_features_type,
                    "%s_imgid2idx.pkl" % split,
                ),
                "rb",
            )
        )
        self.val_imgid2idx = pickle.load(
            open(
                os.path.join(
                    dataroot,
                    "%s_image_features" % image_features_type,
                    "%s_imgid2idx.pkl" % "val",
                ),
                "rb",
            )
        )

        if self.coco_data:
            coco_h5_features = os.path.join(
                dataroot, "coco_bottom_up_features", "features.hdf5"
            )
            coco_h5_boxes = os.path.join(
                dataroot, "coco_bottom_up_features", "boxes.hdf5"
            )
            coco_h5_classes = os.path.join(
                dataroot, "coco_bottom_up_features", "classes.hdf5"
            )

            self.coco_features = h5py.File(coco_h5_features, "r")
            self.coco_pos_boxes = h5py.File(coco_h5_boxes, "r")
            self.coco_h5_classes = h5py.File(coco_h5_classes, "r")
            self.coco_img_ids = np.array(self.coco_h5_classes)
            self.coco_int_img_ids = [
                int(self.coco_img_ids[i]) for i in range(len(self.coco_img_ids))
            ]

            self.vg_classes = []
            with open(
                os.path.join(dataroot, "coco_bottom_up_features", "vg_object_vocab.txt")
            ) as f:
                for object in f.readlines():
                    self.vg_classes.append(object.split(",")[0].lower().strip())

        if self.split == "train":
            self.all_ids = list(self.imgid2idx.keys())
            # self.train_image_ids = np.random.choice(
            #     self.all_ids, int(0.2 * len(self.all_ids)), False
            # ).tolist()

            self.train_image_ids = self.all_ids
            self.val_image_ids = json.load(
                open(
                    os.path.join(
                        dataroot,
                        "cin_annotations",
                        "%s_image_ids.json" % "val",
                    ),
                    "rb",
                )
            )
            if self.ssl:
                # self.image_ids = np.random.choice(self.val_image_ids, int(0.2 * len(self.val_image_ids)), False).tolist()
                self.image_ids = self.val_image_ids
            else:
                self.image_ids = self.train_image_ids
                if self.coco_data:
                    for x in self.coco_int_img_ids:
                        self.image_ids.append(x)

        else:
            self.image_ids = json.load(
                open(
                    os.path.join(
                        dataroot,
                        "cin_annotations",
                        "%s_image_ids.json" % split,
                    ),
                    "rb",
                )
            )
            self.train_image_ids = self.image_ids
        # Load visual features based on image_features_type
        h5_path = os.path.join(
            dataroot,
            "%s_image_features" % image_features_type,
            "%s_features_compress.hdf5" % split,
        )

        with h5py.File(h5_path, "r") as hf:
            self.features = np.array(hf.get("features"))
            self.pos_boxes = np.array(hf.get("pos_bboxes"))

        # Load detected boxes, labels and attributes
        self.obj_detection_dict = os.path.join(
            dataroot,
            "%s_image_features" % image_features_type,
            "%s_detection_dict.json" % split,
        )
        self.obj_detection_dict = json.load(open(self.obj_detection_dict, "rb"))

        ### for Validation data
        # Load visual features based on image_features_type
        val_h5_path = os.path.join(
            dataroot,
            "%s_image_features" % image_features_type,
            "%s_features_compress.hdf5" % "val",
        )

        with h5py.File(val_h5_path, "r") as hf:
            self.val_features = np.array(hf.get("features"))
            self.val_pos_boxes = np.array(hf.get("pos_bboxes"))

        # Load detected boxes, labels and attributes
        self.val_obj_detection_dict = os.path.join(
            dataroot,
            "%s_image_features" % image_features_type,
            "%s_detection_dict.json" % "val",
        )
        self.val_obj_detection_dict = json.load(open(self.val_obj_detection_dict, "rb"))

        # caption vocabulary - 4421 unique words
        self.vocab = json.load(open(os.path.join(dataroot, "flk30k_LN.json"), "rb"))[
            "ix_to_word"
        ]

        self.imagesid_from_captiondata = json.load(
            open(os.path.join(dataroot, "flk30k_LN.json"), "rb")
        )["images"]
        wordembedding = word_embedding
        self.indexer = wordembedding.word_indexer
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)

        # Load localized narrative captions
        with h5py.File(os.path.join(dataroot, "flk30k_LN_label.h5"), "r") as hf:
            self.caption_label_start = np.array(hf.get("label_start_ix"))
            self.caption_label_end = np.array(hf.get("label_end_ix"))
            self.caption_labels = np.array(hf.get("labels"))

        # Load localized narrative captions
        if self.coco_data:
            self.coco_vocab = json.load(
                open(os.path.join(dataroot, "coco_LN.json"), "rb")
            )["ix_to_word"]
            self.coco_imagesid_from_captiondata = json.load(
                open(os.path.join(dataroot, "coco_LN.json"), "rb")
            )["images"]
            with h5py.File(os.path.join(dataroot, "coco_LN_label.h5"), "r") as hf:
                self.coco_caption_label_start = np.array(hf.get("label_start_ix"))
                self.coco_caption_label_end = np.array(hf.get("label_end_ix"))
                self.coco_caption_labels = np.array(hf.get("labels"))

        # Files for test and validation annotations
        self.localized_narratives_annotated_testval_file = os.path.join(
            dataroot,
            "cin_annotations/testval_annotations.json",
        )

        self.localized_narratives_testval_data = json.load(
            open(self.localized_narratives_annotated_testval_file, "rb")
        )

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_dir = os.path.join(self.dataroot, "flickr30k-images")
        if self.coco_data:
            # or int(image_id) in self.coco_int_img_ids
            if image_id in self.coco_int_img_ids:
                mouse_trace = np.load(
                    os.path.join(
                        self.dataroot, "coco_LN_trace_box", str(image_id) + ".npy"
                    )
                )
            else:
                mouse_trace = np.load(
                    os.path.join(
                        self.dataroot, "flk30k_LN_trace_box", str(image_id) + ".npy"
                    )
                )
        else:
            mouse_trace = np.load(
                os.path.join(
                    self.dataroot, "flk30k_LN_trace_box", str(image_id) + ".npy"
                )
            )

        # Load visual features
        if int(image_id) in self.train_image_ids or self.split == "test":
            idx = self.imgid2idx[int(image_id)]  # to retrieve pos in pos_box
            pos = self.pos_boxes[idx]
            image = Image.open(
                os.path.join(self.dataroot, "flickr30k-images", str(image_id) + ".jpg")
            )
            im_width = image.size[0]
            im_height = image.size[1]
            hw_array = [im_width, im_height, im_width, im_height]
            feature = torch.from_numpy(self.features[pos[0] : pos[1]]).float()
            # Pad features
            if feature.size(0) < self.max_detected_boxes:
                pad = nn.ZeroPad2d((0, 0, 0, self.max_detected_boxes - feature.size(0)))
                feature = pad(feature)
            else:
                feature = feature[: self.max_detected_boxes]

            # Load detected boxes, labels and attributes
            image_id = str(image_id)
            bboxes = np.array(self.obj_detection_dict[image_id]["bboxes"])  # [B, 4]
            scaled_bboxes = bboxes / hw_array
            scaled_bboxes = scaled_bboxes.tolist()
            bboxes = bboxes.tolist()
            labels = self.obj_detection_dict[image_id]["classes"]
            attrs = (
                self.obj_detection_dict[image_id]["attrs"]
                if "attrs" in self.obj_detection_dict[image_id].keys()
                else []
            )

            # Pad detected bounding boxes
            padbox = [0, 0, 0, 0]

            while len(bboxes) < self.max_detected_boxes:
                bboxes.append(padbox)
            bboxes = bboxes[: self.max_detected_boxes]
            bboxes = torch.tensor(bboxes)
            area = (bboxes[..., 3] - bboxes[..., 1]) * (bboxes[..., 2] - bboxes[..., 0])

            height = bboxes[..., 3] - bboxes[..., 1]
            width = bboxes[..., 2] - bboxes[..., 0]
            bboxes = torch.cat(
                (
                    bboxes,
                    area.unsqueeze_(-1),
                ),
                -1,
            )
            
        else:
            idx = self.val_imgid2idx[int(image_id)]  # to retrieve pos in pos_box
            pos = self.val_pos_boxes[idx]
            image = Image.open(
                os.path.join(self.dataroot, "flickr30k-images", str(image_id) + ".jpg")
            )
            im_width = image.size[0]
            im_height = image.size[1]
            hw_array = [im_width, im_height, im_width, im_height]
            feature = torch.from_numpy(self.val_features[pos[0] : pos[1]]).float()
            # Pad features
            if feature.size(0) < self.max_detected_boxes:
                pad = nn.ZeroPad2d((0, 0, 0, self.max_detected_boxes - feature.size(0)))
                feature = pad(feature)
            else:
                feature = feature[: self.max_detected_boxes]

            # Load detected boxes, labels and attributes
            image_id = str(image_id)
            bboxes = np.array(self.val_obj_detection_dict[image_id]["bboxes"])  # [B, 4]
            scaled_bboxes = bboxes / hw_array
            scaled_bboxes = scaled_bboxes.tolist()
            bboxes = bboxes.tolist()
            labels = self.val_obj_detection_dict[image_id]["classes"]
            attrs = (
                self.val_obj_detection_dict[image_id]["attrs"]
                if "attrs" in self.val_obj_detection_dict[image_id].keys()
                else []
            )

            # Pad detected bounding boxes
            padbox = [0, 0, 0, 0]

            while len(bboxes) < self.max_detected_boxes:
                bboxes.append(padbox)
            bboxes = bboxes[: self.max_detected_boxes]
            bboxes = torch.tensor(bboxes)
            area = (bboxes[..., 3] - bboxes[..., 1]) * (bboxes[..., 2] - bboxes[..., 0])
            height = bboxes[..., 3] - bboxes[..., 1]
            width = bboxes[..., 2] - bboxes[..., 0]
            bboxes = torch.cat(
                (
                    bboxes,
                    area.unsqueeze_(-1),
                ),
                -1,
            )
            # bboxes = bboxes / bboxes.sum()
        # Convert label classes and attributes to vocab index.
        num_objects = min(len(labels), self.max_detected_boxes)

        if self.encoding_type == "glove":
            object_label_input_ids = [0] * self.max_detected_boxes
            object_label_attn_mask = [0] * self.max_detected_boxes
            object_label_input_ids[:num_objects] = [
                max(self.indexer.index_of(re.split(" ,", w)[-1]), 1) for w in labels
            ]

            object_label_input_ids = object_label_input_ids[: self.max_detected_boxes]
            object_label_attn_mask = object_label_attn_mask[: self.max_detected_boxes]
            object_label_input_ids = torch.tensor(object_label_input_ids)
            object_label_attn_mask = torch.tensor(object_label_attn_mask)
        else:
            # Encode the labels
            object_label_input_ids = []
            object_label_attn_mask = []
            for l in labels:
                obj_encoded = self.tokenizer.encode_plus(
                    text=[l],  # the sentence to be encoded
                    add_special_tokens=False,  # Add [CLS] and [SEP]
                    max_length=16,
                    pad_to_max_length=True,  # maximum length of a sentence
                    return_attention_mask=True,  # Generate the attention mask
                    return_tensors="pt",  # ask the functin to return PyTorch tensors
                )
                # Get the input IDs and attention mask in tensor format

                object_label_input_ids.append(
                    (obj_encoded["input_ids"]).numpy().tolist()
                )
                object_label_attn_mask.append(
                    (obj_encoded["attention_mask"]).numpy().tolist()
                )
                # Pad labels

            padbox = [0] * 16

            while len(object_label_input_ids) < self.max_detected_boxes:
                object_label_input_ids.append([padbox])
                object_label_attn_mask.append([padbox])
            object_label_input_ids = object_label_input_ids[: self.max_detected_boxes]
            object_label_input_ids = torch.tensor(object_label_input_ids)
            object_label_attn_mask = object_label_attn_mask[: self.max_detected_boxes]
            object_label_attn_mask = torch.tensor(object_label_attn_mask)

        if self.split == "train":
            if int(image_id) in self.train_image_ids:
                # get captions from the corresponding image id
                caption_idx = [
                    idx
                    for idx in range(len(self.imagesid_from_captiondata))
                    if self.imagesid_from_captiondata[idx]["id"] == int(image_id)
                ][0]
                label_start_ix = self.caption_label_start[caption_idx]
                label_end_ix = self.caption_label_end[caption_idx]
                # caption_seq_idx = get_captions(caption_idx, label_start_ix, label_end_ix, label)
                # TODO(goarushi): Make it compatible with multiple captions.
                caption_seq_idx = self.caption_labels[
                    label_start_ix - 1
                ]  # label_start_ix starts from 1
                caption_seq = [
                    "".join(self.vocab[str(caption_seq_idx[i])])
                    for i in range(len(caption_seq_idx))
                    if caption_seq_idx[i] > 0
                ]
                caption_seq = " ".join(caption_seq)
                # convert captions into queries for train otheriwse load from annotations.
                phrase_queries = []
                phrase_queries_start_end_idx = []
                mouse_trace_for_phrases = []
                phrase_queries_input_ids = []
                phrase_queries_attention_mask = []
                start_end_phrases = []
                pos_tags = []
                rel_phrases = []
                target_bboxes = []
                doc = nlp(str(caption_seq))
                noun_phrases = doc.noun_chunks
                sense2vec_feats = []

                char_indx = -1
                tokenized_caption_seq = self.tokenizer.tokenize(caption_seq)

                for k in range(len(doc)):
                    try:
                        sense2vec_feats.append(torch.tensor(doc[k]._.s2v_vec))
                    except:
                        sense2vec_feats.append(torch.tensor([0.0] * 128))

                for _, entity in enumerate(noun_phrases):
                    if entity.text not in stop_words:
                        phrase = entity.text
                        # start_idx, end_idx, char_indx = data_utils.find_sublist(
                        #     tokenized_caption_seq,
                        #     self.tokenizer.tokenize(phrase),
                        #     char_indx + 1,
                        # )
                        phrase_queries.append(phrase)
                        phrase_queries_start_end_idx.append([entity.start, entity.end])

                        phrase_input_ids = []
                        phrase_attention_mask = []
                        s = self.tokenizer.tokenize(phrase)
                        lis = [0] * self.max_query_length
                        attn_masks = [0] * self.max_query_length
                        for i in range(min(len(s), self.max_query_length)):
                            lis[i] = max(self.indexer.index_of(s[i]), 1)
                            attn_masks[i] = 1
                        phrase_input_ids.append(lis)
                        phrase_attention_mask.append(attn_masks)
                        phrase_queries_input_ids.append(torch.tensor(phrase_input_ids))
                        phrase_queries_attention_mask.append(
                            torch.tensor(phrase_attention_mask)
                        )
                        # phrase_queries_start_end_idx.append([start_idx+1, end_idx+1])
                        if len(mouse_trace) > 1:
                            try:
                                mouse_trace_idx = mouse_trace[
                                    int((entity.start + entity.end) / 2)
                                ]
                            except:
                                mouse_trace_idx = mouse_trace[len(mouse_trace) - 1]
                                print(len(mouse_trace))

                            mouse_trace_for_phrases.append(
                                [
                                    [
                                        mouse_trace_idx[0] * im_width,
                                        mouse_trace_idx[1] * im_height,
                                        mouse_trace_idx[2] * im_width,
                                        mouse_trace_idx[3] * im_height,
                                    ]
                                ]
                            )
                            mouse_traces = mouse_trace_for_phrases[0]
                gt_coref_matrix = np.zeros((len(phrase_queries), len(phrase_queries)))

                if len(mouse_trace) > 1:
                    for j in range(len(mouse_traces)):
                        for k in range(len(mouse_traces)):
                            iou = utils.calculate_iou(mouse_traces[j], mouse_traces[k])
                            gt_coref_matrix[j, k] = iou

                target_bboxes.append([[0.0, 0.0, 0.0, 0.0]])
                for _, entity in enumerate(doc.noun_chunks):
                    if entity.text not in stop_words:
                        rel_phrases.append(entity.text)
                        pos_tags.append(entity.root.tag_)
                        start_end_phrases.append([entity.start, entity.end])
                rule_coref_matrix = data_utils.prior_coref_rules(
                    rel_phrases, pos_tags, start_end_phrases
                )

            else:
                # TODO(goarushi): add this from our annotations.
                phrase_queries = []
                phrase_queries_start_end_idx = []
                target_bboxes = []
                phrase_queries_input_ids = []
                phrase_queries_attention_mask = []
                mouse_trace_for_phrases = []
                phrase_file = [
                    idx
                    for idx in range(len(self.localized_narratives_testval_data))
                    if self.localized_narratives_testval_data[idx]["image"]
                    == str(image_id)
                ]
                caption_seq = self.localized_narratives_testval_data[phrase_file[0]][
                    "captions"
                ]
                org_entities = self.localized_narratives_testval_data[phrase_file[0]][
                    "query"
                ]
                org_clusters = self.localized_narratives_testval_data[phrase_file[0]][
                    "cluster"
                ]
                org_t_bboxes = self.localized_narratives_testval_data[phrase_file[0]][
                    "target_bboxes"
                ]
                query_start_end_char_index = self.localized_narratives_testval_data[
                    phrase_file[0]
                ][
                    "query_start_end"
                ]  # this is from labelstudio in character format
                query_start_char_list = []
                for start_end in query_start_end_char_index:
                    query_start_char_list.append(start_end[0])
                sorted_query_char_idx = sorted(
                    range(len(query_start_char_list)),
                    key=lambda k: query_start_char_list[k],
                )
                entities = [org_entities[x] for x in sorted_query_char_idx]
                t_bboxes = [org_t_bboxes[x] for x in sorted_query_char_idx]
                clusters = [org_clusters[x] for x in sorted_query_char_idx]
                char_indx = -1
                sense2vec_feats = []
                doc = nlp(str(caption_seq))
                tokenized_caption_seq = self.tokenizer.tokenize(caption_seq)
                for k in range(len(doc)):
                    try:
                        sense2vec_feats.append(torch.tensor(doc[k]._.s2v_vec))
                    except:
                        sense2vec_feats.append(torch.tensor([0.0] * 128))
                for _, entity in enumerate(entities):
                    if entity not in stop_words:
                        phrase = entity
                        start_idx, end_idx, char_indx = data_utils.find_sublist(
                            tokenized_caption_seq,
                            self.tokenizer.tokenize(phrase),
                            char_indx + 1,
                        )
                        phrase_queries.append(phrase)
                        phrase_queries_start_end_idx.append([start_idx, end_idx])
                        phrase_input_ids = []
                        phrase_attention_mask = []
                        s = self.tokenizer.tokenize(phrase)
                        lis = [0] * self.max_query_length
                        attn_masks = [0] * self.max_query_length
                        for i in range(min(len(s), self.max_query_length)):
                            lis[i] = max(self.indexer.index_of(s[i]), 1)
                            attn_masks[i] = 1
                        phrase_input_ids.append(lis)
                        phrase_attention_mask.append(attn_masks)
                        phrase_queries_input_ids.append(torch.tensor(phrase_input_ids))
                        phrase_queries_attention_mask.append(
                            torch.tensor(phrase_attention_mask)
                        )
                        if len(mouse_trace) > 1:
                            try:
                                mouse_trace_idx = mouse_trace[
                                    int((start_idx + end_idx) / 2)
                                ]
                            except:
                                mouse_trace_idx = mouse_trace[len(mouse_trace) - 1]

                            mouse_trace_for_phrases.append(
                                [
                                    [
                                        mouse_trace_idx[0] * im_width,
                                        mouse_trace_idx[1] * im_height,
                                        mouse_trace_idx[2] * im_width,
                                        mouse_trace_idx[3] * im_height,
                                    ]
                                ]
                            )

                for i, entity in enumerate(entities):
                    if len(t_bboxes[i]) > 0:
                        img_height = self.localized_narratives_testval_data[
                            phrase_file[0]
                        ]["img_height"]
                        img_width = self.localized_narratives_testval_data[
                            phrase_file[0]
                        ]["img_width"]

                        # there are more than 1 bounding box for certain queries
                        bbox = t_bboxes[i][0]
                        # bbox = t_bboxes[i][0]
                        x = (bbox[0] * img_width) / 100.0
                        y = (bbox[1] * img_height) / 100.0
                        w = (bbox[2] * img_width) / 100.0
                        h = (bbox[3] * img_height) / 100.0
                        left = int(x)  # xmin
                        top = int(y)  # ymin
                        right = int(x + w)  # xmax
                        bottom = int(y + h)  # ymax
                        target_bboxes.append([[left, top, right, bottom]])
                    else:
                        target_bboxes.append([[0.0, 0.0, 0.0, 0.0]])
                gt_coref_matrix = data_utils.get_gt_coref_matrix(entities, clusters)
                rule_coref_matrix = np.zeros((len(phrase_queries), len(phrase_queries)))

            if self.sentence_patch_sim:
                sentences = str.split(caption_seq, ".")
                max_num_sentences = 16
                (
                    clip_img_embeds,
                    patch_sentence_sim,
                ) = patch_sentence_similarity_clip.clip_sentence_patch_similarity(
                    sentences, image_dir, str(image_id), clip_model, clip_processor
                )
                sense2vec_sentence_feats = []
                for s in range(len(sentences)):
                    doc = nlp(str(sentences[s]))
                    sense2vec_single_feats = []
                    for k in range(32):
                        try:
                            sense2vec_single_feats.append(
                                torch.tensor(doc[k]._.s2v_vec)
                            )
                        except:
                            sense2vec_single_feats.append(torch.tensor([0.0] * 128))

                    sense2vec_single_feats = torch.stack(sense2vec_single_feats)
                    sense2vec_sentence_feats.append((sense2vec_single_feats))
                sense2vec_sentence_feats = (
                    torch.stack(sense2vec_sentence_feats).numpy().tolist()
                )
                s_encoded = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=tuple(
                        sentences
                    ),  # the sentence to be encoded
                    add_special_tokens=False,  # Add [CLS] and [SEP]
                    max_length=32,
                    pad_to_max_length=True,  # maximum length of a sentence
                    return_attention_mask=True,  # Generate the attention mask
                    return_tensors="pt",  # ask the functin to return PyTorch tensors
                )
                # Get the input IDs and attention mask in tensor format
                sentence_input_ids = s_encoded["input_ids"].tolist()
                sentence_attn_mask = s_encoded["attention_mask"].tolist()
                padbox = [0.0] * 32
                padbox_s2v = torch.zeros((32, 128)).numpy().tolist()
                while len(sentence_input_ids) < max_num_sentences:
                    sentence_input_ids.append(padbox)
                    sentence_attn_mask.append(padbox)
                    sense2vec_sentence_feats.append(padbox_s2v)
                sentence_input_ids = torch.tensor(sentence_input_ids)[
                    :max_num_sentences
                ]
                sentence_attn_mask = torch.tensor(sentence_attn_mask)[
                    :max_num_sentences
                ]
                sense2vec_sentence_feats = torch.tensor(sense2vec_sentence_feats)[
                    :max_num_sentences
                ]
                patch_sentence_sim = (patch_sentence_sim).numpy()
                sen_len = len(sentences)
                if len(sentences) > max_num_sentences:
                    sen_len = max_num_sentences
                pad_width = max_num_sentences - sen_len
                patch_sentence_sim = np.pad(
                    patch_sentence_sim, (0, pad_width), mode="constant"
                )[: len(clip_img_embeds), :max_num_sentences]
                patch_sentence_sim = torch.tensor(patch_sentence_sim)
                num_sentences = min(sen_len, max_num_sentences)
        else:  # split as test / val
            # TODO(goarushi): add this from our annotations.
            mouse_trace_for_phrases = []
            phrase_file = [
                idx
                for idx in range(len(self.localized_narratives_testval_data))
                if self.localized_narratives_testval_data[idx]["image"] == str(image_id)
            ]

            caption_seq = self.localized_narratives_testval_data[phrase_file[0]][
                "captions"
            ]
            org_entities = self.localized_narratives_testval_data[phrase_file[0]][
                "query"
            ]
            org_t_bboxes = self.localized_narratives_testval_data[phrase_file[0]][
                "target_bboxes"
            ]
            org_clusters = self.localized_narratives_testval_data[phrase_file[0]][
                "cluster"
            ]

            query_start_end_char_index = self.localized_narratives_testval_data[
                phrase_file[0]
            ][
                "query_start_end"
            ]  # this is from labelstudio in character format
            query_start_char_list = []
            for start_end in query_start_end_char_index:
                query_start_char_list.append(start_end[0])
            sorted_query_char_idx = sorted(
                range(len(query_start_char_list)),
                key=lambda k: query_start_char_list[k],
            )
            entities = [org_entities[x] for x in sorted_query_char_idx]
            t_bboxes = [org_t_bboxes[x] for x in sorted_query_char_idx]
            clusters = [org_clusters[x] for x in sorted_query_char_idx]
            sort_query_start_end_char_index = [
                query_start_end_char_index[x] for x in sorted_query_char_idx
            ]
            tokenized_caption_seq = self.tokenizer.tokenize(caption_seq)
            query_start_end = []
            query_char_start_end = []
            phrase_queries = []
            target_bboxes = []
            phrase_queries_input_ids = []
            phrase_queries_attention_mask = []
            phrase_queries_start_end_idx = []
            phrase_queries_start_end_char_idx = []
            sense2vec_feats = []
            max_assignments = []
            rel_phrases = []
            pos_tags = []
            rel_query_start_end = []

            doc = nlp(str(caption_seq))
            for k in range(len(doc)):
                try:
                    sense2vec_feats.append(torch.tensor(doc[k]._.s2v_vec))
                except:
                    sense2vec_feats.append(torch.tensor([0.0] * 128))
            entities = entities[: self.max_queries]
            char_indx = -1
            count = 0
            for k in clusters:
                for i in range(len(clusters)):
                    if k == clusters[i]:
                        count = count + 1
                max_assignments.append(count)
                count = 0

            if len(entities) > 0:
                for edx, entity in enumerate(entities):
                    start, end, char_indx = data_utils.find_sublist(
                        tokenized_caption_seq,
                        self.tokenizer.tokenize(entity),
                        char_indx + 1,
                    )
                    start_c = sort_query_start_end_char_index[edx][0]
                    end_c = sort_query_start_end_char_index[edx][1] - 1
                    query_char_start_end.append([start_c, end_c])
                    query_start_end.append([start, end])
                for i, entity in enumerate(entities):
                    if len(t_bboxes[i]) > 0:
                        img_height = self.localized_narratives_testval_data[
                            phrase_file[0]
                        ]["img_height"]
                        img_width = self.localized_narratives_testval_data[
                            phrase_file[0]
                        ]["img_width"]
                        phrase_queries.append(entity)
                        phrase_queries_start_end_idx.append(query_start_end[i])
                        phrase_queries_start_end_char_idx.append(
                            query_char_start_end[i]
                        )
                        phrase_input_ids = []
                        phrase_attention_mask = []
                        s = self.tokenizer.tokenize(entity)
                        lis = [0] * self.max_query_length
                        attn_masks = [0] * self.max_query_length
                        for k in range(min(len(s), self.max_query_length)):
                            lis[k] = max(self.indexer.index_of(s[k]), 1)
                            attn_masks[k] = 1
                        phrase_input_ids.append(lis)
                        phrase_attention_mask.append(attn_masks)
                        phrase_queries_input_ids.append(torch.tensor(phrase_input_ids))
                        phrase_queries_attention_mask.append(
                            torch.tensor(phrase_attention_mask)
                        )

                        start_idx = query_start_end[i][0]
                        end_idx = query_start_end[i][1]
                        if len(mouse_trace) > 1:
                            try:
                                mouse_trace_idx = mouse_trace[
                                    int((start_idx + end_idx) / 2)
                                ]
                            except:
                                mouse_trace_idx = mouse_trace[len(mouse_trace) - 1]

                            mouse_trace_for_phrases.append(
                                [
                                    [
                                        mouse_trace_idx[0] * img_width,
                                        mouse_trace_idx[1] * img_height,
                                        mouse_trace_idx[2] * img_width,
                                        mouse_trace_idx[3] * img_height,
                                    ]
                                ]
                            )

                    for bdx in range(len(t_bboxes[i])):
                        # there are more than 1 bounding box for certain queries
                        bbox = t_bboxes[i][bdx]
                        # bbox = t_bboxes[i][0]
                        x = (bbox[0] * img_width) / 100.0
                        y = (bbox[1] * img_height) / 100.0
                        w = (bbox[2] * img_width) / 100.0
                        h = (bbox[3] * img_height) / 100.0
                        left = int(x)  # xmin
                        top = int(y)  # ymin
                        right = int(x + w)  # xmax
                        bottom = int(y + h)  # ymax

                        target_bboxes.append([[left, top, right, bottom]])
                    
                    if self.eval_grounding:
                        if len(t_bboxes[i]) > 1:
                            for _ in range(len(t_bboxes[i]) - 1):
                                phrase_queries.append(entity)
                                phrase_queries_start_end_idx.append(query_start_end[i])
                                phrase_queries_input_ids.append(
                                    torch.tensor(phrase_input_ids)
                                )
                                phrase_queries_attention_mask.append(
                                    torch.tensor(phrase_attention_mask)
                                )
            gt_coref_matrix = np.zeros((len(phrase_queries), len(phrase_queries)))
            for k, entity in enumerate(phrase_queries):
                spacy_entity = nlp(entity)
                spacy_entity_np = list(spacy_entity.noun_chunks)
                if len(spacy_entity_np) > 0:
                    pos_tags.append(spacy_entity_np[-1].root.tag_)
                else:
                    pos_tags.append("NN")

                rel_query_start_end.append(phrase_queries_start_end_idx[k])

                rel_phrases.append(entity)
            rule_coref_matrix = data_utils.prior_coref_rules(
                rel_phrases, pos_tags, rel_query_start_end
            )
            # rule_coref_matrix = np.zeros((len(phrase_queries), len(phrase_queries)))
        # Pad target bboxes
        padbox = [0.0, 0.0, 0.0, 0.0]
        num_obj = min(len(labels), self.max_detected_boxes)
        while len(target_bboxes) < self.max_queries:
            target_bboxes.append([padbox])
        while len(mouse_trace_for_phrases) < self.max_queries:
            mouse_trace_for_phrases.append([padbox])
        target_bboxes = target_bboxes[: self.max_queries]
        mouse_trace_for_phrases = mouse_trace_for_phrases[: self.max_queries]
        padbox = [0.0, 0.0, 0.0, 0.0, 0.0]
        mouse_trace_feats = mouse_trace.tolist()
        while len(mouse_trace_feats) < self.max_caption_length:
            mouse_trace_feats.append(padbox)
        mouse_trace_feats = mouse_trace_feats[: self.max_caption_length]

        # convert caption sequence into vocab index based on embedding type
        # Encode the sentence
        encoded = self.tokenizer.encode_plus(
            text=caption_seq,  # the sentence to be encoded
            add_special_tokens=False,  # Add [CLS] and [SEP]
            max_length=self.max_caption_length,
            pad_to_max_length=True,  # maximum length of a sentence
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the functin to return PyTorch tensors
        )
        # Get the input IDs and attention mask in tensor format
        caption_input_ids = torch.tensor(encoded["input_ids"])
        caption_attn_mask = torch.tensor(encoded["attention_mask"])
        num_query = min(len(phrase_queries), self.max_queries)
        # Pad phrase_queries and phrase_queries_start_end_idx
        while len(phrase_queries) < self.max_queries:
            phrase_queries.append("")
            phrase_queries_start_end_idx.append([0] * 2)
            phrase_queries_start_end_char_idx.append([0] * 2)
            # max_assignments.append(0)
        phrase_queries = phrase_queries[: self.max_queries]
        phrase_queries_start_end_idx = phrase_queries_start_end_idx[: self.max_queries]
        phrase_queries_start_end_char_idx = phrase_queries_start_end_char_idx[
            : self.max_queries
        ]
        while len(phrase_queries_input_ids) < self.max_queries:
            phrase_queries_input_ids.append(
                torch.zeros((1, self.max_query_length), dtype=int)
            )
            phrase_queries_attention_mask.append(
                torch.zeros((1, self.max_query_length), dtype=int)
            )

        phrase_queries_input_ids = torch.stack(phrase_queries_input_ids)
        phrase_queries_attention_mask = torch.stack(phrase_queries_attention_mask)
        phrase_queries_input_ids = phrase_queries_input_ids[: self.max_queries]
        phrase_queries_attention_mask = phrase_queries_attention_mask[
            : self.max_queries
        ]

        while len(sense2vec_feats) < self.max_caption_length:
            sense2vec_feats.append(torch.tensor([0.0] * 128))
        pad_width = self.max_queries - num_query
        gt_coref_matrix = np.pad(gt_coref_matrix, (0, pad_width), mode="constant")
        gt_coref_matrix = gt_coref_matrix[: self.max_queries, : self.max_queries]
        rule_coref_matrix = np.pad(rule_coref_matrix, (0, pad_width), mode="constant")
        rule_coref_matrix = rule_coref_matrix[: self.max_queries, : self.max_queries]
        assert len(phrase_queries) == self.max_queries
        assert len(phrase_queries_start_end_idx) == self.max_queries
        assert len(phrase_queries_start_end_char_idx) == self.max_queries
        if self.split == "train":
            if self.sentence_patch_sim:
                return (
                    torch.tensor(int(image_id)),
                    phrase_queries,
                    feature,
                    object_label_input_ids,
                    object_label_attn_mask,
                    bboxes,
                    torch.tensor(phrase_queries_input_ids),
                    torch.tensor(phrase_queries_attention_mask),
                    torch.tensor(caption_input_ids),
                    torch.tensor(caption_attn_mask),
                    torch.stack(sense2vec_feats),
                    torch.tensor(num_sentences),
                    clip_img_embeds,
                    sentence_input_ids,
                    sentence_attn_mask,
                    sense2vec_sentence_feats,
                    patch_sentence_sim,
                    torch.tensor(phrase_queries_start_end_idx),
                    torch.tensor(num_obj),
                    torch.tensor(num_query),
                    torch.tensor(target_bboxes),
                    torch.tensor(mouse_trace_for_phrases),
                    torch.tensor(mouse_trace_feats),
                    torch.tensor(gt_coref_matrix),
                    torch.tensor(rule_coref_matrix),
                )
            else:
                return (
                    torch.tensor(int(image_id)),
                    phrase_queries,
                    feature,
                    object_label_input_ids,
                    object_label_attn_mask,
                    bboxes,
                    torch.tensor(phrase_queries_input_ids),
                    torch.tensor(phrase_queries_attention_mask),
                    torch.tensor(caption_input_ids),
                    torch.tensor(caption_attn_mask),
                    torch.stack(sense2vec_feats),
                    torch.tensor(phrase_queries_start_end_idx),
                    torch.tensor(num_obj),
                    torch.tensor(num_query),
                    torch.tensor(target_bboxes),
                    torch.tensor(mouse_trace_for_phrases),
                    torch.tensor(mouse_trace_feats),
                    torch.tensor(gt_coref_matrix),
                    torch.tensor(rule_coref_matrix),
                )

        else:
            return (
                torch.tensor(int(image_id)),
                phrase_queries,
                feature,
                object_label_input_ids,
                object_label_attn_mask,
                bboxes,
                torch.tensor(phrase_queries_input_ids),
                torch.tensor(phrase_queries_attention_mask),
                torch.tensor(caption_input_ids),
                torch.tensor(caption_attn_mask),
                torch.stack(sense2vec_feats),
                torch.tensor(phrase_queries_start_end_idx),
                torch.tensor(phrase_queries_start_end_char_idx),
                torch.tensor(num_obj),
                torch.tensor(num_query),
                torch.tensor(target_bboxes),
                torch.tensor(mouse_trace_for_phrases),
                torch.tensor(mouse_trace_feats),
                torch.tensor(gt_coref_matrix),
                torch.tensor(rule_coref_matrix),
                torch.tensor(max_assignments),
            )

    def __len__(self):
        return len(self.image_ids)
