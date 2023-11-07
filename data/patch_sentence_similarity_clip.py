from PIL import Image
import sys
import os
from split_image import split_image
import glob
import torch
import h5py

sys.path.append("/disk/nfs/gazinasvolume1/s1985335/cr-image-narrations-ssl")
device = torch.device("cuda")
## load pretrained CLIP model
# from transformers import CLIPProcessor, CLIPModel

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
out_dir = (
    "/disk/nfs/gazinasvolume1/s1985335/cr-image-narrations-ssl/data/tmp_split_images/"
)

# features_f = h5py.File(os.path.join(output_dir,'clip_image_features.hdf5'),'w')
# sim_f = h5py.File(os.path.join(output_dir,'clip_patch_sim.hdf5'),'w')

def clip_sentence_patch_similarity(caption, image_path, im_name, model, processor):
    ## 3 rows and 4 cols
    # im = image_path + "/" + im_name + ".jpg"
    # os.system("split-image --output-dir /disk/nfs/gazinasvolume1/s1985335/coreference-and-grounding/data/tmp_split_images/ --quiet {} 2 2".format(im))
    patch_sentence_sim = []
    split_image_embeds = []
    model = model.to(device)
    for i in range(2 * 2):
        im_split = out_dir + im_name + "_" + str(i) + ".jpg"
        image = Image.open(im_split)
        # this will return the
        inputs = processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(device)
        outputs = model(**inputs)
        img_embeds = outputs.image_embeds
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities
        patch_sentence_sim.append(probs)
        split_image_embeds.append(img_embeds)
    patch_sentence_sim = torch.stack(patch_sentence_sim).cpu().detach().squeeze(-2)
    split_image_embeds = torch.stack(split_image_embeds).cpu().detach().squeeze(-2)
    
    
    # features_f.create_dataset(im_name,data=split_image_embeds)
    # sim_f.create_dataset(im_name,data=patch_sentence_sim)
    
    ## delete the saved split images from the folder
    # files = os.listdir(out_dir)
    # for f in files:
    #     os.remove(out_dir + f)

    return split_image_embeds, patch_sentence_sim


