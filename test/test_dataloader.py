#encoding: utf-8
import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
sys.path.append('.')
from transformers import CLIPTokenizer
from dataloader.dataloader import get_dataloader, get_bucket_dataloader
from utils.image_utils import plot_images
from conf import exp1 as conf


tokenizer = CLIPTokenizer.from_pretrained(
    conf.model_conf["pretrained_model_name_or_path"], subfolder="tokenizer", revision=conf.model_conf["revision"]
)


#ds = get_dataloader(conf.data_dirs, tokenizer, conf.prompt_conf_dict,conf.batch_size)
ds = get_bucket_dataloader(conf.data_dirs, tokenizer, conf.prompt_conf_dict,conf.batch_size)
idx = 0
for ims, labels in tqdm(ds):
    print(ims.shape)
    print(labels.shape)
    print(labels[0])
    ims = (ims + 1)*127.5
    ims = ims.type(torch.uint8)
    ims = np.transpose(ims,[0,2,3,1])
    ims = ims.numpy()
    ims = ims.tolist()
    plot_images(ims)
    cv2.imwrite(f'outputs/{idx}.jpg', np.array(ims[0])[:, :, -1::])
    idx += 1
    break