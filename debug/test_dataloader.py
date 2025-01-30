import sys
sys.path.append('.')

import numpy as np
from tqdm import tqdm
from dataloader.dataloader import get_dataloader, get_bucket_dataloader

from conf import person2 as conf

from transformers import CLIPTokenizer
from dataloader.dataloader import get_dataloader, get_bucket_dataloader
from utils.image_utils import plot_images


tokenizer = CLIPTokenizer.from_pretrained(
    conf.model_conf["pretrained_model_name_or_path"], subfolder="tokenizer", revision=conf.model_conf["revision"]
)

data_loader = get_bucket_dataloader(conf.data_dirs, tokenizer, conf.prompt_conf_dict, conf.batch_size)

for step, (images, tokens) in tqdm(enumerate(data_loader)):
    print(images.shape, tokens.shape)
    images = (images[0] * 0.5 + 0.5) * 255
    images = images.cpu().numpy().astype(np.uint8)
    images = np.transpose(images, [0, 2, 3, 1])
    plot_images(images)
    #cv2.imshow(images)
    break
