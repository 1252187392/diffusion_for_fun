#encoding:utf-8
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from tqdm import tqdm
from torchvision.io import read_image
from torchvision import transforms
import random

def pad_embedding(embedding, tokenizer):
    return embedding + (
            [tokenizer.end_of_text] * (MAX_PROMPT_LENGTH - len(embedding))
    )


def check_prompt(prompt, prompt_conf_dict):
    for w in prompt_conf_dict['filter_tokens']:
        if w in prompt:
            return False
    for w in prompt_conf_dict['must_tokens']:
        if w not in prompt:
            return False
    return True

def read_from_path(image_path):
    im = read_image(image_path)
    #if im.shape[0] == 4:
    #    im = im[:3,...]
    label_file = image_path.split('.')
    label_file = '.'.join(label_file[:-1])+'_format.txt'
    if os.path.exists(label_file):
        prompt = open(label_file, encoding='utf8').readlines()[0]
    else:
        prompt = ''
    return im, prompt

def random_tags(prompt, r=0.1):
    prompt = prompt.split(',')
    start_ = prompt[:1]
    descriptions = prompt[1:]
    random.shuffle(descriptions)
    cut = int(len(descriptions)*r)
    prompt = ','.join(start_ + descriptions[:-cut])
    return prompt

def get_prompt_token(prompt, tokenizer, prompt_conf_dict):
    if len(prompt):
        prompt = prompt.strip().replace('_', ' ').replace('-', ' ')
        if prompt_conf_dict['random_tags'] > 0:
            prompt = random_tags(prompt, prompt_conf_dict['random_tags'])
        #if len(prompt_conf_dict['add_token']) > 0:
        #    prompt = f'{prompt_conf_dict["add_token"]}, {prompt}'
        token = tokenizer(prompt, max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt")
    else:
        prompt = random.choice(prompt_conf_dict['prompts_templates']).format(prompt_conf_dict['add_token'])
        token = tokenizer(prompt, max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt")
    return token

def get_paths(data_dir):
    paths = []
    for root, dirs, fs in tqdm(os.walk(data_dir)):
        for f in fs:
            if f[-4:] not in ['.jpg']:
                continue
            path = os.path.join(root, f)
            paths.append(path)
    return paths

def get_data_paths(data_dirs):
    all_paths = []
    for data_dir in data_dirs:
        paths = get_paths(data_dir['dir'])
        if "max" in data_dir:
            paths = paths[:data_dir["max"]]
        print(f"dir:{data_dir['dir']},ori_nums:{len(paths)}")
        paths = paths * data_dir["repeat"]
        print(f"dir:{data_dir['dir']},train_nums:{len(paths)}")
        all_paths += paths
    for i in range(len(all_paths)-1,-1,-1):
        label_file = all_paths[i].split('.')
        label_file = '.'.join(label_file[:-1]) + '_format.txt'
        if not os.path.exists(label_file):
            all_paths.pop(i)
    random.shuffle(all_paths)
    return all_paths

class CustomImageDataset(Dataset):
    def __init__(self,  data_dir, tokenizer, prompt_conf=None):
        self.paths = get_data_paths(data_dir)
        self.tokenizer = tokenizer
        self.resolution = 512
        self.prompt_conf = prompt_conf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #while True:
            #try:
            #    image, label = read_from_path(self.paths[idx])
            #    break
            #except Exception as e:
            #    print(e)
            #    print(self.paths[idx])
            #    self.paths[idx] = random.choice(self.paths)
        image, label = read_from_path(self.paths[idx])
        image = transforms.Resize(self.resolution,
                                  interpolation=transforms.InterpolationMode.BILINEAR)(image)
        image = transforms.CenterCrop((512, 512))(image) # h,w
        #image = image.float()
        image = transforms.Normalize([0.5], [0.5])(image.float()/255)

        tokens = get_prompt_token(label, self.tokenizer, self.prompt_conf)
        tokens, mask = tokens.input_ids[0], tokens.attention_mask[0]
        return image, tokens


def get_dataloader(data_dir, tokenizer, prompt_conf_dict,batch_size=2):
    ds = CustomImageDataset(data_dir, tokenizer, prompt_conf_dict)
    train_dataloader = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        #collate_fn=collate_fn,
        batch_size=batch_size,
        #num_workers=args.dataloader_num_workers,
    )
    return train_dataloader