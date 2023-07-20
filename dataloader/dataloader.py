#encoding:utf-8
import torch
from torch._C import _unset_default_mobile_cpu_allocator
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

def read_from_path(image_path, label_file, default_value=''):
    im = read_image(image_path)
    if label_file is not None and os.path.exists(label_file):
        prompt = open(label_file, encoding='utf8').readlines()[0]
    else:
        prompt = default_value
    return im, prompt

def random_tags(prompt, r=0.1):
    prompt = prompt.split(',')
    start_ = prompt[:1]
    descriptions = prompt[1:35]
    random.shuffle(descriptions)
    cut = int(len(descriptions)*r)
    if cut == 0:
        cut = 1
    prompt = start_ + descriptions[:-cut]
    random.shuffle(prompt)
    prompt = ','.join(prompt)

    #random.shuffle(prompt)
    #prompt = ','.join(prompt)
    return prompt

def get_prompt_token(prompt, tokenizer, prompt_conf_dict):
    if len(prompt):
        prompt = prompt.strip().replace('_', ' ').replace('-', ' ')
        #if prompt_conf_dict['random_tags'] > 0:
        prompt = random_tags(prompt, prompt_conf_dict['random_tags'])
        #if len(prompt_conf_dict['add_token']) > 0:
        #    prompt = f'{prompt_conf_dict["add_token"]}, {prompt}'
        #print(prompt)
        token = tokenizer(prompt, max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt")
        #print(token)
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

def get_data_paths(data_dirs, use_default=False):
    all_paths = []
    for data_dir in data_dirs:
        paths = get_paths(data_dir['dir'])
        if "max" in data_dir:
            paths = paths[:data_dir["max"]]
        print(f"dir:{data_dir['dir']},ori_nums:{len(paths)}")
        paths = paths * data_dir["repeat"]
        print(f"dir:{data_dir['dir']},train_nums:{len(paths)}")
        all_paths += paths
    pair_paths = []
    for i in range(len(all_paths)):
        label_file = all_paths[i].split('.')
        label_file = '.'.join(label_file[:-1]) + '_format.txt'
        if use_default:
            pair_paths.append((all_paths[i], None))
            continue
        if not os.path.exists(label_file):
            label_file = label_file.replace('_format.txt', '.txt')
            if not os.path.exists(label_file):
                continue
        pair_paths.append((all_paths[i], label_file))
    random.shuffle(pair_paths)
    return pair_paths

class CustomImageDataset(Dataset):
    def __init__(self,  data_dir, tokenizer, prompt_conf=None):
        self.paths = get_data_paths(data_dir, prompt_conf['use_default'])
        self.tokenizer = tokenizer
        self.resolution = 512
        self.prompt_conf = prompt_conf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image, label = read_from_path(self.paths[idx][0], self.paths[idx][1], random.choice(self.prompt_conf['prompts_templates']))
        image = transforms.Resize(self.resolution,
                                  interpolation=transforms.InterpolationMode.BILINEAR)(image)
        image = transforms.CenterCrop((512, 512))(image) # h,w
        #image = image.float()
        image = transforms.Normalize([0.5], [0.5])(image.float()/255)
        image = image[:3, :, :]
        tokens = get_prompt_token(label, self.tokenizer, self.prompt_conf)
        tokens, mask = tokens.input_ids[0], tokens.attention_mask[0]
        return image, tokens


class BucketImageDataset(Dataset):
    def __init__(self, data_dir, tokenizer, prompt_conf=None):
        self.paths = get_data_paths(data_dir, prompt_conf['use_default'])
        self.tokenizer = tokenizer
        self.resolution = 512
        self.prompt_conf = prompt_conf
        self.batch = 3
        self.buckets = [[512,512], [512,640], [512, 768],[448, 768],[384, 768]]
        #                   1         1.25       1.5         1.7        2
        self.ratios = [x[1]/x[0] for x in self.buckets]
        self.batch_datas = [[] for i in range(len(self.buckets))]
        self.data_idx = 0

    def select_size(self, r):
        for i in range(len(self.ratios)-1, -1, -1):
            if r >= self.ratios[i]:
                return i, self.buckets[i]
        return 0,self.buckets[0]

    def prepare_data(self):
        for i in range(self.data_idx, len(self.paths)):
            image, label = read_from_path(self.paths[i][0], self.paths[i][1], random.choice(self.prompt_conf['prompts_templates']))
            _, h,w = image.shape
            idx, (W, H) = self.select_size(h/w)
            image = transforms.Resize(W,
                                      interpolation=transforms.InterpolationMode.BILINEAR)(image)
            image = transforms.CenterCrop((H, W))(image)  # h,w
            image = transforms.Normalize([0.5], [0.5])(image.float() / 255)
            image = image[:3, :, :]
            tokens = get_prompt_token(label, self.tokenizer, self.prompt_conf)
            tokens, mask = tokens.input_ids[0], tokens.attention_mask[0]
            self.batch_datas[idx].append([image, tokens])
            if len(self.batch_datas[idx]) == self.batch:
                self.data_idx = (i + 1) % len(self.paths)
                return idx

    def __len__(self):
        return len(self.paths)//self.batch

    def __getitem__(self, idx):
        batch_idx = self.prepare_data()
        datas = self.batch_datas[batch_idx]
        self.batch_datas[batch_idx] = []
        images = [d[0] for d in datas]
        tokens = [d[1] for d in datas]
        images = torch.stack(images,dim=0)
        tokens = torch.stack(tokens,dim=0)
        return images, tokens

def get_dataloader(data_dir, tokenizer, prompt_conf_dict, batch_size=2):
    ds = CustomImageDataset(data_dir, tokenizer, prompt_conf_dict)
    train_dataloader = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        #collate_fn=collate_fn,
        batch_size=batch_size,
        #num_workers=args.dataloader_num_workers,
    )
    return train_dataloader

def get_bucket_dataloader(data_dir, tokenizer, prompt_conf_dict, batch_size=2):
    ds = BucketImageDataset(data_dir, tokenizer, prompt_conf_dict)
    train_dataloader = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        #collate_fn=collate_fn,
        #num_workers=args.dataloader_num_workers,
        batch_size=None
    )
    return train_dataloader