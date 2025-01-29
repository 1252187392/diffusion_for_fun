import os
import sys

import math
import importlib
import torch
torch.distributed.init_process_group("gloo")
import torch.utils.checkpoint

from diffusers import DiffusionPipeline
from diffusers.loaders import AttnProcsLayers

from models.model_utils import init_models, init_accelerator, set_unet_lora
from dataloader.dataloader import get_dataloader, get_bucket_dataloader
from models.train_step import train_step, generate_image
from tqdm import tqdm
import itertools
'''
from lora_diffusion import (
    inject_trainable_lora,
    save_lora_weight,
    extract_lora_ups_down,
    #monkeypatch_lora,
    tune_lora_scale,
)
'''
from lora import inject_trainable_lora,save_lora_weight

conf = sys.argv[1]
print('using conf:', conf)
conf = importlib.import_module(f'conf.{conf}')

accelerator = init_accelerator(conf.accelerator_conf)

noise_scheduler, tokenizer, text_encoder, vae, unet = init_models(conf.model_conf)

# freeze parameters of models to save more memory
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# mixed precision

# as these models are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float16

# Move unet, vae and text_encoder to device and cast to weight_dtype
unet.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)
text_encoder.to(accelerator.device, dtype=weight_dtype)

# lora 注入放在后面，因为训练权重需要FP32

if conf.load_weights:
    unet_loras = conf.load_weights + '/lora_unet.pt'
    # text encoder file
    text_loras = conf.load_weights + '/lora_text_encoder.pt'
    if not os.path.exists(text_loras):
        text_loras = None
    print(f'load_weights:{conf.load_weights}, text_encoder:f{text_loras}')
else:
    unet_loras, text_loras = None, None

#unet.enable_xformers_memory_efficient_attention()


unet_lora_params, train_names = inject_trainable_lora(unet, loras=unet_loras, r=4)

# set text encoder
text_encoder_lora_params, text_encoder_names = inject_trainable_lora(
            text_encoder, target_replace_module=["CLIPAttention"],
            loras=text_loras, #pt file
            r=4,
)

params_to_optimize = (
    [
        {
            "params": itertools.chain(*unet_lora_params),
            "lr": 1e-4,
        },
        {
            "params": itertools.chain(*text_encoder_lora_params),
            "lr": 5e-5,
        },
    ]
)

#sys.exit()
optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=conf.optimizer_conf['learning_rate'],
    betas=(conf.optimizer_conf['adam_beta1'], conf.optimizer_conf['adam_beta2']),
    weight_decay=conf.optimizer_conf['adam_weight_decay'],
    eps=conf.optimizer_conf['adam_epsilon'],
)

pipeline = DiffusionPipeline.from_pretrained(
        conf.model_conf['pretrained_model_name_or_path'],
        text_encoder = accelerator.unwrap_model(text_encoder),
        unet=accelerator.unwrap_model(unet),
        scheduler=noise_scheduler,
        revision=conf.model_conf['revision'],
        torch_dtype=weight_dtype,
        safety_checker=None,
        mirror='tuna'
    )

pipeline = pipeline.to(accelerator.device)

#data_loader = get_dataloader(conf.data_dirs, tokenizer, conf.prompt_conf_dict, conf.batch_size)
print('batch_size:', conf.batch_size)
data_loader = get_bucket_dataloader(conf.data_dirs, tokenizer, conf.prompt_conf_dict, conf.batch_size)


# Prepare everything with our `accelerator`.
lora_layers, text_encoder, optimizer, data_loader = accelerator.prepare(
    unet, text_encoder, optimizer, data_loader
)

num_update_steps_per_epoch = math.ceil(len(data_loader) / conf.accelerator_conf['gradient_accumulation_steps'])

total_batch_size = conf.batch_size * accelerator.num_processes * conf.accelerator_conf['gradient_accumulation_steps']
print(f'total_batch_size:{total_batch_size}')

def save(epoch):
    if not os.path.exists(conf.save_weights):
        os.mkdir(conf.save_weights)
    if not os.path.exists(f"{conf.save_weights}/{epoch}"):
        os.mkdir(f"{conf.save_weights}/{epoch}")
    save_lora_weight(
        unet,
        os.path.join(f"{conf.save_weights}/{epoch}", "lora_unet.pt"),
    )
    save_lora_weight(
        text_encoder,
        os.path.join(f"{conf.save_weights}/{epoch}", "lora_text_encoder.pt"),
        target_replace_module=["CLIPAttention"],
    )


for epoch in range(conf.start_epoch+1, conf.epochs):
    #unet.train()
    train_loss = 0
    for step, batch_data in tqdm(enumerate(data_loader)):
        if step >= len(data_loader):
            break
        #train_step(batch_data, vae, text_encoder, unet, noise_scheduler, weight_dtype)
        if len(batch_data[0].shape) == 5:
            batch_data = (batch_data[0][0], batch_data[1][0])
        #if accelerator.is_main_process:
        #    print(batch_data[0].shape)
        with accelerator.accumulate(unet):
            loss = train_step(batch_data, vae, text_encoder, unet, noise_scheduler, weight_dtype)

            avg_loss = accelerator.gather(loss.repeat(conf.batch_size)).mean()
            train_loss += avg_loss.item() / conf.accelerator_conf['gradient_accumulation_steps']

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = lora_layers.parameters()
                accelerator.clip_grad_norm_(params_to_clip, conf.optimizer_conf['max_grad_norm'])
            optimizer.step()
            optimizer.zero_grad()
        if accelerator.is_main_process and step and step % 30 ==0:
            print(f'epoch:{epoch}, step:{step}, loss:{train_loss/step}')
    if epoch % conf.callback_frequency == 0:
        accelerator.wait_for_everyone()
        #unet.save_attn_procs(f"{conf.save_weights}/{epoch}")
        if accelerator.is_main_process:
            save(epoch)

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    save(epoch)

