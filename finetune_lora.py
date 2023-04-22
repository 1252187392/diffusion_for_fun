import os
import sys
#os.environ['TORCH_DISTRIBUTED_BACKEND'] = 'gloo' # for windows， else comment this
import math
import torch
torch.distributed.init_process_group("gloo")
import torch.utils.checkpoint

from diffusers import DiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from conf import person as conf
from models.model_utils import init_models, init_accelerator, set_unet_lora
from dataloader.dataloader import get_dataloader
from models.train_step import train_step,generate_image
from tqdm import tqdm

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

if conf.load_weights:
    unet.load_attn_procs(conf.load_weights)
    attn_processors = {k: v.to(device=accelerator.device, dtype=torch.float32) for k, v in unet.attn_processors.items()}
    unet.set_attn_processor(attn_processors)
else:
    set_unet_lora(unet)

lora_layers = AttnProcsLayers(unet.attn_processors)
for x in lora_layers.parameters():
    print(x.dtype)
    break
#sys.exit()
optimizer = torch.optim.AdamW(
    lora_layers.parameters(),
    lr=conf.optimizer_conf['learning_rate'],
    betas=(conf.optimizer_conf['adam_beta1'], conf.optimizer_conf['adam_beta2']),
    weight_decay=conf.optimizer_conf['adam_weight_decay'],
    eps=conf.optimizer_conf['adam_epsilon'],
)


data_loader = get_dataloader(conf.data_dirs, tokenizer, conf.prompt_conf_dict, conf.batch_size)

# Prepare everything with our `accelerator`.
lora_layers, optimizer, data_loader = accelerator.prepare(
    lora_layers, optimizer, data_loader
)

num_update_steps_per_epoch = math.ceil(len(data_loader) / conf.accelerator_conf['gradient_accumulation_steps'])

total_batch_size = conf.batch_size * accelerator.num_processes * conf.accelerator_conf['gradient_accumulation_steps']
print(f'total_batch_size:{total_batch_size}')
pipeline = DiffusionPipeline.from_pretrained(
        conf.model_conf['pretrained_model_name_or_path'],
        unet = accelerator.unwrap_model(unet),
        revision=conf.model_conf['revision'],
        torch_dtype=weight_dtype,
        safety_checker=None
    )
#pipeline.safety_checker = lambda images, clip_input: (images, False)
pipeline = pipeline.to(accelerator.device)

for epoch in range(conf.epochs):
    unet.train()
    train_loss = 0
    for step, batch_data in tqdm(enumerate(data_loader)):
        #train_step(batch_data, vae, text_encoder, unet, noise_scheduler, weight_dtype)
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
        if step and step % 20 ==0:
            print(f'epoch:{epoch}, step:{step}, loss:{train_loss/step}')
    if epoch % conf.callback_frequency == 0:
        accelerator.wait_for_everyone()
        unet.save_attn_procs(f"{conf.save_weights}/{epoch}")
        generate_image(pipeline, conf.generation_conf["prompt"], conf.generation_conf["steps"],
                       conf.generation_conf["seed"],
                       conf.generation_conf["nums"],
                       accelerator.device,
                       f'{conf.generation_conf["save_dir"]}{epoch}')

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    generate_image(pipeline, conf.generation_conf["prompt"],
                   conf.generation_conf["steps"],
                   conf.generation_conf["seed"],
                   conf.generation_conf["nums"],
                   accelerator.device,
                   f'{conf.generation_conf["save_dir"]}{epoch}')
    unet.save_attn_procs(f"{conf.save_weights}/{epoch}")


