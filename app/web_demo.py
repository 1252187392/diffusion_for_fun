# encoding: utf-8

import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gradio as gr
import importlib

from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler, StableDiffusionLatentUpscalePipeline
from models.train_step import generate_image, generate_SR_image


from lora_diffusion import (
    monkeypatch_lora,
    tune_lora_scale,
)

assert len(sys.argv) > 1, 'must specify conf file'
conf = sys.argv[1]
print('using conf:', conf)
conf = importlib.import_module(f'conf.{conf}')
ddpm = DDPMScheduler.from_pretrained(conf.model_conf['pretrained_model_name_or_path'], subfolder="scheduler")
#ddim = DDIMScheduler.from_pretrained(conf.model_conf['pretrained_model_name_or_path'], subfolder="scheduler")
#pndm = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")
#lms = LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
#euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
#euler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
#scheduler = DDIMScheduler.from_pretrained(conf.model_conf['pretrained_model_name_or_path'])
pipeline = DiffusionPipeline.from_pretrained(
        conf.model_conf['pretrained_model_name_or_path'],
        scheduler=ddpm,
        revision=conf.model_conf['revision'],
        torch_dtype=torch.float16,
        safety_checker=None
)

#from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
#pipeline.enable_xformers_memory_efficient_attention(MemoryEfficientAttentionFlashAttentionOp)
pipeline = pipeline.to("cuda", torch_dtype=torch.float16)

if conf.load_weights:
    pipeline.unet.load_attn_procs(conf.load_weights)
    monkeypatch_lora(pipeline.text_encoder, torch.load(os.path.join(conf.load_weights, "lora_text_encoder.pt")),
                     target_replace_module=["CLIPAttention"])


def text_to_image(height, width, prompt, prompt2, negative_prompt, seed, text_lora_scale=1, diffusion_lora_scale=1, steps=25):
    print('used prompt:', prompt)
    steps = int(steps)
    tune_lora_scale(pipeline.text_encoder, text_lora_scale)
    with torch.no_grad():
        images1 = generate_image(pipeline, prompt, steps, seed, 2, 'cuda',
                                 None, negative_prompt, diffusion_lora_scale, (width, height))
        images2 = generate_image(pipeline, prompt2, steps, seed, 2, 'cuda',
                                 None, negative_prompt, diffusion_lora_scale, (width, height))

    return images1+images2


def init_examples():
    prompts = [line.strip() for line in open('datas/prompts.txt', encoding='utf8') if len(line.strip()) > 0]
    neg_prompts = [line.strip() for line in open('datas/neg_prompts.txt', encoding='utf8') if len(line.strip()) > 0]
    seed = 137
    scale = 1
    steps = 25
    size = 512
    examples = []
    for idx, prompt in enumerate(prompts):
        examples.append([size, size, prompt, '', neg_prompts[idx%len(neg_prompts)], seed, scale, scale, steps])
    return examples


interface = gr.Interface(fn=text_to_image,
                         inputs=[
                             gr.Slider(256, 1024, value=512, step=64, label="height", info="height"),
                             gr.Slider(256, 1024, value=512, step=64, label="width", info="width"),
                             "text", "text", "text", gr.Slider(1, 20000, value=1, label="seed", info="随机种子"),
                             gr.Slider(0, 1, value=1, label="text_lora_scale", info="text_lora_scale"),
                             gr.Slider(0, 1, value=1, label="diffusion_lora_scale", info="diffusion_lora_scale"),
                             gr.Slider(1, 50, value=1, step=1, label="steps", info="steps")],
                         outputs=["image"] * 4, examples=init_examples())

interface.launch(share=False)
