# encoding: utf-8

import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gradio as gr
import importlib
from PIL import Image

from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from models.train_step import generate_image, image_to_image


from lora_diffusion import (
    monkeypatch_lora,
    tune_lora_scale,
)

assert len(sys.argv) > 1, 'must specify conf file'
conf = sys.argv[1]
print('using conf:', conf)
conf = importlib.import_module(f'conf.{conf}')
repo_id = conf.model_conf['pretrained_model_name_or_path']
scheduler = {
    'ddpm': DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler"),
    'ddim': DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler"),
    #'pndm': PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler"),
    #'lms' : LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler"),
    #'euler_anc' : EulerAncestralDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler"),
    #'euler' : EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler"),
}

base_pipeline = DiffusionPipeline.from_pretrained(
        repo_id,
        scheduler=scheduler['ddpm'],
        revision=conf.model_conf['revision'],
        torch_dtype=torch.float16,
        safety_checker=None
)

#from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
#pipeline.enable_xformers_memory_efficient_attention(MemoryEfficientAttentionFlashAttentionOp)
base_pipeline = base_pipeline.to("cuda", torch_dtype=torch.float16)

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id, scheduler=scheduler['ddpm'],
                                                      revision=conf.model_conf['revision'], torch_dtype=torch.float16,
                                                      safety_checker=None)
img2img_pipe = img2img_pipe.to("cuda")


def load_weights(pipeline):
    if conf.load_weights_web:
        print(os.path.join(conf.load_weights_web, "lora_unet.pt"))
        monkeypatch_lora(pipeline.unet, torch.load(os.path.join(conf.load_weights_web, "lora_unet.pt")),)
        monkeypatch_lora(pipeline.text_encoder, torch.load(os.path.join(conf.load_weights_web, "lora_text_encoder.pt")),
                         target_replace_module=["CLIPAttention"])
load_weights(base_pipeline)
load_weights(img2img_pipe)


def text_to_image(init_image, height, width, prompt, prompt2, negative_prompt, strength, guidance_scale, seed, text_lora_scale=1,
                  diffusion_lora_scale=1,
                  steps=25):
    print('used prompt:', prompt)
    steps = int(steps)
    pipeline = base_pipeline if init_image is None else img2img_pipe
    fn = generate_image if init_image is None else image_to_image
    if init_image is not None:
        init_image = Image.fromarray(init_image).convert('RGB')
        init_image = init_image.resize((width, height))
    tune_lora_scale(pipeline.unet, diffusion_lora_scale)
    tune_lora_scale(pipeline.text_encoder, text_lora_scale)
    with torch.no_grad():
        images1 = fn(pipeline, prompt, steps, seed, 2, 'cuda',
                                 None, negative_prompt, diffusion_lora_scale, (width, height),init_image=init_image,
                                strength=strength, guidance_scale=guidance_scale
                                )
        images2 = fn(pipeline, prompt2, steps, seed, 2, 'cuda',
                                 None, negative_prompt, diffusion_lora_scale, (width, height),init_image=init_image,
                                strength=strength, guidance_scale=guidance_scale)

    return images1 + images2


def init_examples():
    prompts = [line.strip() for line in open('datas/prompts.txt', encoding='utf8') if len(line.strip()) > 0]
    neg_prompts = [line.strip() for line in open('datas/neg_prompts.txt', encoding='utf8') if len(line.strip()) > 0]
    seed = 137
    scale = 1
    steps = 25
    size = 512
    examples = []
    for idx, prompt in enumerate(prompts):
        examples.append([None, size, size, prompt, '', neg_prompts[idx%len(neg_prompts)], 0.75, 7.5, seed, scale, scale, steps])
    return examples


interface = gr.Interface(fn=text_to_image,
                         inputs=[
                             gr.Image(label='init_image'),
                             gr.Slider(256, 1024, value=512, step=64, label="height", info="height"),
                             gr.Slider(256, 1024, value=512, step=64, label="width", info="width"),
                             "text", "text", "text",
                             gr.Slider(0, 1, value=0.75, step=0.01, label="strength", info="noise strength"),
                             gr.Slider(0, 10, value=1, step=0.1, label="guidance_scale", info="text infulence"),
                             gr.Slider(1, 20000, value=1, label="seed", info="随机种子"),
                             gr.Slider(0, 2, value=1, label="text_lora_scale", info="text_lora_scale"),
                             gr.Slider(0, 2, value=1, label="diffusion_lora_scale", info="diffusion_lora_scale"),
                             gr.Slider(1, 50, value=1, step=1, label="steps", info="steps")],
                         outputs=["image"] * 4, examples=init_examples())

interface.launch(share=False)
