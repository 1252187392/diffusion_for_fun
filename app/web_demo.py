import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gradio as gr
import cv2
import importlib

from diffusers import DiffusionPipeline, DDIMScheduler,DDPMScheduler
from models.train_step import generate_image

from lora_diffusion import (
    inject_trainable_lora,
    save_lora_weight,
    extract_lora_ups_down,
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
        #unet = accelerator.unwrap_model(unet),
        scheduler=ddpm,
        revision=conf.model_conf['revision'],
        torch_dtype=torch.float16,
        safety_checker=None
)
pipeline = pipeline.to("cuda")

if conf.load_weights:
    pipeline.unet.load_attn_procs(conf.load_weights)

monkeypatch_lora(pipeline.text_encoder, torch.load(os.path.join(conf.load_weights, "lora_text_encoder.pt")),
                 target_replace_module=["CLIPAttention"])

def text_to_image(prompt, prompt2, negative_prompt, seed, text_lora_scale=1, diffusion_lora_scale=1, steps=25):
    #prompt = f'masterpiece,{prompt}' if 'masterpiece' not in prompt else prompt
    print('used prompt:', prompt)
    tune_lora_scale(pipeline.text_encoder, text_lora_scale)
    images1 = generate_image(pipeline, prompt, steps, seed, 2, 'cuda',
                             None, negative_prompt, diffusion_lora_scale)
    images2 = generate_image(pipeline, prompt2, steps,seed,2,'cuda',
                             None,negative_prompt,diffusion_lora_scale)

    return images1+images2


eg1 = """
masterpiece,1girl, alternate costume, ass, bangs, very long hair, white hair, white shirt, white skirt, window, bare arms, blue e
yes, blunt bangs, blush, bow, breasts, chair, chromatic aberration, clothes pull, crop top, easy chair, eric (tianqijiang), flower knot, from
below, from side, genshin impact, groin, hair bow, hair ribbon, indoors, kamisato ayaka, long hair, looking at viewer, medium breasts, midriff
, miniskirt, on chair, one knee, panties, parted lips, pink bow, pink panties, pleated skirt, ponytail, red ribbon, revision, ribbon, shirt, s
ide-tie panties, sideboob, sidelighting, signature, skirt, skirt pull, sleeveless, sleeveless shirt, solo, thighs, tress ribbon, underwear
"""
eg2 = """
masterpiece,1girl, animal ears, bangs, breasts, cherry blossoms, cowboy shot, crystalfly (genshin impact), detached sleeves, earrings, falling petals, floating hair, floppy ears, floral print, flower knot, fox ears, genshin impact, hair between eyes, hair ornament, hand up, japanese clothes, jewelry, kusunokinawate, long hair, long sleeves, looking at viewer, medium breasts, nontraditional miko, petals, purple eyes, red skirt, ribbon trim, shirt, sidelocks, skirt, sleeveless, sleeveless shirt, solo, tassel, thighs, turtleneck, white sleeves, wide sleeves, yae miko
"""
eg3 = """
masterpiece,1girl, animal ears, bangs, bare shoulders, blush, breasts, earrings, fantongjun, fox ears, genshin impact, hair ornament, jewelry, large breasts, long hair, looking at viewer, naked towel, open mouth, pink hair, purple eyes, raised eyebrows, sidelocks, solo, thighs, towel, very long hair, yae miko
"""
eg4 = """
modelshoot style,(best quality, masterpiece:1.1), (realistic:1.4),intricate elegant, (highly detailed),sharp focus, dramatic,photorealistic,A beautiful Chinese girl,<lora:nana_v10:1.2:FACEH>,(High Detail), smile,china dress,full body,high-heeled shoes,In the lobby,Screen window, corridor,Curtains of fluttering yarn,((Mottled light and shadow,warm light ,depth of field)),
"""
eg5 = """
(8k, RAW photo, best quality, masterpiece:1.2), (realistic, photo-realistic:1.37),1girl,cute, naked,standing,cityscape, night, rain, wet, professional lighting, photon mapping, radiosity, physically-based rendering
"""

neg_eg1 = 'worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry'

neg_eg2 = '(makeup:1.8),(smooth skin:1.3), anime, illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale:1.2)), futanari, full-package_futanari, penis_from_girl, newhalf, collapsed eyeshadow, multiple eyebrows, vaginas in breasts, pink hair, holes on breasts, fleckles, stretched nipples, gigantic penis, nipples on buttocks, analog, analogphoto, anal sex, signatre, logo, render,'
examples = [[eg1.strip(), '', neg_eg1, 137, 1, 1, 25],
            [eg2.strip(), '', neg_eg1, 137, 1, 1,25],
            [eg3.strip(), '', neg_eg1, 137, 1, 1,25],
            [eg4.strip(), '', neg_eg1, 137, 1, 1,25],
            [eg5.strip(), '', neg_eg1, 137, 1, 1,25]
            ]
# interface = gr.Interface(fn=to_black, inputs="image", outputs="image")
# interface.launch(share=True)
interface = gr.Interface(fn=text_to_image,
                         inputs=["text", "text", "text", gr.Slider(1, 20000, value=1, label="seed", info="随机种子"),
                                    gr.Slider(0, 1, value=1, label="text_lora_scale", info="text_lora_scale"),
                                    gr.Slider(0, 1, value=1, label="diffusion_lora_scale", info="diffusion_lora_scale"),
                                    gr.Slider(1, 50, value=1, label="steps", info="steps")],
                         outputs=["image"] * 4, examples=examples)
interface.launch(share=False)
