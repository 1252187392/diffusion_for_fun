
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from accelerate import Accelerator
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer


def init_models(model_conf):
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(model_conf["pretrained_model_name_or_path"], subfolder="scheduler",
                                                    mirror='tuna')
    tokenizer = CLIPTokenizer.from_pretrained(
        model_conf["pretrained_model_name_or_path"], subfolder="tokenizer", revision=model_conf["revision"], mirror='tuna'
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_conf["pretrained_model_name_or_path"], subfolder="text_encoder", revision=model_conf["revision"], mirror='tuna'
    )
    vae = AutoencoderKL.from_pretrained(model_conf["pretrained_model_name_or_path"], subfolder="vae",
                                        revision=model_conf["revision"],mirror='tuna')
    unet = UNet2DConditionModel.from_pretrained(
        model_conf["pretrained_model_name_or_path"], subfolder="unet", revision=model_conf["revision"],mirror='tuna'
    )
    return noise_scheduler, tokenizer, text_encoder, vae, unet


def init_accelerator(conf):
    accelerator_project_config = ProjectConfiguration(total_limit=conf['checkpoints_total_limit'])

    accelerator = Accelerator(
        gradient_accumulation_steps=conf['gradient_accumulation_steps'],
        mixed_precision=conf['mixed_precision'],
        log_with=conf['report_to'],
        logging_dir=conf['logging_dir'],
        #project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    return accelerator

def set_unet_lora(unet, r=4):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,rank=r)

    unet.set_attn_processor(lora_attn_procs)


