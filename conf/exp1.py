
logging_dir = 'logs'
epochs = 50
batch_size = 3
load_weights = ""
save_weights = "weights/"
callback_frequency = 1
data_dirs = [
    {
        "dir": "datasets", # image dir, caption file should in it
        "repeat": 1, #  data repeat num in one epoch
        "max": 2000  # max data num
     },
]


generation_conf = {
    "prompt": "SKS1 face, araffed asian woman with a very big breast posing for a picture, perfect android girl, photo 8 k, artificial intelligence princess, « attractive, smooth waxy skin, lovely delicate face, very wet, glass skin, resting on chest, sichuan, fine bubbles, cutie, exhibant, suki",
    "steps": 25,
    "seed": 137,
    "nums": 3,
    "save_dir": "outputs/person/"
}

accelerator_conf = {
    'checkpoints_total_limit': None,
    'gradient_accumulation_steps': 1,
    'mixed_precision': 'fp16',  #["no", "fp16", "bf16"],
    'report_to': 'tensorboard',
    'logging_dir': 'logs'
}

model_conf = {
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "revision": None
}

optimizer_conf = {
    "learning_rate": 1e-4,
    "max_grad_norm": 1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-8
}


prompt_conf_dict = {
    'filter_tokens':[],
    'must_tokens': [],
    'prompts_templates': [],
    #'add_token': "fulijii face",
    'random_tags': 0.2   # ratio of tokens in  prompt to drop
}