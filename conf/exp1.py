
logging_dir = 'logs'
epochs = 10  # total epochs
start_epoch = 0  # start epoch , little than epochs
batch_size = 8  # batch_size
load_weights = None # dir of load weights
save_weights = "weights/exp/"  #
callback_frequency = 1        # generate image when finetuning
data_dirs = [
    {"dir": "./faces", "repeat": 10, "max": 3000},
]


generation_conf = {
    "prompt": "a photo of <SKS> face",
    "steps": 25,
    "seed": 137,
    "nums": 3,
    "save_dir": "outputs/"
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
    "learning_rate": 5e-5,
    "max_grad_norm": 1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-8
}


prompt_conf_dict = {
    'filter_tokens':[],
    'must_tokens': [],
    'prompts_templates': ['a photo of <s1> face'],
    #'add_token': "fulijii face",
    'random_tags': 0.,
    'use_default': True
}