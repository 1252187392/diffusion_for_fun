
logging_dir = 'logs'
epochs = 10  # total epochs
start_epoch = 0  # start epoch , little than epochs
batch_size = 8  # batch_size
load_weights = None # dir of load weights
load_weights_web = "weights/exp/9"
save_weights = "/content/drive/MyDrive/doupo/"  #
callback_frequency = 1        # generate image when finetuning
data_dirs = [
    {"dir": "./styles", "repeat": 100, "max": 3000},
]

generation_conf = {
    "prompt": "a painting in the style of  <sks2>",
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
    'prompts_templates': ["a painting in the style of {}",
        "a rendering in the style of {}",
        "a cropped painting in the style of {}",
        "the painting in the style of {}",
        "a clean painting in the style of {}",
        "a dirty painting in the style of {}",
        "a dark painting in the style of {}",
        "a picture in the style of {}",
        "a cool painting in the style of {}",
        "a close-up painting in the style of {}",
        "a bright painting in the style of {}",
        "a cropped painting in the style of {}",
        "a good painting in the style of {}",
        "a close-up painting in the style of {}",
        "a rendition in the style of {}",
        "a nice painting in the style of {}",
        "a small painting in the style of {}",
        "a weird painting in the style of {}",
        "a large painting in the style of {}",
    ],
    'add_token': "<sks2>",
    'random_tags': 0.,
    'use_default': True
}
prompt_conf_dict['prompts_templates'] = [x.format(prompt_conf_dict['add_token']) for x in prompt_conf_dict['prompts_templates'] ]
