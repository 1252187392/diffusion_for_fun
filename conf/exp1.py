
logging_dir = 'logs'
epochs = 20  # total epochs
max_step = 2000 # total step,  训练达到epochs或者max_step停止
start_epoch = 0  # start epoch , little than epochs
batch_size = 6  # batch_size
load_weights = None # dir of load weights
load_weights_web = "weights/debug/19" # 训练完后的模型，web使用
save_weights = "/content/drive/MyDrive/faces/"  # for colab
save_weights = "weights/debug"
callback_frequency = 4        # generate image when finetuning
rank = 16 # lora rank

data_dirs = [
    {"dir": "I:\\datasets\\output1", "repeat": 10}
]

generation_conf = {
    "prompt": "a photo of <sks1>",
    "steps": 25,
    "seed": 137,
    "nums": 3,
    "save_dir": "outputs/"
}

accelerator_conf = {
    'checkpoints_total_limit': None,
    'gradient_accumulation_steps': 3,
    'mixed_precision': 'fp16',  #["no", "fp16", "bf16"],
    'report_to': 'tensorboard',
    'logging_dir': 'logs'
}

model_conf = {
    #"pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "pretrained_model_name_or_path": "I:\models\huggingface\\hub\\models--runwayml--stable-diffusion-v1-5\\snapshots\\aa9ba505e1973ae5cd05f5aedd345178f52f8e6a",
    "revision": None
}

optimizer_conf = {
    "learning_rate": 5e-5,
    "learning_rate_unet": 5e-4,
    "max_grad_norm": 1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-8
}

prompt_conf_dict = {
    'filter_tokens':[],
    'must_tokens': [],
    'prompts_templates': ['a photo of {}',
      "a photo of a {}",
      "a rendering of a {}",
      "a cropped photo of the {}",
      "the photo of a {}",
      "a photo of a clean {}",
      "a close-up photo of a {}",
      "a cropped photo of a {}",
      "a photo of the {}",
      "a good photo of the {}",
      "a close-up photo of the {}",
    ],
    'add_token': "<sks1>",
    'random_tags': 0.,
    'use_default': False # 使用conf中的模版，而不使用caption文件
}

prompt_conf_dict['prompts_templates'] = [x.format(prompt_conf_dict['add_token']) for x in prompt_conf_dict['prompts_templates'] ]