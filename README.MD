# diffusion for fun
simple code to finetune stable diffusion with lora    
modify from huggingface diffusion    
support windows
## requirements
* pytorch
```buildoutcfg
pip install -i requirements.txt
```

## run

### step1
modify params in conf/exp1.py to setting experiments

data_dir format
```buildoutcfg
--image_dir
----image1.jpg  
----image1_format.txt    #the prompt
```

### step2
```buildoutcfg
accelerate launch .\finetune_lora.py
```

### step3
run app/web_demo with new model



