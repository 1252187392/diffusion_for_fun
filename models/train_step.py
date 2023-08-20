#encoding: utf-8
import torch
import torch.nn.functional as F

def train_step(datas, vae, text_encoder, unet, noise_scheduler, weight_dtype):
    latents = vae.encode(datas[0].to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)

    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(datas[1])[0]
    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # Predict the noise residual and compute loss
    #print(encoder_hidden_states.dtype)
    encoder_hidden_states = encoder_hidden_states.to(torch.float16)
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss


def generate_image(pipeline, prompt, steps, seed, nums, device, save_name=None,
                   negative_prompt=None, scale=1, size=(512, 512), strength=0.7, guidance_scale=7.5, init_image=None):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []
    for _ in range(nums):
        image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                         generator=generator,
                         #cross_attention_kwargs={"scale": scale},
                         guidance_scale=guidance_scale,
                         height=size[1], width=size[0]).images[0]
        if save_name is not None:
            image.save(f'{save_name}_{_}.jpg')
        images.append(image)
    return images


def image_to_image(pipeline, prompt, steps, seed, nums, device, save_name=None,
                   negative_prompt=None, scale=1, size=(512, 512), strength=0.7, guidance_scale=7.5, init_image=None):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []
    for _ in range(nums):
        image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                         generator=generator,
                         #cross_attention_kwargs={"scale": scale},
                         strength=strength, guidance_scale=guidance_scale,
                         image=init_image).images[0]
        if save_name is not None:
            image.save(f'{save_name}_{_}.jpg')
        images.append(image)
    return images


def generate_SR_image(pipeline, prompt, steps, seed, nums, device, save_name=None,
                   negative_prompt=None, scale=1, size=(512,512), upscaler=None):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []
    for _ in range(nums):
        low_res_latents = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                         generator=generator,cross_attention_kwargs={"scale": scale},
                         height=size[1], width=size[0],output_type="latent"
                                ).images
        upscaled_image = upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        images.append(upscaled_image)
    return images