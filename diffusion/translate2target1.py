from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
import numpy as np
import os
import gc
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "models/stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

time_t = 1.0

def target_layer_from_sd_name(k):
  if '.processor.to_' in k:
    target_layer = k.split("processor.to_")[0] + k.split(".processor.")[1].split("_lora")[0]
    target_layer = target_layer.replace("to_out", "to_out[0]")
  else:
    target_layer = k.split(".lora.")[0]
  for i in range(10):
    target_layer = target_layer.replace(f".{i}", f"[{i}]")
  return target_layer

lora_model_id = 'checkpoint'


lora_sd, _ = pipeline.lora_state_dict(lora_model_id)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()

    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    if len(size) == 3:
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    else:
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

for k, v in lora_sd.items():
    if k.endswith("down.weight"):
        a1 = lora_sd[k]
        b1 = lora_sd[k.replace("down.weight", "up.weight")]
        omegas_weights = lora_sd[k.replace("down.weight", "diagnal.omegas_weights")]
        matrix = omegas_weights * time_t + torch.eye(8)

        dW = b1 @ matrix.T @ a1

        target_layer = target_layer_from_sd_name(k)
        eval(f'pipeline.{target_layer}').weight.data += dW.to(pipeline.device)

npy_dir = ''
image_dir = ""

if not os.path.exists(image_dir):
    os.makedirs(image_dir)
with torch.no_grad():
    for category in os.listdir(npy_dir):
        category_path = os.path.join(npy_dir, category)
        target_category_dir = os.path.join(image_dir, category)
        
        if not os.path.exists(target_category_dir):
            os.makedirs(target_category_dir)
        
        for npy_file in os.listdir(category_path):
            
            npy_path = os.path.join(category_path, npy_file)
            x_prev_latent = np.load(npy_path)
            x_prev_latent = torch.tensor(x_prev_latent, device='cuda')

            category_name = category.lower().replace('_', ' ')
            prompt = "A " + category_name

            random1 = torch.randn_like(x_prev_latent)
            image = pipeline(prompt=prompt, num_inference_steps=25, latents=x_prev_latent, guidance_scale=1, cross_attention_kwargs={"scale": 0.9}).images[0]


            image_save_path = os.path.join(target_category_dir, npy_file.replace('.npy', '.png'))
            image.save(image_save_path)
