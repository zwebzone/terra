# translate images into npy
from pnp_pipeline import SDXLDDIMPipeline
import torch

pipeline = SDXLDDIMPipeline.from_pretrained(
    "models/stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

lora_model_id = 'checkpoint'

base_model_id = 'models/stabilityai/stable-diffusion-xl-base-1.0'
lora_sd, _ = pipeline.lora_state_dict(lora_model_id)

time_t = 0

def target_layer_from_sd_name(k):
  # They use slightly different naming schemes for attn processors vs the rest
  if '.processor.to_' in k:
    target_layer = k.split("processor.to_")[0] + k.split(".processor.")[1].split("_lora")[0]
    target_layer = target_layer.replace("to_out", "to_out[0]")
  else:
    target_layer = k.split(".lora.")[0]
  # Replace '.1.' with '[1]' and so on:
  for i in range(10):
    target_layer = target_layer.replace(f".{i}", f"[{i}]")
  # Return (skipping the first 'unet.' in this case):
  return target_layer

for k, v in lora_sd.items():
    if k.endswith("down.weight"):
        a1 = lora_sd[k]
        b1 = lora_sd[k.replace("down.weight", "up.weight")]
        omegas_weights = lora_sd[k.replace("down.weight", "diagnal.omegas_weights")]
        matrix = omegas_weights * time_t + torch.eye(32)

        dW = b1 @ matrix.T @ a1

        target_layer = target_layer_from_sd_name(k)
        eval(f'pipeline.{target_layer}').weight.data += dW.to(pipeline.device)


import os
import torch
from torchvision import transforms
import PIL.Image
import numpy as np

source_dir = ""

target_dir = ""

def get_img(img_path, resolution=1024):
    img = PIL.Image.open(img_path)

    w, h = img.size
    if w > h:
        new_w, new_h = resolution, int(resolution * h / w)
    else:
        new_w, new_h = int(resolution * w / h), resolution

    transform = transforms.Compose([
        transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = transform(img)
    return img.unsqueeze(0)


if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    target_category_dir = os.path.join(target_dir, category)
    
    category_name = category.lower().replace('_', ' ')
    
    prompt = "A " + category_name
    
    if not os.path.exists(target_category_dir):
        os.makedirs(target_category_dir)
    
    for img_file in os.listdir(category_path):
        img_path = os.path.join(category_path, img_file)
        init_img = get_img(img_path)
        x = pipeline(prompt=prompt, num_inference_steps=25, negative_prompt=None, image=init_img)
        x_prev_latent = x[0].clone()
        
        npy_path = os.path.join(target_category_dir, img_file.replace('.jpg', '.npy'))
        np.save(npy_path, x_prev_latent.cpu().numpy())