from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionXLPipeline
import torch
import os


lora_model_id = 'checkpoint-5000'

base_model_id = 'models/stabilityai/stable-diffusion-xl-base-1.0'

# pipe.load_lora_weights(lora_model_id)

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


output_folder = ''


os.makedirs(output_folder, exist_ok=True)

CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

def customize_string(s):
    s = s.lower()
    s = s.replace("_", " ")
    return s


import numpy as np

time_t_values1 = np.arange(-2.0, 2.1, 0.5)
time_t_values2 = np.arange(-2.0, 2.1, 0.5)
time_t_index = -1
# for time_t_index, time_t in enumerate(time_t_values):
for time_t_x in time_t_values1:
  for time_t_y in time_t_values2:
    pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    lora_sd, _ = pipe.lora_state_dict(lora_model_id)
    for k, v in lora_sd.items():
        if k.endswith("down.weight"):
            a1 = lora_sd[k]
            b1 = lora_sd[k.replace("down.weight", "up.weight")]
            omegas_weights = lora_sd[k.replace("down.weight", "diagnal.omegas_weights")]

            weights = lora_sd[k.replace("down.weight", "diagnal.weights")]
            time_t = torch.tensor([time_t_x, time_t_y], dtype=torch.float)
            time_t = time_t.reshape(2,1,1)
            cos_v = torch.sum(weights * torch.cos(time_t * omegas_weights), dim=0)
            sin_v = torch.sum(weights * torch.sin(time_t * omegas_weights), dim=0)
            mask = torch.eye(32)
            matrix = (1-mask) * sin_v + mask * cos_v
            dW = b1 @ (matrix.T) @ a1
            target_layer = target_layer_from_sd_name(k)
            eval(f'pipe.{target_layer}').weight.data += dW.to(pipe.device)

    time_t_index += 1

    for cls in CLASSES:
        cls_folder = os.path.join(output_folder, cls)
        os.makedirs(cls_folder, exist_ok=True)
        prompt1 = customize_string(cls)
        print(prompt1)
        
        for seed_increment in range(5): 
          seed = time_t_index * 5 + seed_increment
          print(cls_folder + '/' + f"{cls}_time_t_{time_t_x:.2f}_{time_t_y:.2f}_seed{seed}.png")
          prompt = f"A {prompt1}"
          image = pipe(prompt=prompt, generator=torch.Generator(device="cuda").manual_seed(seed), num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.9}).images[0]
          image.save(cls_folder + '/' + f"{cls}_time_t_{time_t_x:.2f}_{time_t_y:.2f}_seed{seed}.png")