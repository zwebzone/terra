from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionXLPipeline
import torch
import os

lora_model_id = 'office31/AtoD/checkpoint-5000'
output_folder = ''

base_model_id = 'models/stabilityai/stable-diffusion-xl-base-1.0'
pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)


os.makedirs(output_folder, exist_ok=True)

# CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
# CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
#                'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
#                'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
#                'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
#                'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
#                'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
#                'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
            'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
            'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
            'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

def customize_string(s):
    s = s.lower()
    s = s.replace("_", " ")
    return s

for cls in CLASSES:
    # Create a negative prompt excluding the current class
    # negative_prompt = ', '.join([c for c in CLASSES if c != cls])
    cls_folder = os.path.join(output_folder, cls)
    os.makedirs(cls_folder, exist_ok=True)
    prompt1 = customize_string(cls)
    print(prompt1)
    for seed in range(50):
        print(cls_folder + '/' + f"{cls}_{seed}.png")
        prompt = f"A {prompt1}"
        image = pipe(prompt=prompt, generator=torch.Generator(device="cuda").manual_seed(seed), num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.9}).images[0]
        image.save(cls_folder + '/' + f"{cls}_seed{seed}.png")

