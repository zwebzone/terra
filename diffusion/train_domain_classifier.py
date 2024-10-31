import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms

import torchvision.models as models

class CustomResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # for param in resnet.parameters():
        #     param.requires_grad = False

        self.features = nn.Sequential(*list(resnet.children())[:-2])


        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Adaptive average pooling to reduce spatial dimensions to 1x1
            nn.Flatten(),  # Flatten the features to a vector
            nn.Linear(2048, num_classes)  # Linear layer to transform features to the desired output size
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

def calculate_kl(data):
    mu = data.mean(dim=0)
    sigma = torch.matmul((data - mu).T, (data - mu)) / (data.size(0) - 1)
    
    n = len(mu)
    norm_squared = torch.norm(mu)**2
    det_sigma = torch.det(sigma)
    tr_sigma = torch.trace(sigma)
    
    kl_divergence = 0.5 * (norm_squared - torch.log(det_sigma) + tr_sigma - n)
    return kl_divergence

# %%
class ContrastiveLoss:
    def __init__(self, margin=1.0):
        self.margin = margin
        self.record = dict()

    def calculate_loss(self, cls, pred_time):
        cls = torch.tensor(cls, dtype=torch.long)

        positive_pairs = [(i, j) for i in range(len(cls)) for j in range(i + 1, len(cls)) if cls[i] == cls[j]]
        negative_pairs = [(i, j) for i in range(len(cls)) for j in range(i + 1, len(cls)) if cls[i] != cls[j]]

        positive_losses = []
        for pair in positive_pairs:
            emb1, emb2 = pred_time[pair[0]], pred_time[pair[1]]
            dist = F.pairwise_distance(emb1.unsqueeze(0), emb2.unsqueeze(0))
            positive_losses.append(dist)
        
        if len(positive_losses) == 0:
            positive_loss = 0
        else:
            positive_loss = torch.mean(torch.stack(positive_losses))

        negative_losses = []
        for pair in negative_pairs:
            emb1, emb2 = pred_time[pair[0]], pred_time[pair[1]]
            dist = F.pairwise_distance(emb1.unsqueeze(0), emb2.unsqueeze(0))
            negative_losses.append(torch.clamp(self.margin - dist, min=0))

        if len(negative_losses) == 0:
            negative_loss = 0
        else:
            negative_loss = torch.mean(torch.stack(negative_losses))

        total_loss = positive_loss + negative_loss
        
        for c in range(len(cls)):
            intc = int(cls[c])
            if intc not in self.record:
                self.record[intc] = pred_time[c].detach().cpu()
            else:
                self.record[intc] = 0.8 * self.record[intc] + 0.2 * pred_time[c].detach().cpu()

        return total_loss
    
    def print_record(self):
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        for c in self.record:
            print(c, self.record[c]) 
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# %%
def changename(name):
    if name == 'marker':
        return 'marker pen'
    elif name == 'mouse':
        return 'computer mouse'
    else:
        return name

# %%
def doublePath(path):
    arr = []
    for l in list(Path(path).iterdir()):
        arr.extend(list(Path(l).iterdir()))
    return arr

class DreamBoothDomainDataset():
    def __init__(
        self,
        instance_prompt,
        class_data_root=None,
        class_num=None,
        size=224,
        center_crop=False,
        randomimage=0.75,
    ):
        self.size = size
        self.randomimage = randomimage
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt


        self.instance_data_root1 = "office_home/Art"
        self.instance_data_root2 = "office_home/Clipart"
        self.instance_data_root3 = "office_home/Product"
        # self.instance_data_root3 = "office_home/Real_World"

        # self.instance_data_root1 = "PACS/photo"
        # self.instance_data_root2 = "PACS/art_painting"
        # self.instance_data_root3 = "PACS/cartoon"
        # self.instance_data_root3 = "PACS/sketch"

        # self.instance_data_root1 = "VLCS/VOC2007"
        # self.instance_data_root2 = "VLCS/LabelMe"
        # self.instance_data_root3 = "VLCS/Caltech101"
        # self.instance_data_root3 = "VLCS/SUN09"


        self.instance_images = [path for path in doublePath(self.instance_data_root1)]
        self.instance_images.extend([path for path in doublePath(self.instance_data_root2)])
        self.instance_images.extend([path for path in doublePath(self.instance_data_root3)])

        self.custom_instance_prompts = ["A %s" % changename(path.parent.name.replace("_", " ").lower()) for path in doublePath(self.instance_data_root1)]
        self.custom_instance_prompts.extend(["A %s" % changename(path.parent.name.replace("_", " ").lower()) for path in doublePath(self.instance_data_root2)])
        self.custom_instance_prompts.extend(["A %s" % changename(path.parent.name.replace("_", " ").lower()) for path in doublePath(self.instance_data_root3)])


        self.styinds = [0 for path in doublePath(self.instance_data_root1)]
        self.styinds.extend([1 for path in doublePath(self.instance_data_root2)])
        self.styinds.extend([2 for path in doublePath(self.instance_data_root3)])


        self.num_instance_images = len(self.instance_images)

        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self._length)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (size, size), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images[index % self.num_instance_images])
        
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt"] = self.custom_instance_prompts[index % self.num_instance_images]
        example["style_index"] = self.styinds[index % self.num_instance_images]

        return example


# %%
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
import torch

# %%
train_dataset = DreamBoothDomainDataset(
        size=224,
        instance_prompt = None
)

# %%
def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    styles = [example["style_index"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.float()

    batch = {"pixel_values": pixel_values, "prompts": prompts, "styles": styles}
    return batch

train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, False),
        num_workers=4,
    )

# %%
MODEL_NAME="/data/models/stabilityai/stable-diffusion-xl-base-1.0"
vae = AutoencoderKL.from_pretrained("%s/vae" % MODEL_NAME, torch_dtype=torch.float32).cuda()

# %%
vae.requires_grad_(False)

# %%
t_pred_cnn = CustomResNet(num_classes=2).cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, t_pred_cnn.parameters()), lr=1e-5)

# %%
loss_calculator = ContrastiveLoss()

# %%
Image.MAX_IMAGE_PIXELS = None

# subfolder = 'PACS'
subfolder = 'VLCS'

# %%
for e in range(10):
    for step, batch in enumerate(train_dataloader):
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        styles = batch["styles"]
        t1 = t_pred_cnn(pixel_values)
        # kl_loss = calculate_kl(t1)
        cons_loss = loss_calculator.calculate_loss(styles, t1)
        # loss = cons_loss + kl_loss
        loss = cons_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 10 == 0:
            print(cons_loss.item())
            loss_calculator.print_record()
    # model_filename = f"t_pred_cnn_office_Art_epoch_{e+1}_head_KL_1e4.pt"
    if (e+1) % 5 == 0:
        model_filename = os.path.join(subfolder, f"t_pred_cnn_office_Real_World_epoch_{e+1}_KL_1e5.pt")
        # model_filename = os.path.join(subfolder, f"t_pred_cnn_S_epoch_{e+1}_KL_1e5.pt")
        torch.save(t_pred_cnn.state_dict(), model_filename)
        print(f'Model saved as {model_filename}!')


