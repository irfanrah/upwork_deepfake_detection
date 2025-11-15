import torch
import torch.nn as nn
from models.dinov2_models_lora import DINOv2ModelWithLoRA
from torchvision import transforms
from PIL import Image
import os

# Initialize model
model = DINOv2ModelWithLoRA(name="dinov2_vitl14", lora_rank=8, lora_alpha=1, lora_targets=None)
THRESHOLD = 0.5

# Load checkpoint first
ckpt = "pretrained/ckpt.pth"

device = 'cuda:0'
checkpoint = torch.load(ckpt, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)


# Set model to training mode for training
model.eval()

# Define the image transformations
test_transform = transforms.Compose([
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# Load the image using PIL
folder_path = "/home/kurnianto/code/deepfake_detection/assets/images"
image_classes = os.listdir(folder_path)
for img_class in image_classes:
    images = sorted(os.listdir(os.path.join(folder_path, img_class)))
    for img in images:
        image_path = os.path.join(folder_path, img_class, img)
        image = Image.open(image_path).convert("RGB")
        input_image = test_transform(image).unsqueeze(0).to(device)
        with torch.no_grad(): 
            output = model(input_image).sigmoid().flatten()
        score = output.item()  # assume 0~1
        percent_fake = score * 100
        percent_real = (1 - score) * 100

        prediction = "fake" if score > THRESHOLD else "real"

        print(
            f"Prediction: {prediction} "
            f"(real: {percent_real:.2f}%, fake: {percent_fake:.2f}%) ; GT {img_class} : image {img}"
        )
