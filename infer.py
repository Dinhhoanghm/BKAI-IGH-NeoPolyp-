import argparse
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Argument parser
parser = argparse.ArgumentParser(description="Run inference on an image using a pretrained segmentation model.")
parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
parser.add_argument('--checkpoint', type=str, default='model.pth', help="Path to the model checkpoint.")
parser.add_argument('--output_dir', type=str, default='result', help="Directory to save the output image.")
args = parser.parse_args()

# Load model
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint.get("model", checkpoint))
model.to(device)
model.eval()

# Transformation
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Color dictionary for visualization
color_dict = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}

# Function to map mask to RGB colors
def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, color in color_dict.items():
        output[mask == k] = color
    return output

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load and preprocess image
if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"Image path '{args.image_path}' does not exist.")
ori_img = cv2.imread(args.image_path)
if ori_img is None:
    raise ValueError(f"Failed to read the image at '{args.image_path}'.")
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
ori_h, ori_w = ori_img.shape[:2]

# Transform and inference
img = cv2.resize(ori_img, (256, 256))
transformed = val_transform(image=img)
input_img = transformed["image"].unsqueeze(0).to(device)

with torch.no_grad():
    output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)

# Post-process output
mask = cv2.resize(output_mask, (ori_w, ori_h))
mask = np.argmax(mask, axis=2)
mask_rgb = mask_to_rgb(mask, color_dict)

# Save segmented image
output_path = os.path.join(args.output_dir, "segmented_image_color.png")
mask_rgb_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, mask_rgb_bgr)

print(f"Segmented image saved at {output_path}")
