import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Load MiDaS model (assuming you have it installed and set up)
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
model.eval()

# Transformation for the model input
transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img_path = "./data_lr_2x/"
depth_path = "./depth_data/"
os.makedirs(depth_path, exist_ok=True)

for img_file in os.listdir(img_path):
    img = Image.open(os.path.join(img_path, img_file))
    input_batch = transform(img).unsqueeze(0)

    # Generate depth map
    with torch.no_grad():
        depth = model(input_batch).squeeze().numpy()

    # Save depth as .npy file
    depth_file_path = os.path.join(
        depth_path, img_file.replace(".jpg", ".npy").replace(".png", ".npy")
    )
    np.save(depth_file_path, depth)
    print(f"Saved depth map to {depth_file_path}")
