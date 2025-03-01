import os
from tkinter import filedialog
import torch
import lpips  # Learned Perceptual Image Patch Similarity
from PIL import Image
from torchvision import transforms


# Load the LPIPS model
# You can use 'vgg', 'alex', or 'squeeze' as the backbone
lpips_model = lpips.LPIPS(net='vgg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips_model.to(device)

# Function to load and preprocess an image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize for consistent comparison
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load images
root = filedialog.askdirectory()

paths = []
for path in os.listdir(root):
    base, ext = os.path.splitext(path)
    if ext in [".png", ".jpg", ".jpeg"]:
        paths.append(os.path.join(root, path))

scores = []
for i, path1 in enumerate(paths[:-1]):
    for j, path2 in enumerate(paths[i+1:]):
        image1 = preprocess_image(path1)
        image2 = preprocess_image(path2)

        image1 = image1.to(device)
        image2 = image2.to(device)

        # Compute the perceptual similarity
        similarity_score = lpips_model(image1, image2)
        
        scores.append((similarity_score.item(), path1, path2))

scores.sort(key=lambda x: x[0])

# print(f"Perceptual Similarity Score: {similarity_score.item():.4f}")

print("Most similar images:")
for score, path1, path2 in scores[:5]:
    print(f"{path1} and {path2} with score {score:.4f}")