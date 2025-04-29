import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ai_models import DnCNN

class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, noise_std=25, image_size=(180, 180)):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.noise_std = noise_std / 255.0  # Normalize noise std to [0,1]
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        if self.transform:
            image = self.transform(image)
        noise = torch.randn_like(image) * self.noise_std
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0., 1.)
        return noisy_image, image

def train_dncnn(data_dir, epochs=10, batch_size=16, learning_rate=1e-3, save_path='dncnn.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    dataset = NoisyImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = DnCNN()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (noisy_imgs, clean_imgs) in enumerate(dataloader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.6f}")
                running_loss = 0.0

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Set the training images directory path
    data_dir = "/Users/sarthakkar/Documents/Clear Pixel AI/Training Image"
    train_dncnn(data_dir)
