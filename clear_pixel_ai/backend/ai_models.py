import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Define DnCNN model architecture (simplified version)
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(64, 64, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, 3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning

# Utility function to load image and convert to tensor with normalization
def image_to_tensor(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Utility function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu()
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV compatibility

# Load pretrained DnCNN model weights from local file
def load_dncnn_model(weights_path):
    model = DnCNN()
    if os.path.exists(weights_path):
        try:
            # Use weights_only=False to avoid PyTorch 2.6+ loading issues
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            print("DnCNN model loaded successfully.")
        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")
            model = None
    else:
        print(f"Pretrained weights not found at {weights_path}. Model will not denoise.")
        model = None
    return model

# Initialize DnCNN model
dncnn_weights_path = os.path.join(os.path.dirname(__file__), 'dncnn.pth')
dncnn_model = load_dncnn_model(dncnn_weights_path)

def ai_denoise(image):
    if dncnn_model is None:
        print("DnCNN model not loaded, returning original image")
        return image
    image_rgb = Image.fromarray(image[:, :, ::-1])  # Convert BGR to RGB PIL Image
    input_tensor = image_to_tensor(image_rgb)
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    with torch.no_grad():
        output = dncnn_model(input_tensor)
    print(f"Output tensor shape: {output.shape}, dtype: {output.dtype}")
    output_image = tensor_to_image(output)
    return output_image

# Placeholder classes for other models (unchanged)
class DeOldify:
    def __init__(self):
        pass
    def restore(self, image):
        return image

class ESRGAN:
    def __init__(self):
        pass
    def super_resolve(self, image):
        return image

class LaMa:
    def __init__(self):
        pass
    def inpaint(self, image, mask):
        return image

class HED:
    def __init__(self):
        pass
    def detect_edges(self, image):
        return image

deoldify_model = DeOldify()
esrgan_model = ESRGAN()
lama_model = LaMa()
hed_model = HED()

def ai_restore(image):
    return deoldify_model.restore(image)

def ai_super_resolve(image):
    return esrgan_model.super_resolve(image)

def ai_inpaint(image, mask):
    return lama_model.inpaint(image, mask)

def ai_edge_detection(image):
    return hed_model.detect_edges(image)
