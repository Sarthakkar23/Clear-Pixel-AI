import torch
import numpy as np
from PIL import Image
from ai_models import DnCNN

def test_dncnn_inference():
    model = DnCNN()
    weights_path = "clear_pixel_ai/backend/dncnn.pth"
    try:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    # Create a synthetic noisy image tensor
    clean_image = torch.ones(1, 3, 180, 180) * 0.5  # Gray image
    noise = torch.randn_like(clean_image) * 0.1
    noisy_image = clean_image + noise
    noisy_image = torch.clamp(noisy_image, 0., 1.)

    with torch.no_grad():
        denoised = model(noisy_image)

    print(f"Noisy image mean: {noisy_image.mean().item()}")
    print(f"Denoised image mean: {denoised.mean().item()}")
    diff = (noisy_image - denoised).abs().mean().item()
    print(f"Mean absolute difference between noisy and denoised: {diff}")

if __name__ == "__main__":
    test_dncnn_inference()
