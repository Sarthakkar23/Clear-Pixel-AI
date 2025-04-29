import torch
from ai_models import DnCNN

def test_model():
    model = DnCNN()
    model.eval()
    input_tensor = torch.rand(1, 3, 180, 180)  # Random input
    with torch.no_grad():
        output = model(input_tensor)
    print("Input tensor mean:", input_tensor.mean().item())
    print("Output tensor mean:", output.mean().item())
    diff = (input_tensor - output).abs().mean().item()
    print("Mean absolute difference between input and output:", diff)

if __name__ == "__main__":
    test_model()
