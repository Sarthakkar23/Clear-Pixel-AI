import urllib.request
import os

def download_dncnn_weights():
    # Download a different pretrained weights file for DnCNN from a reliable source
    url = "https://github.com/cszn/DnCNN/releases/download/v1.0/DnCNN-3.pth"
    save_path = os.path.join(os.path.dirname(__file__), "dncnn.pth")
    if os.path.exists(save_path):
        print(f"dncnn.pth already exists at {save_path}")
        return
    print(f"Downloading dncnn.pth weights to {save_path}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download complete.")

if __name__ == "__main__":
    download_dncnn_weights()
