import os
import torch

def check_file_permissions(file_path):
    try:
        if os.access(file_path, os.R_OK | os.W_OK):
            print(f"Read and write permissions are available for {file_path}")
        else:
            raise PermissionError(f"Permission issue with {file_path}")
    except Exception as e:
        print(f"Error checking permissions for {file_path}: {e}")
        raise

def test_loading_file(file_path):
    try:
        check_file_permissions(file_path)
        print(f"Attempting to load file: {file_path}")
        data = torch.load(file_path)
        print(f"Successfully loaded file: {file_path}")
    except PermissionError as e:
        print(f"Permission denied when accessing {file_path}. Please check the file permissions.")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

# Paths to test
test_paths = [
    "C:\\Users\\Admin\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\models\\diffusers\\raemumix_v90\\model_index.json",
    "C:\\Users\\Admin\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\models\\diffusers\\raemumix_v90\\vae\\diffusion_pytorch_model.safetensors",
    "C:\\Users\\Admin\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\models\\diffusers\\raemumix_v90\\unet\\diffusion_pytorch_model.safetensors",
    "C:\\Users\\Admin\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\models\\diffusers\\raemumix_v90\\text_encoder\\model.safetensors",
]

for path in test_paths:
    test_loading_file(path)
