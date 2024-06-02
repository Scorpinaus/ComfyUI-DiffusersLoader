import os
import utils

# Define the base model path
base_model_path = "C:\\Users\\Admin\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\models\\diffusers\\raemumix_v90"

# Test loading the main model
try:
    model = utils.load_model_from_path(os.path.join(base_model_path, "model_index.json"))
    print("Main model loaded successfully.")
except Exception as e:
    print(f"Error loading main model: {e}")

# Test loading the VAE model
try:
    vae = utils.load_vae_from_path(base_model_path)
    print("VAE model loaded successfully.")
except Exception as e:
    print(f"Error loading VAE model: {e}")

# Test loading the CLIP model
try:
    clip = utils.load_text_encoder_from_path(base_model_path, encoder_id=None)
    print("CLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")

# Test loading the text encoders for SDXL
try:
    text_encoder_1 = utils.load_text_encoder_from_path(base_model_path, 1)
    print("Text encoder 1 loaded successfully.")
except Exception as e:
    print(f"Error loading text encoder 1: {e}")

try:
    text_encoder_2 = utils.load_text_encoder_from_path(base_model_path, 2)
    print("Text encoder 2 loaded successfully.")
except Exception as e:
    print(f"Error loading text encoder 2: {e}")
