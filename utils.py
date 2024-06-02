import os
import json
import torch
from safetensors import safe_open

def load_diffusers_model(model_path, output_vae=True, output_clip=True):
    try:
        model = load_model_from_path(model_path)
        vae = load_vae_from_path(model_path) if output_vae else None
        clip = load_text_encoder_from_path(model_path, encoder_id=None) if output_clip else None
        return model, vae, clip
    except Exception as e:
        print(f"Error loading diffusers model from {model_path}: {e}")
        raise

def load_diffusers_sdxl_model(model_path, output_vae=True, output_clip=True, embedding_directory=None):
    try:
        model = load_model_from_path(model_path)
        vae = load_vae_from_path(model_path) if output_vae else None
        clip = load_text_encoder_from_path(model_path, encoder_id=1) if output_clip else None
        text_encoder_2 = load_text_encoder_from_path(model_path, encoder_id=2)
        
        if embedding_directory:
            # Load or handle embeddings if applicable
            pass

        return model, vae, clip, text_encoder_2
    except Exception as e:
        print(f"Error loading SDXL diffusers model from {model_path}: {e}")
        raise

def load_model_from_path(model_path):
    try:
        print(f"Loading model from {model_path}")
        check_file_permissions(model_path)
        if model_path.endswith('.safetensors'):
            model = load_safetensors(model_path)
        else:
            model = load_json(model_path)
        return model
    except PermissionError as e:
        print(f"Permission denied when accessing {model_path}. Please check the file permissions.")
        raise
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

def load_vae_from_path(model_path):
    try:
        print(f"Loading VAE from {model_path}")
        vae_path = os.path.join(model_path, "vae", "diffusion_pytorch_model.safetensors")
        check_file_permissions(vae_path)
        vae = load_safetensors(vae_path)
        return vae
    except PermissionError as e:
        print(f"Permission denied when accessing {vae_path}. Please check the file permissions.")
        raise
    except Exception as e:
        print(f"Error loading VAE from {vae_path}: {e}")
        raise

def load_clip_from_path(model_path):
    try:
        print(f"Loading CLIP model from {model_path}")
        clip_path = os.path.join(model_path, "clip", "diffusion_pytorch_model.safetensors")
        check_file_permissions(clip_path)
        clip = load_safetensors(clip_path)
        return clip
    except PermissionError as e:
        print(f"Permission denied when accessing {clip_path}. Please check the file permissions.")
        raise
    except Exception as e:
        print(f"Error loading CLIP model from {clip_path}: {e}")
        raise

def load_text_encoder_from_path(model_path, encoder_id=None):
    try:
        if encoder_id is not None:
            encoder_path = os.path.join(model_path, f"text_encoder_{encoder_id}", "model.safetensors")
            print(f"Loading text encoder {encoder_id} from {encoder_path}")
        else:
            encoder_path = os.path.join(model_path, "text_encoder", "model.safetensors")
            print(f"Loading text encoder from {encoder_path}")
        
        check_file_permissions(encoder_path)
        text_encoder = load_safetensors(encoder_path)
        return text_encoder
    except PermissionError as e:
        print(f"Permission denied when accessing {encoder_path}. Please check the file permissions.")
        raise
    except Exception as e:
        print(f"Error loading text encoder from {encoder_path}: {e}")
        raise

def load_json(json_path):
    try:
        print(f"Loading JSON from {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {json_path}: {e}")
        raise

def load_safetensors(file_path):
    try:
        print(f"Loading safetensors from {file_path}")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}
    except Exception as e:
        print(f"Error loading safetensors from {file_path}: {e}")
        raise

def check_file_permissions(file_path):
    try:
        if os.access(file_path, os.R_OK | os.W_OK):
            print(f"Read and write permissions are available for {file_path}")
        else:
            raise PermissionError(f"Permission issue with {file_path}")
    except Exception as e:
        print(f"Error checking permissions for {file_path}: {e}")
        raise
