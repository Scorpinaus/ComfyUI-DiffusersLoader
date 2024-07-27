# __init__.py
from .combined_diffusers_loader import CombinedDiffusersLoader
from .unet_loader import DiffusersUNETLoader
from .clip_loader import DiffusersClipLoader
from .vae_loader import DiffusersVAELoader

NODE_CLASS_MAPPINGS = {
    "CombinedDiffusersLoader": CombinedDiffusersLoader,
    "DiffusersUNETLoader": DiffusersUNETLoader,
    "DiffusersClipLoader": DiffusersClipLoader,
    "DiffusersVAELoader": DiffusersVAELoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombinedDiffusersLoader": "Combined Diffusers Loader",
    "DiffusersUNETLoader": "Diffusers UNET Loader",
    "DiffusersClipLoader": "Diffusers CLIP Loader",
    "DiffusersVAELoader": "Diffusers VAE Loader"
}