from .combined_diffusers_loader import CombinedDiffusersLoader
from .diffusers_clip_loader import DiffusersClipLoader
from .diffusers_unet_loader import DiffusersUNETLoader
from .diffusers_vae_loader import DiffusersVAELoader

NODE_CLASS_MAPPINGS = {
    "CombinedDiffusersLoader": CombinedDiffusersLoader,
    "DiffusersClipLoader": DiffusersClipLoader,
    "DiffusersUNETLoader": DiffusersUNETLoader,
    "DiffusersVAELoader": DiffusersVAELoader
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombinedDiffusersLoader": "Combined Diffusers Loader",
    "DiffusersClipLoader": "Diffusers CLIP Loader",
    "DiffusersUNETLoader": "Diffusers UNET Loader",
    "DiffusersVAELoader": "Diffusers VAE Loader"
}
