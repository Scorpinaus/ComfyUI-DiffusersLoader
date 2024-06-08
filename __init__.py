from .sd15_unet_loader import SD15UNETLoader
from .sd15_vae_loader import SD15VAELoader
from .sd15_clip_loader import SD15CLIPLoader
from .sdxl_unet_loader import SDXLUNETLoader
from .sdxl_vae_loader import SDXLVAELoader
from .sdxl_clip_loader import SDXLCLIPLoader
from .sd15_combined_loader import CombinedDiffusersSD15Loader
from .sdxl_combined_loader import CombinedDiffusersSDXLLoader
from .combined_diffusers_loader import CombinedDiffusersLoader
from .diffusers_clip_loader import DiffusersClipLoader
from .diffusers_unet_loader import DiffusersUNETLoader
from .diffusers_vae_loader import DiffusersVAELoader

NODE_CLASS_MAPPINGS = {
    "SD15UNETLoader": SD15UNETLoader,
    "SD15VAELoader": SD15VAELoader,
    "SD15CLIPLoader": SD15CLIPLoader,
    "SDXLUNETLoader": SDXLUNETLoader,
    "SDXLVAELoader": SDXLVAELoader,
    "SDXLCLIPLoader": SDXLCLIPLoader,
    "CombinedDiffusersSD15Loader": CombinedDiffusersSD15Loader,
    "CombinedDiffusersSDXLLoader": CombinedDiffusersSDXLLoader,
    "CombinedDiffusersLoader": CombinedDiffusersLoader,
    "DiffusersClipLoader": DiffusersClipLoader,
    "DiffusersUNETLoader": DiffusersUNETLoader,
    "DiffusersVAELoader": DiffusersVAELoader
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD15UNETLoader": "SD1.5 UNET Loader",
    "SD15VAELoader": "SD1.5 VAE Loader",
    "SD15CLIPLoader": "SD1.5 CLIP Loader",
    "SDXLUNETLoader": "SDXL UNET Loader",
    "SDXLVAELoader": "SDXL VAE Loader",
    "SDXLCLIPLoader": "SDXL CLIP Loader", 
    "CombinedDiffusersSD15Loader" : "Combined Diffusers SD15 Loader",
    "CombinedDiffusersSDXLLoader" : "Combined Diffusers SDXL Loader",
    "CombinedDiffusersLoader": "Combined Diffusers Loader",
    "DiffusersClipLoader": "Diffusers CLIP Loader",
    "DiffusersUNETLoader": "Diffusers UNET Loader",
    "DiffusersVAELoader": "Diffusers VAE Loader"
}
