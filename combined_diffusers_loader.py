# combined_diffusers_loader.py
from .utils import DiffusersUtils
from .unet_loader import DiffusersUNETLoader
from .clip_loader import DiffusersClipLoader
from .vae_loader import DiffusersVAELoader
from .base_loader import DiffusersLoaderBase
import os

class CombinedDiffusersLoader:
    @classmethod
    def INPUT_TYPES(cls):
        display_names, _ = DiffusersUtils.get_unique_display_names(DiffusersUtils.get_model_directories())
        
        return {
            "required": {
                "sub_directory": (display_names,),
                "clip_type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "sdxl", "flux"],),
                "transformer_parts": (["all", "part_1", "part_2", "part_3"],),
                "vae_type": (["default", "taesd", "taesdxl", "taesd3", "taef1"],),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],)
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_models"
    CATEGORY = "DiffusersLoader/Combined"

    @classmethod
    def load_models(cls, sub_directory, clip_type="stable_diffusion", transformer_parts="all", vae_type="default", weight_dtype="default"):
        
        model_directories = DiffusersUtils.get_model_directories()
        _, unique_names = DiffusersUtils.get_unique_display_names(model_directories)
        
        if "(" in sub_directory:
            dir_name, index = sub_directory.rsplit(" (", 1)
            index = int(index[:-1]) - 1
            full_path = unique_names[dir_name][index]
        else:
            full_path = unique_names[sub_directory][0]
        
        if not os.path.exists(full_path):
            raise ValueError(f"Model directory '{full_path}' not found.")        

        unet_model, = DiffusersUNETLoader.load_model(full_path, transformer_parts, weight_dtype)

        clip_model = DiffusersClipLoader.load_model(full_path, clip_type)
        
        vae_model = DiffusersVAELoader.load_model(full_path, vae_type)

        return unet_model, clip_model, vae_model

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"CombinedDiffusersLoader": CombinedDiffusersLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CombinedDiffusersLoader": "Combined Diffusers Loader"}