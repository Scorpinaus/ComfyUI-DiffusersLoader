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
        base_path = DiffusersUtils.get_base_path()
        model_directories = DiffusersUtils.get_model_directories(base_path)
        return {
            "required": {
                "sub_directory": (model_directories,),
                "clip_type": (["stable_diffusion", "stable_cascade"],),
                "clip_parts": (["all", "part_1", "part_2"],),
                "transformer_parts": (["all", "part_1", "part_2", "part_3"],),
                "vae_type": (["default", "taesd", "taesdxl", "taesd3", "taef1"],),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],)
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_models"
    CATEGORY = "DiffusersLoader/Combined"

    @classmethod
    def load_models(cls, sub_directory, clip_type="stable_diffusion", clip_parts="all", transformer_parts="all", vae_type="default", weight_dtype="default"):
        
        base_path = DiffusersUtils.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        model_type = DiffusersLoaderBase.detect_model_type(sub_dir_path)
        print(f"Detected model type: {model_type}")

        unet_model = DiffusersUNETLoader.load_model(sub_directory, transformer_parts, weight_dtype)
        
        if model_type == "Flux":
            clip_model = DiffusersClipLoader.load_model(sub_directory, "stable_diffusion", "all")
        else:
            clip_model = DiffusersClipLoader.load_model(sub_directory, clip_type, clip_parts)
        
        vae_model = DiffusersVAELoader.load_model(sub_directory, vae_type)

        return unet_model, clip_model, vae_model

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"CombinedDiffusersLoader": CombinedDiffusersLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CombinedDiffusersLoader": "Combined Diffusers Loader"}