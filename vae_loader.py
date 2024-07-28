# vae_loader.py
import os
import comfy.sd
import comfy.utils
from .base_loader import DiffusersLoaderBase
from .utils import DiffusersUtils

class DiffusersVAELoader(DiffusersLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sub_directory": (DiffusersUtils.get_model_directories(DiffusersUtils.get_base_path()),),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "DiffusersLoader"

    @classmethod
    def load_vae(cls, sub_directory):
        return (cls.load_model(sub_directory),)

    @classmethod
    def load_model(cls, sub_directory):
        
        base_path = DiffusersUtils.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        
        vae_folder = os.path.join(sub_dir_path, "vae")
        vae_path = DiffusersUtils.find_model_file(vae_folder)
        
        DiffusersUtils.check_and_clear_cache('vae', vae_path)
        
        print(f"DiffusersVAELoader: Attempting to load VAE model from: {vae_path}")

        try:
            vae_sd = comfy.utils.load_torch_file(vae_path)
            return comfy.sd.VAE(sd=vae_sd)
        except Exception as e:
            print(f"DiffusersVAELoader: Error loading VAE model: {e}")
            raise