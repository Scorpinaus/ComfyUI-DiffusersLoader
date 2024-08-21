# vae_loader.py
import os
import torch
import comfy.sd
import comfy.utils
from .base_loader import DiffusersLoaderBase
from .utils import DiffusersUtils

class DiffusersVAELoader(DiffusersLoaderBase):
    # Moving the VAE_Configs to a Json file might be for a future refactor!
    VAE_CONFIGS = {
        "taesd": {"scale": 0.18215, "shift": 0.0},
        "taesdxl": {"scale": 0.13025, "shift": 0.0},
        "taesd3": {"scale": 1.5305, "shift": 0.0609},
        "taef1": {"scale": 0.3611, "shift": 0.1159},
    }
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sub_directory": (DiffusersUtils.get_model_directories(DiffusersUtils.get_base_path()),),
                "vae_type": (["default"] + list(cls.VAE_CONFIGS.keys()),),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "DiffusersLoader"

    @classmethod
    def load_vae(cls, sub_directory, vae_type="default"):
        return (cls.load_model(sub_directory, vae_type),)

    @classmethod
    def load_model(cls, sub_directory, vae_type="default"):
        if vae_type == "default":
            return cls.load_default_vae(sub_directory)
        else:
            return cls.load_taesd(vae_type)

    @classmethod
    def load_default_vae(cls, sub_directory):       
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
        
    @staticmethod
    def load_taesd(vae_type):
        sd = {}
        vae_approx_path = os.path.join(os.path.dirname(DiffusersUtils.get_base_path()), "vae_approx")
        # Validation check for vae_approx_path
        if not os.path.exists(vae_approx_path):
            raise FileNotFoundError(f"The vae_approx path '{vae_approx_path}' does not exist.")
        
        encoder_file = next(f for f in os.listdir(vae_approx_path) if f.startswith(f"{vae_type}_encoder."))
        decoder_file = next(f for f in os.listdir(vae_approx_path) if f.startswith(f"{vae_type}_decoder."))
        
        #Validation check for presence of encoder and decoder_file
        if encoder_file is None or decoder_file is None:
            raise FileNotFoundError(f"No encoder or decoder file found in {vae_approx_path}.")
        
        enc = comfy.utils.load_torch_file(os.path.join(vae_approx_path, encoder_file))
        for k in enc:
            sd[f"taesd_encoder.{k}"] = enc[k]
        
        dec = comfy.utils.load_torch_file(os.path.join(vae_approx_path, decoder_file))
        for k in dec:
            sd[f"taesd_decoder.{k}"] = dec[k]
        
        if vae_type in DiffusersVAELoader.VAE_CONFIGS:
            config = DiffusersVAELoader.VAE_CONFIGS[vae_type]
            sd["vae_scale"] = torch.tensor(config["scale"])
            sd["vae_shift"] = torch.tensor(config["shift"])
        else:
            print(f"Warning: No configuration found for {vae_type}. Using default values.")
            sd["vae_scale"] = torch.tensor(1.0)
            sd["vae_shift"] = torch.tensor(0.0)
        
        return comfy.sd.VAE(sd=sd)