import os
import comfy.utils
import torch

class SDXLVAELoader:
    @staticmethod
    def get_base_path():
        # Adjust the base path to the location where the SDXL subdirectory is located
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers', 'SDXL')
        base_path = os.path.abspath(base_path)  # Convert to absolute path
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"SDXLVAELoader:The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_subdirectories():
        base_path = SDXLVAELoader.get_base_path()
        return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    @staticmethod
    def find_safetensors_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors"):
                return file
        raise FileNotFoundError(f"SDXLVAELoader:No .safetensors file found in {directory}")

    @classmethod
    def INPUT_TYPES(cls):
        subdirectories = cls.get_subdirectories()
        return {
            "required": {
                "sub_directory": (subdirectories, ),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "DiffusersLoader/SDXL"

    def load_vae(self, sub_directory):
        base_path = self.get_base_path()
        vae_folder = os.path.join(base_path, sub_directory, "vae")
        vae_filename = self.find_safetensors_file(vae_folder)
        vae_path = os.path.join(vae_folder, vae_filename)
        full_path = os.path.abspath(vae_path)

        print(f"SDXLVAELoader:Attempting to load VAE model from: {full_path}")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"SDXLVAELoader:The VAE file '{full_path}' does not exist.")

        sd = comfy.utils.load_torch_file(full_path)
        vae = comfy.sd.VAE(sd=sd)
        return (vae,)

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"SDXLVAELoader": SDXLVAELoader}
NODE_DISPLAY_NAME_MAPPINGS = {"SDXLVAELoader": "SDXL VAE Loader"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
