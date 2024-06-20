import os
import comfy.utils
import torch

class SD15VAELoader:
    @staticmethod
    def get_base_path():
        # Adjust the base path to the location where the SD15 subdirectory is located
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers', 'SD15')
        base_path = os.path.abspath(base_path)  # Convert to absolute path
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_subdirectories():
        base_path = SD15VAELoader.get_base_path()
        return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    @staticmethod
    def find_model_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                return file
        raise FileNotFoundError(f"No .safetensors file or .bin file found in {directory}")

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
    CATEGORY = "DiffusersLoader/SD1.5"

    def load_vae(self, sub_directory):
        base_path = self.get_base_path()
        vae_folder = os.path.join(base_path, sub_directory, "vae")
        vae_filename = self.find_model_file(vae_folder)
        vae_path = os.path.join(vae_folder, vae_filename)
        full_path = os.path.abspath(vae_path)

        print(f"Attempting to load VAE model from: {full_path}")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The VAE file '{full_path}' does not exist.")

        sd = comfy.utils.load_torch_file(full_path)
        vae = comfy.sd.VAE(sd=sd)
        return (vae,)

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"SD15VAELoader": SD15VAELoader}
NODE_DISPLAY_NAME_MAPPINGS = {"SD15VAELoader": "SD1.5 VAE Loader"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
