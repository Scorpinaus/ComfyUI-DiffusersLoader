import os
import comfy.utils
import comfy.sd
import torch

class DiffusersVAELoader:
    @staticmethod
    def get_base_path():
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers')
        base_path = os.path.abspath(base_path)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"DiffusersVAELoader: The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_model_directories(base_path):
        model_dirs = []
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                if "model_index.json" in files:
                    model_dirs.append(os.path.relpath(root, base_path))
                    print(f"DiffusersVAELoader: Found model directory: {os.path.relpath(root, base_path)}")
                dirs[:] = [d for d in dirs if d not in ["vae", "unet", "text_encoder", "text_encoder_2"]]
        else:
            print(f"DiffusersVAELoader: Path does not exist: {base_path}")
        return model_dirs

    @staticmethod
    def find_model_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                return file
        raise FileNotFoundError(f"DiffusersVAELoader: No .safetensors file or .bin file found in {directory}")

    @classmethod
    def INPUT_TYPES(cls):
        base_path = cls.get_base_path()
        model_directories = cls.get_model_directories(base_path)
        return {
            "required": {
                "sub_directory": (model_directories,),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "DiffusersLoader"

    def load_vae(self, sub_directory):
        base_path = self.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)

        model_type = self.detect_model_type(sub_dir_path)

        vae_folder = os.path.join(sub_dir_path, "vae")
        vae_filename = self.find_model_file(vae_folder)
        vae_path = os.path.join(vae_folder, vae_filename)
        full_path = os.path.abspath(vae_path)

        print(f"DiffusersVAELoader: Attempting to load VAE model from: {full_path}")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"DiffusersVAELoader: The VAE file '{full_path}' does not exist.")

        sd = comfy.utils.load_torch_file(full_path)
        vae = comfy.sd.VAE(sd=sd)
        return (vae,)

    def detect_model_type(self, sub_dir_path):
        """Detect the model type based on the directory structure."""
        text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
        text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")

        if os.path.exists(text_encoder_dir1) and os.path.exists(text_encoder_dir2):
            return "SDXL"
        elif os.path.exists(text_encoder_dir1):
            return "SD15"
        else:
            raise FileNotFoundError("DiffusersVAELoader: No valid text_encoder directories found. This model is not SD15 or SDXL")

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"DiffusersVAELoader": DiffusersVAELoader}
NODE_DISPLAY_NAME_MAPPINGS = {"DiffusersVAELoader": "Diffusers VAE Loader"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
