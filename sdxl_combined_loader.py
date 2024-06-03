import os
import comfy.utils
import comfy.sd
import torch

class CombinedDiffusersSDXLLoader:
    @staticmethod
    def get_base_path():
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers', 'SDXL'))
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"CombinedDiffusersSDXLLoader: The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_subdirectories():
        base_path = CombinedDiffusersSDXLLoader.get_base_path()
        return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    @staticmethod
    def find_safetensors_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors"):
                return os.path.join(directory, file)
        raise FileNotFoundError(f"CombinedDiffusersSDXLLoader: No .safetensors file found in {directory}")

    @classmethod
    def INPUT_TYPES(cls):
        subdirectories = cls.get_subdirectories()
        return {
            "required": {
                "sub_directory": (subdirectories, ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_models"
    CATEGORY = "DiffusersLoader/SDXL"

    def load_models(self, sub_directory):
        base_path = self.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)

        vae_model = self.load_vae(sub_dir_path)
        unet_model = self.load_unet(sub_dir_path)
        clip_model = self.load_clip(sub_dir_path)

        return unet_model, clip_model, vae_model

    def load_vae(self, base_path):
        vae_folder = os.path.join(base_path, "vae")
        vae_path = self.find_safetensors_file(vae_folder)
        print(f"CombinedDiffusersSDXLLoader: Attempting to load VAE model from: {vae_path}")

        vae_sd = comfy.utils.load_torch_file(vae_path)
        return comfy.sd.VAE(sd=vae_sd)

    def load_unet(self, base_path):
        unet_folder = os.path.join(base_path, 'unet')
        unet_path = self.find_safetensors_file(unet_folder)
        print(f"CombinedDiffusersSDXLLoader: Attempting to load UNET model from: {unet_path}")

        try:
            return comfy.sd.load_unet(unet_path)
        except PermissionError as e:
            print(f"PermissionError: {e}")
            raise PermissionError(f"CombinedDiffusersSDXLLoader: Permission denied: {unet_path}")
        except Exception as e:
            print(f"CombinedDiffusersSDXLLoader: Error loading UNET model: {e}")
            self.handle_corrupted_file(unet_folder)
            raise e

    def load_clip(self, base_path):
        text_encoder_dir1 = os.path.join(base_path, "text_encoder")
        text_encoder_dir2 = os.path.join(base_path, "text_encoder_2")

        text_encoder_path1 = self.find_safetensors_file(text_encoder_dir1)
        text_encoder_path2 = self.find_safetensors_file(text_encoder_dir2)

        print(f"CombinedDiffusersSDXLLoader: Checking paths:\n{text_encoder_path1}\n{text_encoder_path2}")

        return comfy.sd.load_clip(ckpt_paths=[text_encoder_path1, text_encoder_path2], embedding_directory=os.path.join(base_path, "embeddings"))

    @staticmethod
    def handle_corrupted_file(unet_folder):
        """Handle corrupted UNET folder by renaming it and logging the action."""
        corrupted_path = unet_folder + ".corrupted"
        try:
            os.rename(unet_folder, corrupted_path)
            print(f"CombinedDiffusersSDXLLoader: Moved corrupted folder to: {corrupted_path}")
        except PermissionError as e:
            print(f"CombinedDiffusersSDXLLoader: PermissionError while moving corrupted folder: {e}")
            raise PermissionError(f"Permission denied while moving: {unet_folder}")
        except Exception as e:
            print(f"CombinedDiffusersSDXLLoader: Error while moving corrupted folder: {e}")
            raise e

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"CombinedDiffusersSDXLLoader": CombinedDiffusersSDXLLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CombinedDiffusersSDXLLoader": "Combined Diffusers SDXL Loader"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
