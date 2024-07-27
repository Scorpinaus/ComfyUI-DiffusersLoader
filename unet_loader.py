# unet_loader.py
import os
import comfy.sd
from .base_loader import DiffusersLoaderBase
from .utils import DiffusersUtils

class DiffusersUNETLoader(DiffusersLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sub_directory": (DiffusersUtils.get_model_directories(DiffusersUtils.get_base_path()),),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "DiffusersLoader"

    @classmethod
    def load_unet(cls, sub_directory):
        return (cls.load_model(sub_directory),)

    @classmethod
    def load_model(cls, sub_directory):
        base_path = DiffusersUtils.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        model_type = cls.detect_model_type(sub_dir_path)

        if model_type == "SD3":
            unet_folder = os.path.join(sub_dir_path, 'transformer')
        else:
            unet_folder = os.path.join(sub_dir_path, 'unet')

        if not os.path.exists(unet_folder):
            raise FileNotFoundError(f"The directory '{unet_folder}' does not exist for model type '{model_type}'")

        unet_path = DiffusersUtils.find_model_file(unet_folder)
        print(f"DiffusersUNETLoader: Attempting to load UNET model from {unet_path}")

        try:
            return comfy.sd.load_unet(unet_path)
        except PermissionError as e:
            print(f"DiffusersUNETLoader: PermissionError: {e}")
            raise PermissionError(f"Permission denied: {unet_path}")
        except FileNotFoundError as e:
            print(f"DiffusersUNETLoader: FileNotFoundError: {e}")
            raise FileNotFoundError(f"UNET file not found: {unet_path}")
        except Exception as e:
            print(f"DiffusersUNETLoader: Error loading UNET model: {e}")
            raise e