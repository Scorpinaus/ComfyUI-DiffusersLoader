import os
import comfy.sd

def get_sub_directory_list(base_path):
    """Returns a list of sub-directories within the given base path."""
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', base_path))
    print(f"SDXLUNETLoader:Looking for sub-directories in: {base_directory}")
    
    if os.path.exists(base_directory):
        sub_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
        print(f"SDXLUNETLoader:Found sub-directories: {sub_dirs}")
        return sub_dirs
    else:
        print(f"SDXLUNETLoader:Path does not exist: {base_directory}")
    return []

def handle_corrupted_file(unet_folder):
    """Handle corrupted UNET folder by renaming it and logging the action."""
    corrupted_path = unet_folder + ".corrupted"
    try:
        os.rename(unet_folder, corrupted_path)
        print(f"SDXLUNETLoader:Moved corrupted folder to: {corrupted_path}")
    except PermissionError as e:
        print(f"SDXLUNETLoader:PermissionError while moving corrupted folder: {e}")
        raise PermissionError(f"Permission denied while moving: {unet_folder}")
    except Exception as e:
        print(f"SDXLUNETLoader:Error while moving corrupted folder: {e}")
        raise e

class SDXLUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sub_directory": (get_sub_directory_list("diffusers/SDXL"),),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "testing_nodes"

    def load_unet(self, sub_directory):
        unet_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers', 'SDXL', sub_directory, 'unet'))
        diffusion_model_path = self.find_safetensors_file(unet_folder)
        print(f"SDXLUNETLoader:Attempting to load UNET model from: {diffusion_model_path}")
        
        if not os.path.exists(diffusion_model_path):
            raise FileNotFoundError(f"SDXLUNETLoader:{os.path.basename(diffusion_model_path)} not found in the directory: {diffusion_model_path}")

        try:
            model = comfy.sd.load_unet(diffusion_model_path)
            return (model,)
        except PermissionError as e:
            print(f"PermissionError: {e}")
            raise PermissionError(f"SDXLUNETLoader:Permission denied: {diffusion_model_path}")
        except Exception as e:
            print(f"SDXLUNETLoader:Error loading UNET model: {e}")
            handle_corrupted_file(unet_folder)
            raise e

    @staticmethod
    def find_safetensors_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors"):
                return os.path.join(directory, file)
        raise FileNotFoundError(f"SDXLUNETLoader:No .safetensors file found in {directory}")

# Mapping the node to make it recognizable in the ComfyUI framework
NODE_CLASS_MAPPINGS = {
    "SDXLUNETLoader": SDXLUNETLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLUNETLoader": "SDXL UNET Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
