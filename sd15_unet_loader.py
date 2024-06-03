import os
import comfy.sd

def get_sub_directory_list(base_path):
    """Returns a list of sub-directories within the given base path."""
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', base_path))
    print(f"SD15UNETLoader:Looking for sub-directories in: {base_directory}")
    
    if os.path.exists(base_directory):
        sub_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
        print(f"SD15UNETLoader:Found sub-directories: {sub_dirs}")
        return sub_dirs
    else:
        print(f"SD15UNETLoader:Path does not exist: {base_directory}")
    return []

def handle_corrupted_file(unet_folder):
    """Handle corrupted UNET folder by renaming it and logging the action."""
    corrupted_path = unet_folder + ".corrupted"
    try:
        os.rename(unet_folder, corrupted_path)
        print(f"SD15UNETLoader:Moved corrupted folder to: {corrupted_path}")
    except PermissionError as e:
        print(f"SD15UNETLoader:PermissionError while moving corrupted folder: {e}")
        raise PermissionError(f"SD15UNETLoader:Permission denied while moving: {unet_folder}")
    except Exception as e:
        print(f"SD15UNETLoader:Error while moving corrupted folder: {e}")
        raise e

class SD15UNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sub_directory": (get_sub_directory_list("diffusers/SD15"),),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "DiffusersLoader/SD1.5"

    def load_unet(self, sub_directory):
        unet_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers', 'SD15', sub_directory, 'unet'))
        diffusion_model_path = self.find_safetensors_file(unet_folder)
        print(f"SD15UNETLoader:Attempting to load UNET model from: {diffusion_model_path}")
        
        if not os.path.exists(diffusion_model_path):
            raise FileNotFoundError(f"{os.path.basename(diffusion_model_path)} not found in the directory: {diffusion_model_path}")

        try:
            model = comfy.sd.load_unet(diffusion_model_path)
            return (model,)
        except PermissionError as e:
            print(f"SD15UNETLoader:PermissionError: {e}")
            raise PermissionError(f"SD15UNETLoader:Permission denied: {diffusion_model_path}")
        except Exception as e:
            print(f"SD15UNETLoader:Error loading UNET model: {e}")
            handle_corrupted_file(unet_folder)
            raise e

    @staticmethod
    def find_safetensors_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors"):
                return os.path.join(directory, file)
        raise FileNotFoundError(f"SD15UNETLoader:No .safetensors file found in {directory}")

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"SD15UNETLoader": SD15UNETLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"SD15UNETLoader": "SD1.5 UNET Loader"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
