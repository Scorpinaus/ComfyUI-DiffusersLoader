import os
import comfy.sd

def get_model_directories(base_path):
    """Returns a list of sub-directories containing models within the given base path."""
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', base_path))
    print(f"DiffusersUNETLoader: Looking for model directories in: {base_directory}")
    
    model_dirs = []
    if os.path.exists(base_directory):
        for root, dirs, files in os.walk(base_directory):
            if "model_index.json" in files:
                model_dirs.append(os.path.relpath(root, base_directory))
                print(f"DiffusersUNETLoader: Found model directory: {os.path.relpath(root, base_directory)}")
            dirs[:] = [d for d in dirs if d not in ["vae", "unet", "text_encoder", "text_encoder_2"]]
    else:
        print(f"DiffusersUNETLoader: Path does not exist: {base_directory}")
    return model_dirs

def handle_corrupted_file(unet_folder):
    """Handle corrupted UNET folder by renaming it and logging the action."""
    corrupted_path = unet_folder + ".corrupted"
    try:
        os.rename(unet_folder, corrupted_path)
        print(f"DiffusersUNETLoader: Moved corrupted folder to: {corrupted_path}")
    except PermissionError as e:
        print(f"DiffusersUNETLoader: PermissionError while moving corrupted folder: {e}")
        raise PermissionError(f"DiffusersUNETLoader: Permission denied while moving: {unet_folder}")
    except Exception as e:
        print(f"DiffusersUNETLoader: Error while moving corrupted folder: {e}")
        raise e

class DiffusersUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        base_path = "diffusers"
        model_directories = get_model_directories(base_path)
        return {
            "required": {
                "sub_directory": (model_directories,),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "DiffusersLoader"

    def load_unet(self, sub_directory):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers'))
        sub_dir_path = os.path.join(base_path, sub_directory)
        
        model_type = self.detect_model_type(sub_dir_path)

        if model_type == "SD15":
            unet_folder = os.path.join(sub_dir_path, 'unet')
        elif model_type == "SDXL":
            unet_folder = os.path.join(sub_dir_path, 'unet')
        else:
            raise ValueError("DiffusersUNETLoader: Unsupported model type")

        diffusion_model_path = self.find_model_file(unet_folder)
        print(f"DiffusersUNETLoader: Attempting to load UNET model from: {diffusion_model_path}")

        if not os.path.exists(diffusion_model_path):
            raise FileNotFoundError(f"DiffusersUNETLoader: {os.path.basename(diffusion_model_path)} not found in the directory: {diffusion_model_path}")

        try:
            model = comfy.sd.load_unet(diffusion_model_path)
            return (model,)
        except PermissionError as e:
            print(f"DiffusersUNETLoader: PermissionError: {e}")
            handle_corrupted_file(unet_folder)
            raise e
        except Exception as e:
            print(f"DiffusersUNETLoader: Error loading UNET model: {e}")
            handle_corrupted_file(unet_folder)
            raise e

    def detect_model_type(self, sub_dir_path):
        """Detect the model type based on the directory structure."""
        text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
        text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")

        if os.path.exists(text_encoder_dir1) and os.path.exists(text_encoder_dir2):
            return "SDXL"
        elif os.path.exists(text_encoder_dir1):
            return "SD15"
        else:
            raise FileNotFoundError("DiffusersUNETLoader: No valid text_encoder directories found. This model is not SD15 or SDXL")

    @staticmethod
    def find_model_file(directory):
        """Find the .safetensors or .bin file in the given directory."""
        for file in os.listdir(directory):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                return os.path.join(directory, file)
        raise FileNotFoundError(f"DiffusersUNETLoader: No .safetensors file or .bin file found in {directory}")

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"DiffusersUNETLoader": DiffusersUNETLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"DiffusersUNETLoader": "Diffusers UNET Loader"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
