import os
import comfy.sd
import comfy.utils
import torch

class CombinedDiffusersLoader:
    @staticmethod
    def get_base_path():
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers'))
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"CombinedDiffusersLoader: The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_model_directories(base_path):
        model_dirs = []
        for root, dirs, files in os.walk(base_path):
            if "model_index.json" in files:
                model_dirs.append(os.path.relpath(root, base_path))
            #Remove unwanted sub-directories from dirs
            dirs[:] = [d for d in dirs if d not in ["vae", "unet", "text_encoder", "text_encoder_2"]] 
        
        return model_dirs

    @staticmethod
    def find_model_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                return os.path.join(directory, file)
        raise FileNotFoundError(f"CombinedDiffusersLoader: No .safetensors file or .bin file found in {directory}")

    @classmethod
    def INPUT_TYPES(cls):
        base_path = cls.get_base_path()
        model_directories = cls.get_model_directories(base_path)
        return {
            "required": {
                "sub_directory": (model_directories,),
                "clip_type": (["stable_diffusion", "stable_cascade"],)
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_models"
    CATEGORY = "DiffusersLoader/Combined"

    def load_models(self, sub_directory, clip_type="stable_diffusion"):
        base_path = self.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        
        model_type = self.detect_model_type(sub_dir_path)
        
        vae_model = self.load_vae(sub_dir_path)
        unet_model = self.load_unet(sub_dir_path)
        clip_model = self.load_clip(sub_dir_path, clip_type, model_type)

        return unet_model, clip_model, vae_model

    def detect_model_type(self, sub_dir_path):
        text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
        text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")

        if os.path.exists(text_encoder_dir1) and os.path.exists(text_encoder_dir2):
            return "SDXL"
        elif os.path.exists(text_encoder_dir1):
            return "SD15"
        else:
            print(f"CombinedDiffusersLoader: Sub_dir_path: \n{sub_dir_path}")
            raise FileNotFoundError("No valid text_encoder directories found. This model is not SD15 or SDXL")

    def load_clip(self, base_path, clip_type="stable_diffusion", model_type="SD15"):
        clip_type_enum = comfy.sd.CLIPType.STABLE_DIFFUSION
        if clip_type == "stable_cascade":
            clip_type_enum = comfy.sd.CLIPType.STABLE_CASCADE

        text_encoder_dir1 = os.path.join(base_path, "text_encoder")
        text_encoder_paths = [self.find_model_file(text_encoder_dir1)]

        if model_type == "SDXL":
            text_encoder_dir2 = os.path.join(base_path, "text_encoder_2")
            text_encoder_paths.append(self.find_model_file(text_encoder_dir2))

        print(f"CombinedDiffusersLoader: Checking paths: \n{text_encoder_paths}")
        
        clip_model = comfy.sd.load_clip(ckpt_paths=text_encoder_paths, embedding_directory=os.path.join(base_path, "embeddings"), clip_type=clip_type_enum)

        # Debugging statement to check if the loaded model has 'tokenize' method
        if not hasattr(clip_model, 'tokenize'):
            raise AttributeError("Loaded clip model does not have 'tokenize' method.")

        return clip_model
    
    def load_unet(self, base_path):
        unet_folder = os.path.join(base_path, 'unet')
        unet_path = self.find_model_file(unet_folder)
        print(f"CombinedDiffusersLoader: Attempting to load UNET model from {unet_path}")

        try:
            return comfy.sd.load_unet(unet_path)
        except PermissionError as e:
            print(f"Permissionerror:{e}")
            raise PermissionError(f"Permission denied: {unet_path}")
        except Exception as e:
            print(f"Error loading UNET model:{e}")
            raise e

    def load_vae(self, base_path):
        vae_folder = os.path.join(base_path, "vae")
        vae_path = self.find_model_file(vae_folder)
        print(f"Attempting to load VAE model from: {vae_path}")

        vae_sd = comfy.utils.load_torch_file(vae_path)
        return comfy.sd.VAE(sd=vae_sd)

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"CombinedDiffusersLoader": CombinedDiffusersLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CombinedDiffusersLoader": "Combined Diffusers Loader"}
