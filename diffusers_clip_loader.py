import os
from comfy import sd
import comfy.utils
import torch

class DiffusersClipLoader:
    @staticmethod
    def get_base_path():
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers'))
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"DiffusersClipLoader: The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_model_directories(base_path):
        model_dirs = []
        for root, dirs, files in os.walk(base_path):
            if "model_index.json" in files:
                model_dirs.append(os.path.relpath(root, base_path))
            # Remove unwanted subdirectories from dirs so that os.walk doesn't traverse them
            dirs[:] = [d for d in dirs if d not in ["vae", "unet", "text_encoder", "text_encoder_2"]]
        return model_dirs

    @staticmethod
    def find_model_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                return file
        raise FileNotFoundError(f"DiffusersClipLoader: No .safetensors file or .bin file found in {directory}")

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

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "DiffusersLoader"

    def load_clip(self, sub_directory, clip_type="stable_diffusion"):
        base_path = self.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)

        model_type = self.detect_model_type(sub_dir_path)

        clip_model = self.load_clip_model(sub_dir_path, clip_type, model_type)

        return (clip_model,)

    def detect_model_type(self, sub_dir_path):
        text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
        text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")

        if os.path.exists(text_encoder_dir1) and os.path.exists(text_encoder_dir2):
            return "SDXL"
        elif os.path.exists(text_encoder_dir1):
            return "SD15"
        else:
            raise FileNotFoundError("No valid text_encoder directories found. This model is not SD15 or SDXL")

    def load_clip_model(self, sub_dir_path, clip_type="stable_diffusion", model_type="SD15"):
        clip_type_enum = sd.CLIPType.STABLE_DIFFUSION
        if clip_type == "stable_cascade":
            clip_type_enum = sd.CLIPType.STABLE_CASCADE

        text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
        text_encoder_paths = [os.path.join(text_encoder_dir1, self.find_model_file(text_encoder_dir1))]

        if model_type == "SDXL":
            text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")
            text_encoder_paths.append(os.path.join(text_encoder_dir2, self.find_model_file(text_encoder_dir2)))

        print(f"DiffusersClipLoader: Checking paths: \n{text_encoder_paths}")

        clip_model = sd.load_clip(ckpt_paths=text_encoder_paths, embedding_directory=os.path.join(sub_dir_path, "embeddings"), clip_type=clip_type_enum)

        # Debugging statement to check if the loaded model has 'tokenize' method
        if not hasattr(clip_model, 'tokenize'):
            raise AttributeError("Loaded clip model does not have 'tokenize' method.")

        return clip_model

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"DiffusersClipLoader": DiffusersClipLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"DiffusersClipLoader": "Diffusers CLIP Loader"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
