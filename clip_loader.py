# clip_loader.py
import os
import comfy.sd
from .base_loader import DiffusersLoaderBase
from .utils import DiffusersUtils

class DiffusersClipLoader(DiffusersLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sub_directory": (DiffusersUtils.get_model_directories(DiffusersUtils.get_base_path()),),
                "clip_type": (["stable_diffusion", "stable_cascade"],),
                "file_parts": (["none", "all", "part_1", "part_2"],)
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "DiffusersLoader"

    @classmethod
    def load_clip(cls, sub_directory, clip_type="stable_diffusion", file_parts="all"):
        return (cls.load_model(sub_directory, clip_type, file_parts),)

    @classmethod
    def load_model(cls, sub_directory, clip_type="stable_diffusion", file_parts="all"):
        base_path = DiffusersUtils.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        model_type = cls.detect_model_type(sub_dir_path)

        clip_type_enum = comfy.sd.CLIPType.STABLE_DIFFUSION
        if clip_type == "stable_cascade":
            clip_type_enum = comfy.sd.CLIPType.STABLE_CASCADE

        text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
        text_encoder_paths = [DiffusersUtils.find_model_file(text_encoder_dir1)]

        if model_type in ["SDXL", "SD3"]:
            text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")
            text_encoder_paths.append(DiffusersUtils.find_model_file(text_encoder_dir2))
            
        if model_type == "SD3":
            text_encoder_dir3 = os.path.join(sub_dir_path, "text_encoder_3")
            if file_parts == "all":
                combined_file_path = os.path.join(text_encoder_dir3, "combined_text_encoder.safetensors")
                if not os.path.exists(combined_file_path):
                    combined_file_path = DiffusersUtils.combine_safetensor_files(text_encoder_dir3, base_path)
                text_encoder_paths.append(combined_file_path)
            elif file_parts == "part_1":
                text_encoder_paths += DiffusersUtils.find_model_files(text_encoder_dir3, ["00001-of-00002"])
            elif file_parts == "part_2":
                text_encoder_paths += DiffusersUtils.find_model_files(text_encoder_dir3, ["00002-of-00002"])
            
        print(f"DiffusersClipLoader: Checking paths: \n{text_encoder_paths}")
        
        try:
            clip_model = comfy.sd.load_clip(ckpt_paths=text_encoder_paths, embedding_directory=os.path.join(sub_dir_path, "embeddings"), clip_type=clip_type_enum)
        except Exception as e:
            print(f"DiffusersClipLoader: Error loading clip model: {e}")
            raise

        if not hasattr(clip_model, 'tokenize'):
            raise AttributeError("DiffusersClipLoader: Loaded clip model does not have 'tokenize' method.")

        return clip_model