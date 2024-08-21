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
                "clip_type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "sdxl", "flux"],),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "DiffusersLoader"

    @classmethod
    def load_clip(cls, sub_directory, clip_type="stable_diffusion"):
        return (cls.load_model(sub_directory, clip_type,),)

    @classmethod
    def load_model(cls, sub_directory, clip_type="stable_diffusion"):
        
        base_path = DiffusersUtils.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        model_type = cls.detect_model_type(sub_dir_path)

        clip_type_enum = cls.get_clip_type_enum(clip_type)
        
        text_encoder_paths = cls.get_text_encoder_paths(sub_dir_path, model_type)
        
        for path in text_encoder_paths:
            DiffusersUtils.check_and_clear_cache('clip', path)
            
        print(f"DiffusersClipLoader: Loading CLIP model(s) from: {text_encoder_paths}")
        
        try:
            clip_model = comfy.sd.load_clip(ckpt_paths=text_encoder_paths, 
            embedding_directory=os.path.join(sub_dir_path, "embeddings"), clip_type=clip_type_enum)
        except Exception as e:
            print(f"DiffusersClipLoader: Error loading clip model: {e}")
            raise

        if not hasattr(clip_model, 'tokenize'):
            raise AttributeError("DiffusersClipLoader: Loaded clip model does not have 'tokenize' method.")

        return clip_model
    
    @staticmethod 
    def get_clip_type_enum(clip_type):
        clip_type_map = {
            "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,
            "stable_cascade": comfy.sd.CLIPType.STABLE_CASCADE,
            "sd3": comfy.sd.CLIPType.SD3,
            "stable_audio": comfy.sd.CLIPType.STABLE_AUDIO,
            "sdxl": comfy.sd.CLIPType.STABLE_DIFFUSION,
            "flux": comfy.sd.CLIPType.FLUX
        }
        
        return clip_type_map.get(clip_type, comfy.sd.CLIPType.STABLE_DIFFUSION)
    
    @classmethod
    def get_text_encoder_paths(cls, sub_dir_path, model_type):
        text_encoder_paths = []
        
        if model_type in ["SDXL", "SD3", "Flux"]:
            text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
            text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")
            
            text_encoder_paths.append(DiffusersUtils.find_model_file(text_encoder_dir1))
            
            if model_type == "Flux":
                # Combine split files for Flux models
                combined_file_path = os.path.join(text_encoder_dir2, "combined_text_encoder.safetensors")
                if not os.path.exists(combined_file_path):
                    combined_file_path = DiffusersUtils.combine_safetensor_files(text_encoder_dir2, sub_dir_path, num_parts=2)
                text_encoder_paths.append(combined_file_path)
            else:
                text_encoder_paths.append(DiffusersUtils.find_model_file(text_encoder_dir2))

            
            if model_type == "SD3":
                text_encoder_dir3 = os.path.join(sub_dir_path, "text_encoder_3")
                text_encoder_paths.append(DiffusersUtils.find_model_file(text_encoder_dir3))
        
        else: #For SD15 or other single text encoder models
            text_encoder_dir = os.path.join(sub_dir_path, "text_encoder")
            text_encoder_paths = [DiffusersUtils.find_model_file(text_encoder_dir)]
        
        return text_encoder_paths