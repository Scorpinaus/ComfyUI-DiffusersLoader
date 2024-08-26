# clip_loader.py
import os
import json
import torch
import comfy.sd
import comfy.utils
from .base_loader import DiffusersLoaderBase
from .utils import DiffusersUtils

class DiffusersClipLoader(DiffusersLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sub_directory": (DiffusersUtils.get_unique_display_names(DiffusersUtils.get_model_directories())[0],),
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
        if os.path.exists(sub_directory):
            full_path = sub_directory
        else:
            model_directories = DiffusersUtils.get_model_directories()
            _, unique_names = DiffusersUtils.get_unique_display_names(model_directories)
        
            if "(" in sub_directory:
                dir_name, index = sub_directory.rsplit(" (", 1)
                index = int(index[:-1]) - 1
                full_path = unique_names[dir_name][index]
            else:
                full_path = unique_names[sub_directory][0]
        
        if not os.path.exists(full_path):
            raise ValueError(f"Selected directory does not exist: {full_path}")
        
        model_type = cls.detect_model_type(full_path)

        clip_type_enum = cls.get_clip_type_enum(clip_type)
        
        text_encoder_paths = cls.get_text_encoder_paths(full_path, model_type)
        
        clip_data = []
        for path in text_encoder_paths:
            try:
                if isinstance(path, dict):
                    clip_data.append(path)
                elif path.endswith('index.json'):
                    with open(path, 'r') as f:
                        index_data = json.load(f)
                    weight_map = index_data['weight_map']
                    state_dict = {}
                    
                    for key, file in weight_map.items():
                        file_path = os.path.join(os.path.dirname(path), file)
                        part_dict = comfy.utils.load_torch_file(file_path, safe_load=True)
                        state_dict.update(part_dict)
                        del part_dict
                        torch.cuda.empty_cache()
                    
                    clip_data.append(state_dict)
                    
                else:
                    DiffusersUtils.check_and_clear_cache('clip', path)
                    clip_data.append(comfy.utils.load_torch_file(path, safe_load=True))
            except Exception as e:
                print(f"Error loading clip model part from {path}: {e}")
                raise
            
        print(f"DiffusersClipLoader: Loading CLIP model(s) from: {text_encoder_paths}")
        
        try:
            clip_model = comfy.sd.load_text_encoder_state_dicts(clip_data, 
            embedding_directory=os.path.join(full_path, "embeddings"), clip_type=clip_type_enum)
        except Exception as e:
            print(f"DiffusersClipLoader: Error loading clip model: {e}")
            raise

        if not hasattr(clip_model, 'tokenize'):
            raise AttributeError("DiffusersClipLoader: Loaded clip model does not have 'tokenize' method.")
        
        del clip_data
        torch.cuda.empty_cache()

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
    def get_text_encoder_paths(cls, full_path, model_type):
        text_encoder_paths = []
        
        if model_type in ["SDXL", "SD3", "Flux"]:
            text_encoder_dir1 = os.path.join(full_path, "text_encoder")
            text_encoder_dir2 = os.path.join(full_path, "text_encoder_2")
            
            text_encoder_paths.append(DiffusersUtils.find_model_file(text_encoder_dir1))
            
            if model_type == "Flux":
                # use the index file
                index_files = [f for f in os.listdir(text_encoder_dir2) if f.endswith('index.json')]
                if index_files:
                    text_encoder_paths.append(os.path.join(text_encoder_dir2, index_files[0]))
                else:
                    print(f"No index file found in {text_encoder_dir2}. Checking for combined text encoder step")
                    text_encoder_paths.append(DiffusersUtils.find_model_file(text_encoder_dir2))
            else:
                text_encoder_paths.append(DiffusersUtils.find_model_file(text_encoder_dir2))
            
            if model_type == "SD3":
                text_encoder_dir3 = os.path.join(full_path, "text_encoder_3")
                
                #For text_encoder_3, we will use the index file
                index_file = os.path.join(text_encoder_dir3, "text_encoder_3_model.safetensors.index.fp16.json")
                if os.path.exists(index_file):
                    text_encoder_paths.append(cls.load_sd3_text_encoder_3(index_file))
                else:
                    #If index file not found
                    text_encoder_paths.append(DiffusersUtils.find_model_file(text_encoder_dir3))
        
        else: #For SD15 or other single text encoder models
            text_encoder_dir = os.path.join(full_path, "text_encoder")
            text_encoder_paths = [DiffusersUtils.find_model_file(text_encoder_dir)]
        
        return text_encoder_paths
    
    @classmethod
    def load_sd3_text_encoder_3(cls, index_file):
        with open(index_file, 'r') as f:
            index_data = json.load(f)
            
        weight_map = index_data['weight_map']
        base_path = os.path.dirname(index_file)
        
        sd = {}
        for key, file_name in weight_map.items():
            file_path = os.path.join(base_path, file_name)
            if os.path.exists(file_path):
                part_sd = comfy.utils.load_torch_file(file_path, safe_load=True)
                sd.update({key: part_sd[key] for key in part_sd if key in weight_map})
        
        return sd