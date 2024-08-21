# base_loader.py
import os
import json
from .utils import DiffusersUtils

class DiffusersLoaderBase:
    @classmethod
    def detect_model_type(cls, sub_dir_path):
        model_index_path = os.path.join(sub_dir_path, "model_index.json")
        
        if os.path.exists(model_index_path):
            with open(model_index_path, 'r') as f:
                model_info = json.load(f)
            
            class_name = model_info.get("_class_name")
            print("Class Name:", class_name)
            
            if class_name:
                if class_name == "FluxPipeline":
                    return "Flux"
                elif class_name == "StableDiffusionPipeline":
                    if "text_encoder_2" in model_info:
                        return "SDXL"
                    else:
                        return "SD15"
                elif class_name == "StableDiffusion3Pipeline":
                    return "SD3"
                elif class_name == "AuraFlowPipeline":
                    return "AuraFlow"
        
        # If we couldn't determine the type from model_index.json, detect based on folder structure
        if os.path.exists(os.path.join(sub_dir_path, "transformer")):
            return "Flux"
        elif os.path.exists(os.path.join(sub_dir_path, "text_encoder_2")):
            return "SDXL"
        elif os.path.exists(os.path.join(sub_dir_path, "text_encoder")):
            return "SD15"
        else:
            return "Unknown"

    @classmethod
    def load_model(cls, sub_directory):
        raise NotImplementedError