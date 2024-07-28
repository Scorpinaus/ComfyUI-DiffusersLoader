# base_loader.py
import os
from .utils import DiffusersUtils

class DiffusersLoaderBase:
    @classmethod
    def detect_model_type(cls, sub_dir_path):
        text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
        text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")
        text_encoder_dir3 = os.path.join(sub_dir_path, "text_encoder_3")
        transformer = os.path.join(sub_dir_path, "transformer")
        
        if os.path.exists(text_encoder_dir3) and os.path.exists(transformer):
            return "SD3"
        elif os.path.exists(text_encoder_dir1) and os.path.exists(text_encoder_dir2):
            return "SDXL"
        elif os.path.exists(text_encoder_dir1) and os.path.exists(transformer):
            return "AuraFlow"
        elif os.path.exists(text_encoder_dir1):
            return "SD15"
        else:
            raise FileNotFoundError("No valid text_encoder directories found. This model is not SD15, SDXL, or SD3")

    @classmethod
    def load_model(cls, sub_directory):
        raise NotImplementedError