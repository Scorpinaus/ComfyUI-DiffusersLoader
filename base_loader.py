# base_loader.py
import os
import json
from .utils import DiffusersUtils
from .model_type_config import MODEL_TYPE_CRITERIA

class DiffusersLoaderBase:
    @classmethod
    def detect_model_type(cls, sub_dir_path):
        model_index_path = os.path.join(sub_dir_path, "model_index.json")
        
        if os.path.exists(model_index_path):
            with open(model_index_path, 'r') as f:
                model_info = json.load(f)
            
            class_name = model_info.get("_class_name")
            print("Class Name:", class_name)
            components = set(model_info.keys())
            print("Components:", components)
            
            for model_type, criteria in MODEL_TYPE_CRITERIA.items():
                if class_name == criteria["class_name"]:
                    required_components = set(criteria["required_components"])
                    if required_components.issubset(components):
                        if "absent_components" in criteria:
                            absent_components = set(criteria["absent_components"])
                            if absent_components.intersection(components):
                                continue

                        # Check additional criteria
                        additional_criteria_match = True
                        for key in ["feature_extractor", "requires_safety_checker", "force_zeros_for_empty_prompt", "image_encoder"]:
                            if key in criteria:
                                if model_info.get(key) != criteria[key]:
                                    additional_criteria_match = False
                                    break
                        
                        if additional_criteria_match:
                            return model_type
        
        # If we couldn't determine the type from model_index.json, detect based on folder structure
        if os.path.exists(os.path.join(sub_dir_path, "transformer")):
            if os.path.exists(os.path.join(sub_dir_path, "text_encoder_3")):
                return "SD3"
            elif os.path.exists(os.path.join(sub_dir_path, "text_encoder_2")):
                return "SDXL"
            else:
                # We can't reliably distinguish between AuraFlow and Flux based on folder structure alone
                return "AuraFlow_or_Flux"
        elif os.path.exists(os.path.join(sub_dir_path, "text_encoder_2")):
            return "SDXL"
        elif os.path.exists(os.path.join(sub_dir_path, "text_encoder")):
            # We can't reliably distinguish between SD15 and SD21 based on folder structure alone
            return "SD15_or_SD21"
        
        return "Unknown"

    @classmethod
    def load_model(cls, sub_directory):
        raise NotImplementedError