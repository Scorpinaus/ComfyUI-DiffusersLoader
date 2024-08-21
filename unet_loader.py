# unet_loader.py
import os
import comfy.sd
from .base_loader import DiffusersLoaderBase
from .utils import DiffusersUtils
import torch

class DiffusersUNETLoader(DiffusersLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sub_directory": (DiffusersUtils.get_model_directories(DiffusersUtils.get_base_path()),),
                "transformer_parts": (["all", "part_1", "part_2", "part_3"],),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "DiffusersLoader"

    @classmethod
    def load_unet(cls, sub_directory, transformer_parts="all"):
        return (cls.load_model(sub_directory, transformer_parts),)

    @classmethod
    def load_model(cls, sub_directory, transformer_parts="all"):
        
        base_path = DiffusersUtils.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        model_type = cls.detect_model_type(sub_dir_path)
        print(f"DiffusersUNETLoader: Detected model type: {model_type}")

        if model_type == "AuraFlow":
            unet_folder = os.path.join(sub_dir_path, 'transformer')
            print(f"Using transformer folder for AuraFlow: {unet_folder}")
            unet_path = cls.handle_auraflow_transformer(unet_folder, transformer_parts)
        elif model_type == "Flux":
            unet_folder = os.path.join(sub_dir_path, 'transformer')
            print(f"Using transformer folder for Flux: {unet_folder}")
            unet_path = cls.handle_flux_transformer(unet_folder, transformer_parts)
        elif model_type == "SD3":
            unet_folder = os.path.join(sub_dir_path, 'transformer')
            print(f"Using transformer folder for SD3: {unet_folder}")
            unet_path = cls.handle_sd3_transformer(unet_folder, transformer_parts)
        else:  # SD15, SDXL, etc.
            unet_folder = os.path.join(sub_dir_path, 'unet')
            unet_path = DiffusersUtils.find_model_file(unet_folder)
        
        DiffusersUtils.check_and_clear_cache('unet', unet_path)

        print(f"DiffusersUNETLoader: Attempting to load UNET/Transformer model from {unet_path}")

        try:
            model = comfy.sd.load_unet(unet_path)
            print("UNET/Transformer model loaded successfully")
            return model
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory error. Attempting to clear memory and retry...")
                cls.clear_memory()
                try:
                    model = comfy.sd.load_unet(unet_path)
                    print("UNET/Transformer model loaded successfully after memory clear")
                    return model
                except Exception as retry_e:
                    print(f"Error loading model after memory clear: {retry_e}")
                    raise
            else:
                print(f"DiffusersUNETLoader: Error loading UNET/Transformer model: {e}")
                raise
    
    @classmethod
    def handle_auraflow_transformer(cls, unet_folder, transformer_parts):
        print(f"Handling AuraFlow transformer in folder: {unet_folder}")
        if transformer_parts == "all":
            combined_file_path = os.path.join(unet_folder, "combined_transformer.safetensors")
            if not os.path.exists(combined_file_path):
                print(f"Combined file not found, creating at: {combined_file_path}")
                try:
                    combined_file_path = DiffusersUtils.combine_safetensor_files(unet_folder, os.path.dirname(unet_folder), num_parts=3)
                except Exception as e:
                    print(f"Error combining safetensor files: {e}")
                    # If combining fails, use the first part file
                    combined_file_path = DiffusersUtils.find_model_files(unet_folder, num_parts=1)[0]
            unet_path = combined_file_path
        else:
            part_num = int(transformer_parts.split('_')[1])
            unet_path = DiffusersUtils.find_model_files(unet_folder, num_parts=part_num)[-1]
        
        print(f"Selected AuraFlow transformer path: {unet_path}")
        return unet_path

    @classmethod
    def handle_sd3_transformer(cls, unet_folder, transformer_parts):
        print(f"Handling SD3 transformer in folder: {unet_folder}")
        if transformer_parts == "all":
            combined_file_path = os.path.join(unet_folder, "combined_transformer.safetensors")
            if not os.path.exists(combined_file_path):
                print(f"Combined file not found, creating at: {combined_file_path}")
                try:
                    combined_file_path = DiffusersUtils.combine_safetensor_files(unet_folder, os.path.dirname(unet_folder), num_parts=3)
                except Exception as e:
                    print(f"Error combining safetensor files: {e}")
                    # If combining fails, use the first part file
                    combined_file_path = DiffusersUtils.find_model_files(unet_folder, num_parts=1)[0]
            unet_path = combined_file_path
        else:
            part_num = int(transformer_parts.split('_')[1])
            unet_path = DiffusersUtils.find_model_files(unet_folder, num_parts=part_num)[-1]
        
        print(f"Selected SD3 transformer path: {unet_path}")
        return unet_path
    
    @classmethod
    def handle_flux_transformer(cls, unet_folder, transformer_parts):
        print(f"Handling Flux transformer in folder: {unet_folder}")
        if transformer_parts == "all":
            combined_file_path = os.path.join(unet_folder, "combined_transformer.safetensors")
            if not os.path.exists(combined_file_path):
                print(f"Combined file not found, creating at: {combined_file_path}")
                try:
                    combined_file_path = DiffusersUtils.combine_safetensor_files(unet_folder, os.path.dirname(unet_folder), num_parts=3)
                except Exception as e:
                    print(f"Error combining safetensor files: {e}")
                    # If combining fails, use the first part file
                    combined_file_path = DiffusersUtils.find_model_files(unet_folder, num_parts=1)[0]
            unet_path = combined_file_path
        else:
            part_num = int(transformer_parts.split('_')[1])
            unet_path = DiffusersUtils.find_model_files(unet_folder, num_parts=part_num)[-1]
        
        print(f"Selected Flux transformer path: {unet_path}")
        return unet_path