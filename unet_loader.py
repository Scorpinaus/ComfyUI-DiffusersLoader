# unet_loader.py
import os
import comfy.sd
import comfy.utils
import json
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
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet/transformer"
    CATEGORY = "DiffusersLoader"

    @classmethod
    def load_unet(cls, sub_directory, transformer_parts="all", weight_dtype="default"):
        return (cls.load_model(sub_directory, transformer_parts, weight_dtype),)

    @classmethod
    def load_model(cls, sub_directory, transformer_parts="all", weight_dtype="default"):
        base_path = DiffusersUtils.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        model_type = cls.detect_model_type(sub_dir_path)
        print(f"DiffusersUNETLoader: Detected model type: {model_type}")
        
        unet_path = cls.get_unet_path(sub_dir_path, model_type, transformer_parts)
        DiffusersUtils.check_and_clear_cache('unet', unet_path)
        
        print(f"DiffusersUNETLoader: Attempting to load UNET model from: {unet_path}")
        
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
                    
        try:
            if model_type == "AuraFlow" and transformer_parts == "all" and unet_path.endswith('.json'):
                #Load model using index_file
                with open(unet_path, 'r') as f:
                    index_data = json.load(f)
                weight_map = index_data['weight_map']
                state_dict = {}
                
                for key, file in weight_map.items():
                    file_path = os.path.join(os.path.dirname(unet_path), file)
                    state_dict.update(comfy.utils.load_torch_file(file_path))
                model = cls.load_diffusion_model_from_state_dict(state_dict, model_options=model_options)
            else:
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            print("UNET/Transformer model loaded successfully")
            return model
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory error. Attempting to clear memory and retry...")
                DiffusersUtils.clear_memory()
                try:
                    if model_type == "AuraFlow" and transformer_parts == "all" and unet_path.endswith('.json'):
                        # Load model using the index file (retry)
                        with open(unet_path, 'r') as f:
                            index_data = json.load(f)
                        weight_map = index_data['weight_map']
                        state_dict = {}
                        for key, file in weight_map.items():
                            file_path = os.path.join(os.path.dirname(unet_path), file)
                            state_dict.update(comfy.utils.load_torch_file(file_path))
                        model = cls.load_diffusion_model_from_state_dict(state_dict, model_options=model_options)
                    else:
                        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                    print("UNET/Transformer model loaded successfully after memory clear")
                    return model
                except Exception as retry_e:
                    print(f"Error loading model after memory clear: {retry_e}")
                    raise
            else:
                print(f"DiffusersUNETLoader: Error loading UNET/Transformer model: {e}")
                raise
    
    @classmethod
    def load_diffusion_model_from_state_dict(cls, state_dict, model_options):
        model = comfy.sd.load_diffusion_model_state_dict(state_dict, model_options=model_options)
        return model
    
    @classmethod
    def get_unet_path(cls, sub_dir_path, model_type, transformer_parts):
        if model_type in ["AuraFlow", "Flux", "SD3"]:
            unet_folder = os.path.join(sub_dir_path, 'transformer')
            print(f"Using transformer folder for {model_type}: {unet_folder}")
            if model_type == "SD3":
                return cls.handle_sd3_transformer(unet_folder)
            if model_type == "AuraFlow":
                return cls.handle_auraflow_transformer(unet_folder, transformer_parts)
            elif model_type == "Flux":
                return cls.handle_flux_transformer(unet_folder, transformer_parts)
        else:  # SD15, SDXL, etc.
            unet_folder = os.path.join(sub_dir_path, 'unet')
            if not os.path.exists(unet_folder):
                #Fallback to transformer folder if unet folder does not exist
                unet_folder = os.path.join(sub_dir_path, 'transformer')	
                if os.path.exists(unet_folder):
                    print(f"Unet folder not found, using transformer folder: {unet_folder}")
                    return cls.handle_flux_transformer(unet_folder, transformer_parts)
            return DiffusersUtils.find_model_file(unet_folder)       
    
    @classmethod
    def handle_transformer(cls, unet_folder, transformer_parts, num_parts):
        if transformer_parts == "all":
            combined_file_path = os.path.join(unet_folder, "combined_transformer.safetensors")
            if not os.path.exists(combined_file_path):
                print(f"Combined file not found, creating at: {combined_file_path}")
                try:
                    combined_file_path = DiffusersUtils.combine_safetensor_files(unet_folder, os.path.dirname(unet_folder), num_parts=num_parts)
                except Exception as e:
                    print(f"Error combining safetensor files: {e}")
                    combined_file_path = DiffusersUtils.find_model_files(unet_folder, num_parts=1)[0]
            return combined_file_path
        else:
            part_num = int(transformer_parts.split('_')[1])
            return DiffusersUtils.find_model_files(unet_folder, num_parts=part_num)[-1]

    @classmethod
    def handle_auraflow_transformer(cls, unet_folder, transformer_parts):
        #Find file ending with 'index.json'
        index_files = [f for f in os.listdir(unet_folder) if f.endswith('index.json')] 
        print(f"Index files: {index_files}")
        if index_files:
            index_file = os.path.join(unet_folder, index_files[0])
            print(f"Found index file: {index_file}")
            
            if transformer_parts == "all":
                return index_file
            
            else:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)    
                    
                part_num = int(transformer_parts.split('_')[1])
                part_files = [file for file in index_data['weight_map'].values() if f"-0000{part_num}-of-" in file]
                
                if part_files:
                    return os.path.join(unet_folder, part_files[0])
                else:
                    print(f"No file found for part {part_num}")
                    return None
        else:
            print(f"No index file found in {unet_folder}. Checking for combined transformer step")
            return cls.handle_transformer(unet_folder, transformer_parts, num_parts=2)

    @classmethod
    def handle_sd3_transformer(cls, unet_folder):
        return DiffusersUtils.find_model_file(unet_folder)

    @classmethod
    def handle_flux_transformer(cls, unet_folder, transformer_parts):
        return cls.handle_transformer(unet_folder, transformer_parts, num_parts=3)

