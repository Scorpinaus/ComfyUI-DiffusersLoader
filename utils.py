import os
import safetensors.torch
import torch

class DiffusersUtils:
    @staticmethod
    def get_base_path():
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers'))
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_model_directories(base_path):
        model_dirs = []
        for root, dirs, files in os.walk(base_path):
            if "model_index.json" in files:
                model_dirs.append(os.path.relpath(root, base_path))
            dirs[:] = [d for d in dirs if d not in ["vae", "unet", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer"]]
        return model_dirs

    @staticmethod
    def find_model_files(directory, file_parts=None):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.endswith((".safetensors", ".bin"))]
        
        if not files:
            raise FileNotFoundError(f"No .safetensors or .bin file found in {directory}")
        
        if file_parts:
            return [file for file in files if any(part in file for part in file_parts)]
        return files
    
    @staticmethod
    def find_model_file(directory):
        files = DiffusersUtils.find_model_files(directory)
        if files:
            return files[0]
        raise FileNotFoundError(f"No .safetensors or .bin file found in {directory}")
    
    @staticmethod
    def load_safetensor_paths(file_paths):
        tensors = [safetensors.torch.load_file(file) for file in file_paths]
        combined_tensors = {}
        for tensor_dict in tensors:
            for key, value in tensor_dict.items():
                if key in combined_tensors:
                    combined_tensors[key] = torch.cat((combined_tensors[key], value), dim=0)
                else:
                    combined_tensors[key] = value
        return combined_tensors    
    
    @staticmethod
    def combine_safetensor_files(text_encoder_dir3, base_path):
        part_files = DiffusersUtils.find_model_files(text_encoder_dir3, ["00001-of-00002", "00002-of-00002"])
        combined_tensors = DiffusersUtils.load_safetensor_paths(part_files)
        combined_file_path = os.path.join(text_encoder_dir3, "combined_text_encoder.safetensors")
        safetensors.torch.save_file(combined_tensors, combined_file_path)
        
        return combined_file_path