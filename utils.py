import os
import safetensors.torch
import torch
import gc
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import folder_paths

class DiffusersUtils:
    _model_cache = {'clip': None, 'unet': None, 'vae': None}
    _current_model_hashes = {'clip': None, 'unet': None, 'vae': None}
    
    @staticmethod
    def clear_memory():
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleared CUDA Memory and ran garbage collection")
    
    @classmethod
    def clear_model_cache(cls):
        cls._model_cache = {'clip': None, 'unet': None, 'vae': None}
        cls._current_model_hashes = {'clip': None, 'unet': None, 'vae': None}
        cls.clear_memory()
        print("Cleared model cache")
    
    @classmethod
    def get_model_hash(cls, model_path):
        import hashlib
        with open(model_path, "rb") as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        return file_hash.hexdigest()
    
    @classmethod
    def check_and_clear_cache(cls, model_type, model_path):
        new_hash = cls.get_model_hash(model_path)
        if cls._current_model_hashes[model_type] != new_hash:
            print(f"Detected change in {model_type} model. Clearing cache.")
            cls.clear_model_cache()
            cls._current_model_hashes[model_type] = new_hash
        else:
            print(f"No change detected in {model_type} model.")
    
    @staticmethod
    def get_base_path():
        base_path = folder_paths.get_folder_paths("diffusers")
        #print(f"Base path: {base_path}")
        if not base_path:
            raise FileNotFoundError(f"The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_model_directories():
        paths = []
        base_paths = DiffusersUtils.get_base_path()
        for base_path in base_paths:
            if os.path.exists(base_path):
                for root, _, files in os.walk(base_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=base_path))
        return paths

    @staticmethod
    def find_model_files(directory, file_parts=None):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.endswith((".safetensors", ".bin", "index.json"))]
        
        if not files:
            raise FileNotFoundError(f"No .safetensors or .bin or index.json file found in {directory}")
        
        if file_parts:
            filtered_files = [file for file in files if any(part in file for part in file_parts)]
        else:
            filtered_files = files
        print("Filtered files:", filtered_files)
        return sorted(filtered_files)
    
    
    @staticmethod
    def find_model_file(directory):
        files = DiffusersUtils.find_model_files(directory)
        if files:
            return files[0]
        raise FileNotFoundError(f"No .safetensors or .bin pr index.json file found in {directory}")
    
    @staticmethod
    def load_safetensor_paths(file_paths):
        print("Running Loading Safetensors")
        combined_tensors = {}
        for file_path in file_paths:
            print(f"Loading file: {file_path}")
            try:
                tensor_dict = safetensors.torch.load_file(file_path)
                for key, value in tensor_dict.items():
                    print(f"Processing key: {key}")
                    if key in combined_tensors:
                        try:
                            combined_tensors[key] = torch.cat((combined_tensors[key], value), dim=0)
                        except RuntimeError as e:
                            print(f"Error concatenating tensors for key {key}: {e}")
                            print(f"Shapes: combined {combined_tensors[key].shape}, new {value.shape}")
                            # Instead of raising, we'll skip this tensor and continue
                            print(f"Skipping problematic tensor: {key}")
                    else:
                        combined_tensors[key] = value
                print(f"Finished processing file: {file_path}")
                # Free up memory
                del tensor_dict
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                # Continue to the next file instead of stopping
                continue
        print("Combining Safetensors completed")
        return combined_tensors
        
    #This function is not used. Will be removed in the subsequent release
    @staticmethod
    def combine_safetensor_files(directory, base_path, num_parts):
        print(f"Running combine_safetensor_files for directory: {directory}")
        part_files = DiffusersUtils.find_model_files(directory, num_parts=num_parts)
        print("Path to part_files:", part_files)
        try:
            combined_tensors = DiffusersUtils.load_safetensor_paths(part_files)
            print("Combined tensors keys:", list(combined_tensors.keys()))
            
            if "text_encoder" in directory:
                combined_file_path = os.path.join(directory, "combined_text_encoder.safetensors")
            elif "transformer" in directory:
                combined_file_path = os.path.join(directory, "combined_transformer.safetensors")
            else:
                raise ValueError(f"Unsupported directory for combining files: {directory}")
            
            print(f"Saving combined file to: {combined_file_path}")
            safetensors.torch.save_file(combined_tensors, combined_file_path)
            print("Combined file saved successfully")
            
            return combined_file_path
        except Exception as e:
            print(f"Error in combine_safetensor_files: {e}")
            # If combining fails, return the path to the first file as a fallback
            return part_files[0]
    