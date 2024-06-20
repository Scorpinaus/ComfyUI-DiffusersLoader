import os
import comfy.sd
import comfy.utils
import torch
import safetensors.torch

class CombinedDiffusersLoader:
    @staticmethod
    def get_base_path():
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers'))
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"CombinedDiffusersLoader: The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def get_model_directories(base_path):
        model_dirs = []
        for root, dirs, files in os.walk(base_path):
            if "model_index.json" in files:
                model_dirs.append(os.path.relpath(root, base_path))
            #Remove unwanted sub-directories from dirs
            dirs[:] = [d for d in dirs if d not in ["vae", "unet", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer"]] 
        return model_dirs

    @staticmethod
    def find_model_files(directory, file_parts=None):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        files = []
        for file in os.listdir(directory):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                files.append(os.path.join(directory, file))
        
        if not files:
            raise FileNotFoundError(f"CombinedDiffusersLoader: No .safetensors file or .bin file found in {directory}")
        
        if file_parts:
            return [file for file in files if any(part in file for part in file_parts)]
        return files
    
    @staticmethod
    def find_model_file(directory):
        files = CombinedDiffusersLoader.find_model_files(directory)
        if files:
            return files[0]
        raise FileNotFoundError(f"CombinedDiffusersLoader: No .safetensors or .bin file found in {directory}")
    
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
        part_files = CombinedDiffusersLoader.find_model_files(text_encoder_dir3, ["00001-of-00002", "00002-of-00002"])
        combined_tensors = CombinedDiffusersLoader.load_safetensor_paths(part_files)
        combined_file_path = os.path.join(text_encoder_dir3, "combined_text_encoder.safetensors")
        safetensors.torch.save_file(combined_tensors, combined_file_path)
        
        return combined_file_path

    @classmethod
    def INPUT_TYPES(cls):
        base_path = cls.get_base_path()
        model_directories = cls.get_model_directories(base_path)
        return {
            "required": {
                "sub_directory": (model_directories,),
                "clip_type": (["stable_diffusion", "stable_cascade"],),
                "file_parts": (["none", "all", "part_1", "part_2",],)
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_models"
    CATEGORY = "DiffusersLoader/Combined"

    def load_models(self, sub_directory, clip_type="stable_diffusion", file_parts="all"):
        base_path = self.get_base_path()
        sub_dir_path = os.path.join(base_path, sub_directory)
        
        model_type = self.detect_model_type(sub_dir_path)
        
        vae_model = self.load_vae(sub_dir_path, model_type)
        unet_model = self.load_unet(sub_dir_path, model_type)
        clip_model = self.load_clip(sub_dir_path, clip_type, model_type, file_parts)

        return unet_model, clip_model, vae_model

    def detect_model_type(self, sub_dir_path):
        text_encoder_dir1 = os.path.join(sub_dir_path, "text_encoder")
        text_encoder_dir2 = os.path.join(sub_dir_path, "text_encoder_2")
        text_encoder_dir3 = os.path.join(sub_dir_path, "text_encoder_3")
        transformer = os.path.join(sub_dir_path, "transformer")
        
        if os.path.exists(text_encoder_dir3) and os.path.exists(transformer):
            print("This model is SD3")
            return "SD3"
        elif os.path.exists(text_encoder_dir1) and os.path.exists(text_encoder_dir2):
            print("This model is SDXL")
            return "SDXL"
        elif os.path.exists(text_encoder_dir1):
            print("This model is SD1.5")
            return "SD15"
        else:
            print(f"CombinedDiffusersLoader: Sub_dir_path: \n{sub_dir_path}")
            raise FileNotFoundError("No valid text_encoder directories found. This model is not SD15 or SDXL or SD3")

    def load_clip(self, base_path, clip_type="stable_diffusion", model_type="SD15", file_parts="all"):
        clip_type_enum = comfy.sd.CLIPType.STABLE_DIFFUSION
        if clip_type == "stable_cascade":
            clip_type_enum = comfy.sd.CLIPType.STABLE_CASCADE

        text_encoder_dir1 = os.path.join(base_path, "text_encoder")
        text_encoder_paths = [CombinedDiffusersLoader.find_model_file(text_encoder_dir1)]

        if model_type in ["SDXL", "SD3"]:
            text_encoder_dir2 = os.path.join(base_path, "text_encoder_2")
            text_encoder_paths.append(CombinedDiffusersLoader.find_model_file(text_encoder_dir2))
            
        if model_type == "SD3":
            text_encoder_dir3 = os.path.join(base_path, "text_encoder_3")
            if file_parts == "all":
                combined_file_path = os.path.join(text_encoder_dir3, "combined_text_encoder.safetensors")
                if not os.path.exists(combined_file_path):
                    combined_file_path = CombinedDiffusersLoader.combine_safetensor_files(text_encoder_dir3, base_path)
                text_encoder_paths.append(combined_file_path)
            elif file_parts == "part_1":
                text_encoder_paths += CombinedDiffusersLoader.find_model_files(text_encoder_dir3, ["00001-of-00002"])
            elif file_parts == "part_2":
                text_encoder_paths += CombinedDiffusersLoader.find_model_files(text_encoder_dir3, ["00002-of-00002"])
            
        print(f"CombinedDiffusersLoader: Checking paths: \n{text_encoder_paths}")
        
        try:
            clip_model = comfy.sd.load_clip(ckpt_paths=text_encoder_paths, embedding_directory=os.path.join(base_path, "embeddings"), clip_type=clip_type_enum)
        except Exception as e:
            print(f"Error loading clip model: {e}")
            raise

        # Debugging statement to check if the loaded model has 'tokenize' method
        if not hasattr(clip_model, 'tokenize'):
            raise AttributeError("Loaded clip model does not have 'tokenize' method.")

        return clip_model
    
    def load_unet(self, base_path, model_type):
        if model_type == "SD3":
            unet_folder = os.path.join(base_path, 'transformer')
        else:
            unet_folder = os.path.join(base_path, 'unet')

        if not os.path.exists(unet_folder):
            raise FileNotFoundError(f"CombinedDiffusersLoader: The directory '{unet_folder}' does not exist for model type '{model_type}'")

        unet_path = CombinedDiffusersLoader.find_model_file(unet_folder)
        print(f"CombinedDiffusersLoader: Attempting to load UNET model from {unet_path}")

        try:
            return comfy.sd.load_unet(unet_path)
        except PermissionError as e:
            print(f"Permissionerror:{e}")
            raise PermissionError(f"Permission denied: {unet_path}")
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            raise FileNotFoundError(f"UNET file not found: {unet_path}")
        except Exception as e:
            print(f"Error loading UNET model: {e}")
            raise e

    def load_vae(self, base_path, model_type):
        vae_folder = os.path.join(base_path, "vae")
        vae_path = CombinedDiffusersLoader.find_model_file(vae_folder)
        print(f"Attempting to load VAE model from: {vae_path}")

        vae_sd = comfy.utils.load_torch_file(vae_path)
        return comfy.sd.VAE(sd=vae_sd)

# Ensure the node is registered properly
NODE_CLASS_MAPPINGS = {"CombinedDiffusersLoader": CombinedDiffusersLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CombinedDiffusersLoader": "Combined Diffusers Loader"}
