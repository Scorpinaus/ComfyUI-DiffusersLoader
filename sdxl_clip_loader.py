import os
from comfy import sd

class SDXLCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.abspath(os.path.join(script_dir, "../../models/diffusers/SDXL"))
        print(f"SDXLCLIPLoader: Base path for SDXL models: {base_path}")

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"SDXLCLIPLoader: The base path {base_path} does not exist.")

        sub_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        print(f"SDXLCLIPLoader: Available subdirectories: {sub_dirs}")
        return {
            "required": {
                "sub_directory": (sub_dirs, ),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_sdxl_clip"
    CATEGORY = "testing_nodes"

    def load_sdxl_clip(self, sub_directory):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.abspath(os.path.join(script_dir, "../../models/diffusers/SDXL", sub_directory))

        text_encoder_dir1 = os.path.join(base_path, "text_encoder")
        text_encoder_dir2 = os.path.join(base_path, "text_encoder_2")

        text_encoder_file1 = self.find_safetensors_file(text_encoder_dir1)
        text_encoder_file2 = self.find_safetensors_file(text_encoder_dir2)

        text_encoder_path1 = os.path.join(text_encoder_dir1, text_encoder_file1)
        text_encoder_path2 = os.path.join(text_encoder_dir2, text_encoder_file2)

        print(f"SDXLCLIPLoader: Checking paths:\n{text_encoder_path1}\n{text_encoder_path2}")

        path1_exists = os.path.exists(text_encoder_path1)
        path2_exists = os.path.exists(text_encoder_path2)

        print(f"SDXLCLIPLoader: Path 1 exists: {path1_exists}")
        print(f"SDXLCLIPLoader: Path 2 exists: {path2_exists}")

        if not path1_exists or not path2_exists:
            raise FileNotFoundError(f"SDXLCLIPLoader: One or both text encoder files not found in {base_path}")

        clip = sd.load_clip(ckpt_paths=[text_encoder_path1, text_encoder_path2], embedding_directory=os.path.join(base_path, "embeddings"))
        return (clip,)

    @staticmethod
    def find_safetensors_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors"):
                return file
        raise FileNotFoundError(f"No .safetensors file found in {directory}")

# Mapping the node to make it recognizable in the ComfyUI framework
NODE_CLASS_MAPPINGS = {
    "SDXLCLIPLoader": SDXLCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLCLIPLoader": "SDXL CLIP Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
