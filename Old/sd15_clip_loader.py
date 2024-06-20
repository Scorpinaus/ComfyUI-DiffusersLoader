import os
from comfy import sd

class SD15CLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Path to the 'diffusers/SD15' folder, relative to the script location
        script_dir = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.join(script_dir, "..", "..", "models", "diffusers", "SD15")
        print(f"SD15CLIPLoader:Base path: {base_path}")  # Debug print
        
        # Get the list of sub-directories in the 'diffusers/SD15' folder
        try:
            sub_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            print(f"Sub-directories: {sub_dirs}")  # Debug print
        except FileNotFoundError:
            sub_dirs = []
            print("SD15CLIPLoader:Base path not found.")  # Debug print

        return {"required": {"sub_directory": (sub_dirs,),
                             "clip_type": (["stable_diffusion", "stable_cascade"],),
                            }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "DiffusersLoader/SD1.5"

    def load_clip(self, sub_directory, clip_type="stable_diffusion"):
        # Determine the clip type
        clip_type_enum = sd.CLIPType.STABLE_DIFFUSION
        if clip_type == "stable_cascade":
            clip_type_enum = sd.CLIPType.STABLE_CASCADE

        # Construct the path to the text_encoder directory
        script_dir = os.path.dirname(os.path.realpath(__file__))
        text_encoder_dir = os.path.join(script_dir, "..", "..", "models", "diffusers", "SD15", sub_directory, "text_encoder")
        print(f"SD15CLIPLoader:Text encoder directory: {text_encoder_dir}")  # Debug print

        # Find the .safetensors or .bin file in the text_encoder directory
        text_encoder_file = self.find_model_file(text_encoder_dir)
        text_encoder_path = os.path.join(text_encoder_dir, text_encoder_file)
        print(f"SD15CLIPLoader:Text encoder path: {text_encoder_path}")  # Debug print

        # Ensure the file exists
        if not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"SD15CLIPLoader:File not found: {text_encoder_path}")

        # Load the CLIP model using the constructed path
        clip = sd.load_clip(ckpt_paths=[text_encoder_path], embedding_directory=os.path.join(script_dir, "..", "..", "models", "diffusers", "SD15", sub_directory, "embeddings"), clip_type=clip_type_enum)
        return (clip,)

    @staticmethod
    def find_model_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors") or file.endswith(".bin"):
                return file
        raise FileNotFoundError(f"SD15CLIPLoader:No .safetensors file or .bin file found in {directory}")

# Register the node in the ComfyUI framework
NODE_CLASS_MAPPINGS = {
    "SD15CLIPLoader": SD15CLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD15CLIPLoader": "SD1.5 CLIP Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
