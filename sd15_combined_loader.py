import os
import comfy.sd
import comfy.utils
import torch

class CombinedDiffusersSD15Loader:
    @classmethod
    def INPUT_TYPES(cls):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.join(script_dir, "..", "..", "models", "diffusers", "SD15")
        try:
            sub_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        except FileNotFoundError:
            sub_dirs = []

        return {"required": {"sub_directory": (sub_dirs,),
                             "clip_type": (["stable_diffusion", "stable_cascade"],)}}

    RETURN_TYPES = ("CLIP", "MODEL", "VAE")
    FUNCTION = "load_models"
    CATEGORY = "testing_nodes"

    def load_models(self, sub_directory, clip_type="stable_diffusion"):
        clip = self.load_clip(sub_directory, clip_type)
        unet = self.load_unet(sub_directory)
        vae = self.load_vae(sub_directory)
        return clip, unet, vae

    def load_clip(self, sub_directory, clip_type="stable_diffusion"):
        clip_type_enum = comfy.sd.CLIPType.STABLE_DIFFUSION
        if clip_type == "stable_cascade":
            clip_type_enum = comfy.sd.CLIPType.STABLE_CASCADE

        script_dir = os.path.dirname(os.path.realpath(__file__))
        text_encoder_dir = os.path.join(script_dir, "..", "..", "models", "diffusers", "SD15", sub_directory, "text_encoder")
        text_encoder_file = self.find_safetensors_file(text_encoder_dir)
        text_encoder_path = os.path.join(text_encoder_dir, text_encoder_file)

        if not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"File not found: {text_encoder_path}")

        clip = comfy.sd.load_clip(ckpt_paths=[text_encoder_path], embedding_directory=os.path.join(script_dir, "..", "..", "models", "diffusers", "SD15", sub_directory, "embeddings"), clip_type=clip_type_enum)
        return clip

    def load_unet(self, sub_directory):
        unet_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers', 'SD15', sub_directory, 'unet'))
        diffusion_model_path = self.find_safetensors_file(unet_folder)

        if not os.path.exists(diffusion_model_path):
            raise FileNotFoundError(f"{os.path.basename(diffusion_model_path)} not found in the directory: {diffusion_model_path}")

        try:
            model = comfy.sd.load_unet(diffusion_model_path)
            return model
        except PermissionError as e:
            self.handle_corrupted_file(unet_folder)
            raise e

    def load_vae(self, sub_directory):
        base_path = self.get_base_path()
        vae_folder = os.path.join(base_path, sub_directory, "vae")
        vae_filename = self.find_safetensors_file(vae_folder)
        vae_path = os.path.join(vae_folder, vae_filename)
        full_path = os.path.abspath(vae_path)

        print(f"Attempting to load VAE model from: {full_path}")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The VAE file '{full_path}' does not exist.")

        sd = comfy.utils.load_torch_file(full_path)
        vae = comfy.sd.VAE(sd=sd)
        return vae

    @staticmethod
    def get_base_path():
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diffusers', 'SD15')
        base_path = os.path.abspath(base_path)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"The base path '{base_path}' does not exist.")
        return base_path

    @staticmethod
    def find_safetensors_file(directory):
        for file in os.listdir(directory):
            if file.endswith(".safetensors"):
                return os.path.join(directory, file)
        raise FileNotFoundError(f"No .safetensors file found in {directory}")

    @staticmethod
    def handle_corrupted_file(unet_folder):
        corrupted_path = unet_folder + ".corrupted"
        try:
            os.rename(unet_folder, corrupted_path)
        except Exception as e:
            raise e

# Register the node in the ComfyUI framework
NODE_CLASS_MAPPINGS = {
    "CombinedDiffusersSD15Loader": CombinedDiffusersSD15Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombinedDiffusersSD15Loader": "Combined Diffusers SD1.5 Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
