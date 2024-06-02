import os
from .utils import load_diffusers_model, load_diffusers_sdxl_model
from .folder_paths import get_folder_paths

class DiffusersLoader:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "advanced/loaders/diffusers"

    def load_checkpoint(self, model_path, output_vae=True, output_clip=True):
        for search_path in get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break

        return load_diffusers_model(model_path, output_vae=output_vae, output_clip=output_clip)


class SDXLDiffusersLoader:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "TEXT_ENCODER_2")
    FUNCTION = "load_checkpoint"
    CATEGORY = "advanced/loaders/sdxl"

    def load_checkpoint(self, model_path, output_vae=True, output_clip=True):
        for search_path in get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break

        return load_diffusers_sdxl_model(model_path, output_vae=output_vae, output_clip=output_clip)
