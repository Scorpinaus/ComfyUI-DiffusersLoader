# loader_factory.py
from .unet_loader import DiffusersUNETLoader
from .clip_loader import DiffusersClipLoader
from .vae_loader import DiffusersVAELoader

class LoaderFactory:
    @staticmethod
    def create_loader(loader_type, sub_directory):
        if loader_type == "UNET":
            return DiffusersUNETLoader(sub_directory)
        elif loader_type == "CLIP":
            return DiffusersClipLoader(sub_directory)
        elif loader_type == "VAE":
            return DiffusersVAELoader(sub_directory)
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")