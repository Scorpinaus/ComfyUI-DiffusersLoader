from .loader import DiffusersLoader, SDXLDiffusersLoader

NODE_CLASS_MAPPINGS = {
    "Diffusers_Loader": DiffusersLoader,
    "SDXL_Diffusers_Loader": SDXLDiffusersLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Diffusers_Loader": "Diffusers Loader",
    "SDXL_Diffusers_Loader": "SDXL Diffusers Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
