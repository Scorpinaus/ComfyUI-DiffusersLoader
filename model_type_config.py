MODEL_TYPE_CRITERIA = {
    "SD15": {
        "class_name": "StableDiffusionPipeline",
        "required_components": ["text_encoder", "unet", "vae", "scheduler", "tokenizer"],
        "absent_components": ["text_encoder_2"],
        "feature_extractor": [None, None],
        "requires_safety_checker": True
    },
    "SD21": {
        "class_name": "StableDiffusionPipeline",
        "required_components": ["text_encoder", "unet", "vae", "scheduler", "tokenizer"],
        "absent_components": ["text_encoder_2"],
        "feature_extractor": ["transformers", "CLIPImageProcessor"],
        "requires_safety_checker": False
    },
    "SDXL": {
        "class_name": "StableDiffusionXLPipeline",
        "required_components": ["text_encoder", "text_encoder_2", "unet", "vae", "scheduler", "tokenizer", "tokenizer_2"],
    },
    "SD3": {
        "class_name": "StableDiffusion3Pipeline",
        "required_components": ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae", "scheduler", "tokenizer", "tokenizer_2", "tokenizer_3"]
    },
    "AuraFlow": {
        "class_name": "AuraFlowPipeline",
        "required_components": ["text_encoder", "transformer", "vae", "scheduler", "tokenizer"]
    },
    "Flux": {
        "class_name": "FluxPipeline",
        "required_components": ["text_encoder", "transformer", "vae", "scheduler", "tokenizer"]
    }
}