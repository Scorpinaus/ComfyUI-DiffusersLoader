import os

def get_comfyui_base_path():
    # Assuming the script is in the standard directory structure
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_folder_paths(folder_type):
    base_path = get_comfyui_base_path()
    
    if folder_type == "diffusers":
        return [os.path.join(base_path, "models", "diffusers")]
    if folder_type == "embeddings":
        return [os.path.join(base_path, "models", "embeddings")]
    
    return []

# Example usage:
if __name__ == "__main__":
    print(get_folder_paths("diffusers"))  # Should print the path to the diffusers models
    print(get_folder_paths("embeddings"))  # Should print the path to the embeddings
