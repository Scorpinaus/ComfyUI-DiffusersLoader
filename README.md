# Diffusers Format Checkpoint Loaders for ComfyUI

This project aims to create loaders for diffusers format checkpoint models, making it easier for ComfyUI users to use diffusers format checkpoints instead of the standard checkpoint formats.

This project was created to understand how the DiffusersLoader available in comfyUI works and enhance the functionality by making usable loaders.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project provides tools and loaders for integrating diffusers format checkpoint models with ComfyUI. It helps users who already have diffusers format checkpoints to seamlessly use them without needing to convert them to the standard checkpoint format.


## Installation

Follow these steps to install and set up the project:
1. Download the node via ComfyUI-Manager by searching: DiffusersLoader:
![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/f4e962f9-aee3-4027-9e8b-c559451cf819)

2. Add nodes to your workflow by:
  - Add Nodes -> DiffusersLoader -> Combined -> CombinedDiffusersLoader
  - ![image](https://github.com/user-attachments/assets/79a576b2-dc27-49e4-a7c7-ec0e01bc5bad)

3. Connect the DiffusersLoader to your workflow

## Features
The CombinedDiffusersLoader supports loading of diffusers checkpoints for:
- SD 1.5
- SD 2.1
- SDXL
- SD3
- AuraFlow
- Flux

Supports loading via extra_model_paths.yaml - Using ComfyUI
- If there are 2 model directories of the same name in different places, the model_name(1) refers to the directory in your 1st base path while model_name(2) refers to the directory in your 2nd base path.

### Workflow Example
The combined loader work in the same manner as existing checkpoints loader as seen in this workflow: 

![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/6b079ac4-1479-43e2-87f6-879919e34d0b)

Select the name of the diffusers checkpoint folder from the dropdown list and connect the nodes.


## Limitations & Future Improvements
- Add support for other compatible diffusers format checkpoints
  - Future model_types:
    - HunyuanDiT
    - PixArt
    - Kolors
    Note: This will need to raise a PR on main comfyUI repo
- For Flux, use of default weight_dtype would OOM (Use fp8-based d-types)



## Contributing
Contributions are always welcome! Please follow the contributing guidelines to submit issues or pull requests.

## License
This project is licensed under the MIT License.

## Contact the Author
Hafiz Saffie - hafiz.safs@gmail.com

Project Link: https://github.com/Scorpinaus/ComfyUI-DiffusersLoader

## Update History
26/8 - Improved on model detection to include extra_model_paths.yaml.
25/8 - Implemented multi-file loading support where the combined_safetensor.safetensor file method is used as a back-up
21/8 - Compatibility Update With ComfyUI, Addition of Flux Support
30/7 - Added repo to comfyUI registry
28/7 - Added AuraFlow support
20/6 - Added SD3 support
2/6 - Initial commit
