# Diffusers Format Checkpoint Loaders for ComfyUI

This project aims to create loaders for diffusers format checkpoint models, making it easier for ComfyUI users to use diffusers format checkpoints instead of the standard checkpoint formats.

This project was created to understand how the DiffusersLoader avaliable in comfyUI works and enhance the functionality by making usable loaders.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project provides tools and loaders for integrating diffusers format checkpoint models with ComfyUI. It helps users who already have diffusers format checkpoints to seamlessly use them without needing to convert them to the standard format.


## Installation

Follow these steps to install and set up the project:
1. Download the node via ComfyUI-Manager by searching: DiffusersLoader:
![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/f4e962f9-aee3-4027-9e8b-c559451cf819)

2. Add nodes to your workflow by:
  - Add Nodes -> DiffusersLoader -> Combined -> CombinedDiffusersLoader
  - ![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/68c3232d-7f05-4a48-88a7-f7498ac2139c)
  - ![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/a83724a1-9afb-467a-b3d4-9006afcc6557)



## Features
The CombinedDiffusersLoader supports loading of diffusers checkpoints for:
- SD1.5
- SDXL
- SD3

### Workflow Example
The combined loader work in the same manner as existing checkpoints loader as seen in this workflow: 

![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/6b079ac4-1479-43e2-87f6-879919e34d0b)

Select the name of the diffusers checkpoint folder from the dropdown list and connect the nodes.

Take note for SD3:
- You have the option for the T5 encoder (text_encoder_3) to select either part_1, part_2, or all (both). 
- Selecting all will create a combined T5 encoder in the same folder named combined_text_encoder.safetensors for the first use.

## Limitations & Future Improvements
Future improvements is to update the seperate VAE, CLIP and UNET loaders.
Add support for AuraFlow and other compatible diffusers format checkpoints


## Contributing
Contributions are always welcome! Please follow the contributing guidelines to submit issues or pull requests.

## License
This project is licensed under the MIT License.

## Contact the Author
Hafiz Saffie - hafiz.safs@gmail.com

Project Link: https://github.com/Scorpinaus/ComfyUI-DiffusersLoader
