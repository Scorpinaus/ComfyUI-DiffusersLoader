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
1. Clone the repository
git clone https://github.com/Scorpinaus/ComfyUI-DiffusersLoader.git

2. Navigate to the project directory & add-it to ComfyUI manager under custom nodes folder

3. Add nodes to your workflow

## Features
There are 2 main areas of this node package: SD1.5 Loaders and SDXL Loaders.

### Set-up

For the nodes to work, you will need to create SD15 and SDXL sub-directories under the diffusers folder as seen below:
![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/b6d779da-5481-4666-ba24-faf2063ee76d)  ![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/7f71b2bd-172b-4380-9653-a1a2d0a09799)

Put your converted diffusers model into either SD15 for SD1.5 models or SDXL for SDXL models.

### Stable Diffusion 1.5 Loaders

There are 4 main nodes:
![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/93e8627d-4827-45cc-af98-3f9182133339)


### Stable Diffusion XL Loaders

There are 4 main nodes:

![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/0d3121b6-ba7f-47fb-94a3-018c656598fe)

### Workflow Example
The combined loader work in the same manner as existing checkpoints loader as seen in this workflow: 
![image](https://github.com/Scorpinaus/ComfyUI-DiffusersLoader/assets/85672737/6b079ac4-1479-43e2-87f6-879919e34d0b)

## Limitations & Future Improvements
The combined loader and CLIP loader does not work for inpainting models.


## Contributing
Contributions are always welcome! Please follow the contributing guidelines to submit issues or pull requests.

## License
This project is licensed under the MIT License.

## Contact

Hafiz Saffie - hafiz.safs@gmail.com
Project Link: https://github.com/Scorpinaus/ComfyUI-DiffusersLoader
