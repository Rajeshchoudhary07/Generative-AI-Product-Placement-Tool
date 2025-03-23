# Generative-AI-Product-Placement-Tool
## Overview

This project is an open-source AI-powered tool designed to automatically place e-commerce product images into realistic lifestyle backgrounds. The tool uses a combination of background removal, Stable Diffusion inpainting, and batch processing to seamlessly integrate products into various scenes.

## Features

Batch Processing: Processes multiple product images at once.

AI Background Removal: Uses rembg to extract product images.

Stable Diffusion Inpainting: Places products in backgrounds with realistic lighting and perspective.

Scalability: Supports multiprocessing for efficient execution.

Customization: Allows adjustments to product placement, size, and rotation.

Open-Source and Free: Uses only free and unlimited AI models and libraries.

## Technical Details

1. Background Removal

We utilize rembg, an open-source background removal tool based on U^2-Net, to isolate product images from their backgrounds.

2. Image Placement with AI

Stable Diffusion's runwayml/stable-diffusion-inpainting model is used to blend the product image seamlessly into the lifestyle image while maintaining realistic lighting and shadows.

3. Batch Processing & Parallelization

We leverage Python's multiprocessing to process multiple images in parallel, reducing overall runtime and making the tool efficient for large-scale e-commerce applications.

## Installation

Prerequisites

Python 3.8+

A machine with a CUDA-enabled GPU (optional but recommended for faster processing)

## Install Dependencies

Run the following command to install required packages:

pip install torch torchvision diffusers rembg opencv-python pillow tqdm

## Usage

Place your product images (PNG format) in the input_products/ folder.

Place your lifestyle background images (JPG format) in the input_backgrounds/ folder.

Run the script:

python main.py

Processed images will be saved in the output/ folder.

## folder Structure

├── input_products/        # Product images (without background)
├── input_backgrounds/     # Lifestyle background images
├── output/                # Processed images
├── temp/                  # Temporary processing folder
├── main.py                # Main execution script
├── README.md              # Project documentation

## Performance Considerations

For best results, use high-resolution product images with clean edges.

A CUDA-enabled GPU will significantly speed up processing.

Avoid using extremely large background images to prevent memory issues.

## Future Enhancements

Adding an interactive web-based UI for easy image uploads.

Implementing automatic perspective correction for better realism.

Enhancing product placement using 3D scene understanding models.


## Acknowledgments

rembg for background removal

Stable Diffusion for AI-powered image generation

Open-source AI communities for making such advancements accessible to all.

Contact

For questions or contributions, open an issue or submit a pull request on GitHub.

