import os
import cv2
import torch
import numpy as np
import multiprocessing
from rembg import remove
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

def setup_directories():
    """Creates required directories if they do not exist."""
    directories = ["input_products", "input_backgrounds", "output", "temp"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def remove_bg(input_path, output_path):
    """Removes background from product image."""
    with open(input_path, "rb") as f:
        input_img = f.read()
    output_img = remove(input_img)
    with open(output_path, "wb") as f:
        f.write(output_img)

def load_sd_pipeline():
    """Loads Stable Diffusion inpainting pipeline."""
    model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def blend_images(product_path, background_path, output_path, pipe):
    """Blends the product image into the lifestyle image."""
    try:
        product = Image.open(product_path).convert("RGBA")
        background = Image.open(background_path).convert("RGBA")
        
        product = product.resize((background.width // 3, background.height // 3), Image.ANTIALIAS)
        x_offset = (background.width - product.width) // 2
        y_offset = (background.height - product.height) // 2
        
        mask = product.split()[-1]
        result = pipe(prompt="Realistic product placement", image=background, mask_image=mask).images[0]
        result.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error blending images: {e}")

def process_image(product_file, bg_files, pipe):
    """Processes a single product image against multiple backgrounds."""
    try:
        product_path = os.path.join("input_products", product_file)
        bg_removed_path = os.path.join("temp", f"clean_{product_file}")
        remove_bg(product_path, bg_removed_path)
        
        for bg_file in bg_files:
            bg_path = os.path.join("input_backgrounds", bg_file)
            output_path = os.path.join("output", f"final_{product_file}_{bg_file}.png")
            blend_images(bg_removed_path, bg_path, output_path, pipe)
    except Exception as e:
        print(f"Error processing {product_file}: {e}")

def process_batch():
    """Processes all images in a batch using multiprocessing."""
    setup_directories()
    pipe = load_sd_pipeline()
    
    product_files = [f for f in os.listdir("input_products") if f.endswith(".png")]
    background_files = [f for f in os.listdir("input_backgrounds") if f.endswith(".jpg")]
    
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.starmap(process_image, [(p, background_files, pipe) for p in product_files])

def main():
    """Main function to execute the batch processing."""
    print("Starting batch processing...")
    process_batch()
    print("Processing complete. Check the output folder.")

if __name__ == "__main__":
    main()
