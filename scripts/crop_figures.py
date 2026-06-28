import os
from PIL import Image, ImageChops

def trim_white_space(img_path):
    print(f"Processing {img_path}...")
    if not os.path.exists(img_path):
        print(f"Error: {img_path} does not exist.")
        return
    
    img = Image.open(img_path)
    # Convert to RGB if it is RGBA
    if img.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
        
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    if bbox:
        # Add a tiny padding of 10 pixels around the bounding box
        padding = 10
        w, h = img.size
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(w, bbox[2] + padding)
        bottom = min(h, bbox[3] + padding)
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(img_path)
        print(f"Trimmed and saved {img_path}. Old size: {img.size}, New size: {cropped_img.size}")
    else:
        print(f"No bounding box found for {img_path} (image might be entirely white).")

# Crop the images in Journal/figures/
trim_white_space("Journal/figures/system_architecture.png")
trim_white_space("Journal/figures/cnn_architecture.png")
trim_white_space("Journal/figures/tflite_conversion.png")
