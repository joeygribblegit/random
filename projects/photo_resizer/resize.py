from PIL import Image
import os
from tqdm import tqdm


# Input and output directories
input_folder = "/Users/jgribble/Desktop/top_dec13_noon/"
output_folder = "/Users/jgribble/Desktop/top_dec13_noon/smaller/"
os.makedirs(output_folder, exist_ok=True)

target_size_kb = 200
target_size_bytes = target_size_kb * 1024

# Get all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg'))]

# Add a progress bar for the processing loop
with tqdm(total=len(image_files), desc="Compressing Images", unit="image") as pbar:
    for filename in image_files:
        filepath = os.path.join(input_folder, filename)
        with Image.open(filepath) as img:
            # Reduce resolution
            img = img.resize((img.width // 2, img.height // 2))

            # Compress image
            output_path = os.path.join(output_folder, filename)
            for quality in range(85, 10, -5):  # Gradually reduce quality
                img.save(output_path, "JPEG", quality=quality)
                if os.path.getsize(output_path) <= target_size_bytes:
                    break
        
        # Update the progress bar
        pbar.update(1)

print("Compression complete!")