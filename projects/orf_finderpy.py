import os
import shutil
from glob import glob

def find_matching_orf_files(input_jpg_dir, orf_dir, output_orf_dir):
    # Ensure output directory exists
    os.makedirs(output_orf_dir, exist_ok=True)

    # Get all JPG files from the input directory
    jpg_files = glob(os.path.join(input_jpg_dir, "*.JPG"))

    # Extract the base names of JPG files (without extensions)
    jpg_basenames = {os.path.splitext(os.path.basename(jpg_file))[0] for jpg_file in jpg_files}
    print(f"jpg_basenames: {jpg_basenames}")
    # Get all ORF files from the ORF directory
    orf_files = glob(os.path.join(orf_dir, "*.ORF"))
    print(f"orf_files: {orf_files}")
    # Iterate through the ORF files and copy matching ones
    for orf_file in orf_files:
        orf_basename = os.path.splitext(os.path.basename(orf_file))[0]
        if orf_basename in jpg_basenames:
            shutil.copy(orf_file, output_orf_dir)
            print(f"Copied: {orf_file} to {output_orf_dir}")

if __name__ == "__main__":
    # Input directories
    input_jpg_dir = "/Users/jgribble/Documents/keep"
    orf_dir = "/Users/jgribble/Desktop/100OLYMP"

    # Output directory for matching ORF files
    output_orf_dir = input_jpg_dir

    # Find and copy matching ORF files
    find_matching_orf_files(input_jpg_dir, orf_dir, output_orf_dir)
