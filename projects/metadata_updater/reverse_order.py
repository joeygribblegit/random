# takes a folder with files like file-1.jpg
# sets a custom metadata date, and from largest to smallest, it increments the timestamps
# run with: bazel run //:time_reverser

import os
from datetime import datetime, timedelta
from PIL import Image
import piexif

def update_exif_timestamp(image_path, new_timestamp):
    image = Image.open(image_path)
    try:
        exif_dict = piexif.load(image.info.get("exif", b''))
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
    
    new_timestamp_str = new_timestamp.strftime("%Y:%m:%d %H:%M:%S")
    
    exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = new_timestamp_str.encode('utf-8')
    exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = new_timestamp_str.encode('utf-8')
    exif_dict['0th'][piexif.ImageIFD.DateTime] = new_timestamp_str.encode('utf-8')
    
    exif_bytes = piexif.dump(exif_dict)
    image.save(image_path, "jpeg", exif=exif_bytes, quality=90, subsampling=0)

def extract_file_number(filename):
    base_name = os.path.splitext(filename)[0]
    file_number = int(base_name.split('-')[1])
    return file_number

def update_photos_in_directory(directory_path, start_date):
    file_list = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sorted_file_list = sorted(file_list, key=extract_file_number, reverse=True)
    
    current_timestamp = start_date
    for filename in sorted_file_list:
        file_path = os.path.join(directory_path, filename)
        try:
            update_exif_timestamp(file_path, current_timestamp)
            print(f"Updated {filename} with timestamp {current_timestamp}")
            current_timestamp += timedelta(seconds=1)
        except Exception as e:
            print(f"Failed to update {filename}: {e}")

# Update the directory path to your folder containing the photos
directory_path = '/Users/jgribble/Downloads/album-d406011207-downloads'
start_date = datetime(2024, 6, 1, 0, 0, 0)

update_photos_in_directory(directory_path, start_date)
