# parses file name that contains a timestamp and sets it as the metadata (mostly)
# run with: bazel run //:metadata_updater
import os
from datetime import datetime
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

def parse_filename_to_datetime(filename):
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    date_str = '24-12-12' #I set this manually because it was wrong
    time_str = parts[1]
    period_str = parts[2]
    datetime_str = f"{date_str} {time_str} {period_str}"
    return datetime.strptime(datetime_str, "%y-%m-%d %I%M%S %p")

def update_photos_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png','mp4')):
            file_path = os.path.join(directory_path, filename)
            try:
                new_timestamp = parse_filename_to_datetime(filename)
                update_exif_timestamp(file_path, new_timestamp)
                print(f"Updated {filename} with timestamp {new_timestamp}")
            except Exception as e:
                print(f"Failed to update {filename}: {e}")

# Update the directory path to your folder containing the photos
directory_path = '/Users/jgribble/Desktop/117GOPRO'
update_photos_in_directory(directory_path)
