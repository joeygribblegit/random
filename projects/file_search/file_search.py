import pytsk3
import sys

class Img_Info(pytsk3.Img_Info):
    def __init__(self, image):
        super(Img_Info, self).__init__(image)

def search_files(directory, fs_info, target_name):
    for entry in directory:
        file_name = entry.info.name.name.decode('utf-8')
        if file_name == target_name:
            print(f'Found {target_name} at inode {entry.info.meta.addr}')
        
        if entry.info.meta and entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
            sub_directory = fs_info.open_dir(inode=entry.info.meta.addr)
            search_files(sub_directory, fs_info, target_name)

# Path to your disk image
img_file = '/Users/jgribble/sdcard_backup.img'
offset = 8192  # Adjust this value if you determine a different starting offset

# Open the disk image
try:
    img_info = Img_Info(img_file)
    fs_info = pytsk3.FS_Info(img_info, offset=offset)
    print(fs_info)
    root_dir = fs_info.open_dir(path='/')
    search_files(root_dir, fs_info, 'ZOOM0002_Tr1.WAV.txt')
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
