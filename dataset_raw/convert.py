import os
import subprocess
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()  # lets Pillow open .heic/.heif files natively

FFMPEG_EXTENSIONS = {".png", ".bmp", ".gif", ".tiff", ".tif", ".webp", ".jfif"}
HEIC_EXTENSIONS = {".heic", ".heif"}


def convert_with_ffmpeg(old_path, new_path, quality=95):
    cmd = [
        "ffmpeg",
        "-y",              # overwrite without prompt
        "-i", old_path,    # input file
        "-qscale:v", str(round((100 - quality) / 4)),  # rough quality mapping
        new_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return True, None
    return False, result.stderr.decode(errors="ignore")


def convert_with_pillow_heif(old_path, new_path, quality=95):
    try:
        img = Image.open(old_path)
        img.convert("RGB").save(new_path, "JPEG", quality=quality)
        return True, None
    except Exception as e:
        return False, str(e)


def convert_to_jpg(root_directory, quality=95, delete_original=True):
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            ext_lower = ext.lower()

            if ext_lower not in FFMPEG_EXTENSIONS and ext_lower not in HEIC_EXTENSIONS:
                continue

            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, name + ".jpg")

            # avoid overwriting an existing jpg with the same name
            if os.path.exists(new_path):
                print(f"Skipped (already exists): {new_path}")
                continue

            if ext_lower in HEIC_EXTENSIONS:
                success, error = convert_with_pillow_heif(old_path, new_path, quality)
            else:
                success, error = convert_with_ffmpeg(old_path, new_path, quality)

            if success:
                print(f"Converted: {old_path} -> {new_path}")
                if delete_original:
                    os.remove(old_path)
            else:
                print(f"Failed: {old_path}")
                print(error)


if __name__ == "__main__":
    directory = input("Enter the root dataset directory: ").strip()
    convert_to_jpg(directory)