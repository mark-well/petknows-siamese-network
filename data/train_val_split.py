import os
import shutil
import random


def move_validation_images(
    root_dir="data",
    val_dir="data_val",
    images_per_class=2,
    seed=42
):
    """
    Moves `images_per_class` images from each subfolder (class) in root_dir
    into a mirrored subfolder structure under val_dir.

    Example:
        data/pet1/0.jpg, 1.jpg, ...  ->  data_val/pet1/0.jpg, 1.jpg
        data/pet2/0.jpg, 1.jpg, ...  ->  data_val/pet2/0.jpg, 1.jpg

    Args:
        root_dir: path to the training data root (contains one subfolder per class)
        val_dir: path where validation data will be moved to
        images_per_class: how many images to move out of each class folder
        seed: for reproducible random selection
    """
    random.seed(seed)
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    class_folders = [
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ]

    if not class_folders:
        print(f"No subfolders found in {root_dir}")
        return

    total_moved = 0

    for class_name in class_folders:
        src_folder = os.path.join(root_dir, class_name)
        dst_folder = os.path.join(val_dir, class_name)

        images = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith(valid_extensions)
        ]

        if len(images) < images_per_class:
            print(
                f"Skipping '{class_name}': only {len(images)} image(s) available, "
                f"need {images_per_class}."
            )
            continue

        selected = random.sample(images, images_per_class)

        os.makedirs(dst_folder, exist_ok=True)

        for filename in selected:
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(dst_folder, filename)
            shutil.move(src_path, dst_path)
            total_moved += 1
            print(f"Moved: {src_path} -> {dst_path}")

    print(f"\nDone. Moved {total_moved} image(s) across {len(class_folders)} class folder(s).")


if __name__ == "__main__":
    move_validation_images(
        root_dir="data/processed",
        val_dir="data/data_val",
        images_per_class=2
    )