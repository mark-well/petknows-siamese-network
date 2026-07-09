import os

def rename_folders(parent_directory, prefix):
    for folder_name in os.listdir(parent_directory):
        old_path = os.path.join(parent_directory, folder_name)
        
        if os.path.isdir(old_path):
            new_name = prefix + "_" + folder_name.lower().replace(" ", "_")
            new_path = os.path.join(parent_directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: '{folder_name}' -> '{new_name}'")

def rename_files(directory, prefix="ava"):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()  # ensures consistent ordering
    
    for index, filename in enumerate(files, start=1):
        name, ext = os.path.splitext(filename)
        new_name = f"{prefix}_{index:02d}{ext}"
        
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: '{filename}' -> '{new_name}'")

if __name__ == "__main__":
    mode = int(input("Select a mode: [1]. Rename folder, [2]. Rename file: "))

    if mode == 1:
        directory = input("Enter the directory path: ").strip()
        prefix = input("Enter prefix (dog or cat): ").strip()
        rename_folders(directory, prefix)
        
    elif mode == 2:
        directory = input("Enter the directory path: ").strip()
        prefix = input("Enter prefix (default 'ava'): ").strip() or "ava"
        rename_files(directory, prefix)
    else:
        print("Invalid Mode!")