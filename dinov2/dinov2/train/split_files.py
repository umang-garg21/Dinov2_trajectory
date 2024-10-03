import os
import shutil
import random

def split_data(source_dir, output_dir, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    # Ensure the ratios add up to 1.0
    print("train ratio+ test ratio + val ratio", train_ratio+ test_ratio+val_ratio)
    assert round(train_ratio + test_ratio + val_ratio) == 1.0, "Ratios must sum to 1.0"
    
    # Create output directories if they don't exist
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all the files from the source directory
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(all_files)  # Shuffle the list to ensure randomness

    # Split the files according to the specified ratios
    train_split = int(train_ratio * len(all_files))
    test_split = int(test_ratio * len(all_files)) + train_split
    
    train_files = all_files[:train_split]
    test_files = all_files[train_split:test_split]
    val_files = all_files[test_split:]

    # Function to move files to the corresponding directory
    def move_files(file_list, target_dir):
        for file_name in file_list:
            shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

    # Move files to respective directories
    move_files(train_files, train_dir)
    move_files(test_files, test_dir)
    move_files(val_files, val_dir)

    print(f"Moved {len(train_files)} files to {train_dir}")
    print(f"Moved {len(test_files)} files to {test_dir}")
    print(f"Moved {len(val_files)} files to {val_dir}")

if __name__ == "__main__":
    # Source directory containing all the files

    source_directory = "/data/home/umang/Vader_data/haystac_dinov2/p1tj_traj_npy"   # Replace with your source folder
    # Output directory where train, test, and val folders will be created
    output_directory = "/data/home/umang/Vader_data/haystac_dinov2/p1tj_traj_npy" # Replace with your desired output folder

    split_data(source_directory, output_directory)
