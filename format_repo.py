import os
import shutil
from pathlib import Path

def reorganize_dataset(source_dir, target_dir):
    # Create target directories
    Path(target_dir + "/depth").mkdir(parents=True, exist_ok=True)
    Path(target_dir + "/rgb").mkdir(parents=True, exist_ok=True)
    Path(target_dir + "/mask").mkdir(parents=True, exist_ok=True)
    Path(target_dir + "/loc").mkdir(parents=True, exist_ok=True)

    # Track frame numbers across all videos
    frame_counter = 1
    processed_pairs = {}

    # Sort files to maintain temporal order
    for file in sorted(os.listdir(source_dir)):
        if not file.endswith(('.png', '.txt')):
            continue

        parts = file.split('_')
        if len(parts) < 4:
            continue  # Skip invalid filenames

        # Extract components
        category, instance, video, frame = parts[0], parts[1], parts[2], parts[3].split('.')[0]
        unique_id = f"{video}_{frame}"

        # Add a check for depthmask before depth
        if "_depthmask" in file:
            file_type = "mask"
            suffix = "_depthmask"
        elif "_depth" in file:
            file_type = "depth"
            suffix = "_depth"
        elif "_mask" in file:
            file_type = "mask"
            suffix = "_mask"
        elif "_loc" in file:
            file_type = "loc"
            suffix = ""  # No suffix for localization files
        else:
            file_type = "rgb"
            suffix = ""

        # Maintain consistent numbering across modalities
        if unique_id not in processed_pairs:
            processed_pairs[unique_id] = frame_counter
            frame_counter += 1

        # Construct new filename
        if file_type == "loc":
            new_name = f"align_loc_{processed_pairs[unique_id]}.txt"
        else:
            new_name = f"align_test{suffix}_{processed_pairs[unique_id]}.png"

        # Set target paths
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(target_dir, file_type, new_name)

        # Copy with progress feedback
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {file} → {dst_path}")

if __name__ == "__main__":
    source_directory = r"D:\College\Sem_6\Computer Vision\Project\apple_2\rgbd-dataset\apple\apple_2"  # Update this path
    target_directory = "train/apple_2"  # Update this path
    reorganize_dataset(source_directory, target_directory)
