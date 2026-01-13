!pip install tensorflow opencv-python numpy matplotlib tqdm

import os
import random

# --- Configuration for both datasets ---

# Avenue Dataset Paths
AVENUE_TRAIN_VIDEOS = "/kaggle/input/avenuedataset/Avenuedataset/Avenue_Dataset/Avenue Dataset/training_videos"
AVENUE_TEST_VIDEOS = "/kaggle/input/avenuedataset/Avenuedataset/Avenue_Dataset/Avenue Dataset/testing_videos"

# FUTD (FUTMINNA) Dataset Paths
FUTD_TRAIN_VIDEOS = "/kaggle/input/futminna-dataset/training_videos/training_videos"
FUTD_TEST_VIDEOS = "/kaggle/input/futminna-dataset/testing_videos/testing_videos"

ARMY_VIDEOS = "/kaggle/input/armybandit-dataset"
BANDIT_TEST_VIDEOS = "/kaggle/input/bandit-dataset"

# Output Paths for Combined Datasets
OUTPUT_DIR = "/kaggle/working/3DCAE/3DCAE_processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAINING_VIDEO_PATHS_FILE = os.path.join(OUTPUT_DIR, "combined_training_video_paths.txt")
TESTING_VIDEO_PATHS_FILE = os.path.join(OUTPUT_DIR, "combined_testing_video_paths.txt")
VALIDATION_VIDEO_PATHS_FILE = os.path.join(OUTPUT_DIR, "combined_validation_video_paths.txt")

# --- Helper Function to Get Video Paths ---
def get_video_paths_from_dir(base_dir):
    """
    Scans a directory and returns a list of video file paths.
    """
    video_paths = []
    # Check if the directory exists and is not empty
    if not os.path.exists(base_dir):
        print(f"Warning: Directory not found: {base_dir}")
        return []

    # Get all video files (assuming mp4, avi, etc.)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(root, file))
    return video_paths

# --- Main Script to Generate Combined Path Files ---
def generate_combined_paths():
    """
    Combines video paths from multiple datasets and splits them into
    training, validation, and testing sets.
    """
    print("Combining Avenue and FUTD training video paths...")
    avenue_train_paths = get_video_paths_from_dir(AVENUE_TRAIN_VIDEOS)
    futd_train_paths = get_video_paths_from_dir(FUTD_TRAIN_VIDEOS)
    army_train_paths = get_video_paths_from_dir(ARMY_VIDEOS)
    
    # Simple aggregation for training
    all_training_paths = avenue_train_paths + futd_train_paths + army_train_paths
    random.shuffle(all_training_paths) # Shuffle to mix the two datasets
    
    # Split training paths into a smaller validation set (e.g., 20%)
    validation_split_index = int(len(all_training_paths) * 0.2)
    training_paths = all_training_paths[validation_split_index:]
    validation_paths = all_training_paths[:validation_split_index]
    
    print(f"Total training videos: {len(training_paths)}")
    print(f"Total validation videos: {len(validation_paths)}")

    # Combine testing paths
    print("\nCombining Avenue, army and FUTD testing video paths...")
    avenue_test_paths = get_video_paths_from_dir(AVENUE_TEST_VIDEOS)
    futd_test_paths = get_video_paths_from_dir(FUTD_TEST_VIDEOS)
    bandit_test_path = get_video_paths_from_dir(BANDIT_TEST_VIDEOS)
    
    all_testing_paths = avenue_test_paths + futd_test_paths + bandit_test_path
    print(f"Total testing videos: {len(all_testing_paths)}")

    # Write paths to files
    with open(TRAINING_VIDEO_PATHS_FILE, 'w') as f:
        for path in training_paths:
            f.write(f"{path}\n")

    with open(VALIDATION_VIDEO_PATHS_FILE, 'w') as f:
        for path in validation_paths:
            f.write(f"{path}\n")
    
    with open(TESTING_VIDEO_PATHS_FILE, 'w') as f:
        for path in all_testing_paths:
            f.write(f"{path}\n")

    print(f"\nSuccessfully generated combined path files in {OUTPUT_DIR}:")
    print(f"- {os.path.basename(TRAINING_VIDEO_PATHS_FILE)}")
    print(f"- {os.path.basename(VALIDATION_VIDEO_PATHS_FILE)}")
    print(f"- {os.path.basename(TESTING_VIDEO_PATHS_FILE)}")

if __name__ == '__main__':
    generate_combined_paths()