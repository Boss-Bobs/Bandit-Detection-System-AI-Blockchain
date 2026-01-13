%%writefile /kaggle/working/3DCAE/ave_gt_conv.py
# Cell for .mat to .npy conversion (run this!)
# Cell for .mat to .npy conversion (run this!)
import os
import scipy.io
import numpy as np
from tqdm.notebook import tqdm

print("Starting conversion of .mat ground truth files to .npy...")

INPUT_GT_DIR = "/kaggle/input/avenuedataset/Avenuedataset/ground_truth_demo/ground_truth_demo/testing_label_mask"
OUTPUT_PROCESSED_DATA_DIR = "/kaggle/working/3DCAE/3DCAE_processed_data"
OUTPUT_GT_DIR = os.path.join(OUTPUT_PROCESSED_DATA_DIR, "combined_testing_label_mask_npy")
os.makedirs(OUTPUT_GT_DIR, exist_ok=True)

mat_files = [f for f in os.listdir(INPUT_GT_DIR) if f.endswith(".mat")]

if not mat_files:
    print(f"No .mat files found in {INPUT_GT_DIR}. Please check the ground truth path.")
else:
    for mat_file in tqdm(mat_files, desc="Converting .mat to .npy"):
        mat_path = os.path.join(INPUT_GT_DIR, mat_file)
        
        try:
            mat_data = scipy.io.loadmat(mat_path)
            
            if 'volLabel' in mat_data: 
                mask_data_raw = mat_data['volLabel'] 
                
                frame_level_anomaly = None
                
                if mask_data_raw.ndim == 2 and mask_data_raw.shape[0] == 1 and mask_data_raw.dtype == object:
                    list_of_frame_masks = mask_data_raw[0]
                    frame_level_anomaly = np.array([1 if np.any(frame_mask > 0) else 0 for frame_mask in list_of_frame_masks]).astype(int)
                elif mask_data_raw.ndim == 1:
                    frame_level_anomaly = (mask_data_raw.astype(float) > 0).astype(int)
                elif mask_data_raw.ndim == 3:
                    frame_level_anomaly = np.array([1 if np.any(mask_data_raw[:, :, i].astype(float) > 0) else 0 for i in range(mask_data_raw.shape[2])])
                else:
                    print(f"Warning: Unexpected 'volLabel' shape or dtype in {mat_file}: {mask_data_raw.shape}, {mask_data_raw.dtype}. Skipping conversion.")
                    continue 
                    
                if frame_level_anomaly is not None:
                    # Fix: Correctly format the filename with a leading zero
                    base_name = os.path.splitext(mat_file)[0]
                    try:
                        video_id = int(base_name.split('_')[0])
                        npy_filename = f"{video_id:02d}_label.npy"
                    except (ValueError, IndexError):
                        # Fallback for unexpected naming, though it's less likely.
                        npy_filename = f"{base_name}_label.npy"
                    
                    np.save(os.path.join(OUTPUT_GT_DIR, npy_filename), frame_level_anomaly)
                else:
                    print(f"Warning: Could not process 'volLabel' for {mat_file} after shape handling. Skipping conversion.")

            else:
                print(f"Warning: 'volLabel' key not found in {mat_file}. Skipping conversion.")

        except Exception as e:
            print(f"ERROR: Failed to convert {mat_file}. Error: {e}")

print("Conversion complete.")
print(f"Converted .npy files are saved to: {OUTPUT_GT_DIR}")
