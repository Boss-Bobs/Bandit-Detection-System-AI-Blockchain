import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_generator import VideoSequenceGenerator
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# --- Configuration ---
# IMPORTANT: These MUST match what you successfully used for training in train.py and data_generator.py
SEQUENCE_LENGTH = 16 # <--- Matches the trained model's sequence length
RESIZE_DIM = (128, 128) # <--- Matches the trained model's resize dimensions
CHANNELS = 3
BATCH_SIZE = 2 # Use a batch size of 1 for testing to process one sequence at a time for anomaly scoring

# --- Paths ---
MODEL_PATH = "/kaggle/input/3dcae-model-project/tensorflow1/default/1/3DCAE_model_to_be_used.h5"
TESTING_VIDEO_PATHS_FILE = "/kaggle/working/3DCAE/3DCAE_processed_data/combined_testing_video_paths.txt"
AVENUE_GROUND_TRUTH_VOL = "/kaggle/working/3DCAE/3DCAE_processed_data/combined_testing_label_mask_npy" 
RESULTS_DIR = "/kaggle/working/3DCAE/3DCAE_processed_data/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Function to load ground truth ---
def load_ground_truth(ground_truth_path, video_path):
    video_id_str = os.path.basename(video_path).split('.')[0]
    video_id_for_gt = str(int(video_id_str)) 
    ground_truth_mask_file = os.path.join(ground_truth_path, f"{video_id_for_gt}_label.npy") 
    
    try:
        gt_labels_for_frames = np.load(ground_truth_mask_file)
        return gt_labels_for_frames
    except FileNotFoundError:
        print(f"Warning (GT File Not Found): {ground_truth_mask_file}. Video: {video_path}. Skipping for AUC.")
        return None 
    except Exception as e:
        print(f"Warning (GT Loading Error): {ground_truth_mask_file}. Error: {e}. Video: {video_path}. Skipping for AUC.")
        return None

# --- Main Evaluation Logic ---
if __name__ == '__main__':
    print("Loading trained model...")
    try:
        model = load_model(MODEL_PATH, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model from {MODEL_PATH}. Error: {e}")
        print("Please ensure training completed and the model file exists.")
        exit(1)

    print("Setting up test data generator...")
    test_generator = VideoSequenceGenerator(
        video_paths_file=TESTING_VIDEO_PATHS_FILE,
        sequence_length=SEQUENCE_LENGTH, # <--- ENSURE THIS MATCHES
        resize_dim=RESIZE_DIM, # <--- ENSURE THIS MATCHES
        batch_size=BATCH_SIZE, 
        shuffle=False 
    )
    
    video_paths_for_testing = test_generator._load_video_paths(TESTING_VIDEO_PATHS_FILE)
    print(f"DEBUG TEST: Total test video paths loaded: {len(video_paths_for_testing)}")
    if len(video_paths_for_testing) > 0:
        print(f"DEBUG TEST: First test video path: {video_paths_for_testing[0]}")
    else:
        print("DEBUG TEST: No test video paths found in the list.")


    all_reconstruction_errors = []
    all_true_labels = [] 

    print("\nStarting evaluation on test videos...")
    
    if not video_paths_for_testing:
        print("ERROR: No test video paths available to evaluate. Exiting test script.")
        exit(1)

    for video_path in tqdm(video_paths_for_testing, desc="Evaluating Test Videos"):
        gt_labels_for_frames = load_ground_truth(AVENUE_GROUND_TRUTH_VOL, video_path)
        
        sequences_from_current_video = test_generator._get_video_sequences(video_path)
        
        if not sequences_from_current_video:
            continue

        video_sequence_errors = []
        for sequence in sequences_from_current_video:
            input_batch = np.expand_dims(sequence, axis=0)
            reconstructed_batch = model.predict(input_batch, verbose=0)
            
            reconstruction_error = np.mean(np.square(sequence - reconstructed_batch[0]))
            video_sequence_errors.append(reconstruction_error)

        if gt_labels_for_frames is not None and len(gt_labels_for_frames) >= SEQUENCE_LENGTH:
            gt_labels_for_sequences = []
            for i in range(len(video_sequence_errors)): 
                if (i + SEQUENCE_LENGTH) <= len(gt_labels_for_frames):
                    if np.any(gt_labels_for_frames[i : i + SEQUENCE_LENGTH] > 0):
                        gt_labels_for_sequences.append(1)
                    else:
                        gt_labels_for_sequences.append(0)
                else:
                    gt_labels_for_sequences.append(0)
            
            if len(gt_labels_for_sequences) == len(video_sequence_errors):
                all_reconstruction_errors.extend(video_sequence_errors)
                all_true_labels.extend(gt_labels_for_sequences)
            else:
                print(f"Warning: Mismatch between sequences ({len(video_sequence_errors)}) and GT labels ({len(gt_labels_for_sequences)}) for {video_path}. Skipping for AUC.")
        else:
            print(f"Warning: No valid ground truth to match sequences for {video_path}. Skipping for AUC.")

    if not all_reconstruction_errors or not all_true_labels:
        print("No valid test sequences with matching ground truth were processed. Cannot calculate AUC metrics.")
    else:
        all_reconstruction_errors = np.array(all_reconstruction_errors)
        all_true_labels = np.array(all_true_labels)
        
        all_true_labels = (all_true_labels > 0).astype(int)
        
        # --- Save results to .npy files ---
        results_save_path = os.path.join(RESULTS_DIR, "reconstruction_errors.npy")
        labels_save_path = os.path.join(RESULTS_DIR, "true_labels.npy")
        np.save(results_save_path, all_reconstruction_errors)
        np.save(labels_save_path, all_true_labels)
        print(f"Reconstruction errors saved to: {results_save_path}")
        print(f"True labels saved to: {labels_save_path}")

        if len(np.unique(all_true_labels)) > 1: 
            fpr, tpr, thresholds = roc_curve(all_true_labels, all_reconstruction_errors)
            roc_auc = auc(fpr, tpr)

            print(f"\nEvaluation Complete.")
            print(f"Total sequences processed: {len(all_reconstruction_errors)}")
            print(f"Total anomalous sequences: {np.sum(all_true_labels)}")
            print(f"Anomaly Detection AUC: {roc_auc:.4f}")

            # --- Metrics at a fixed threshold (now for reference, will be interactive in notebook) ---
            # FIXED_THRESHOLD = 0.015 # This is just a placeholder here, actual tuning done interactively
            # y_pred = (all_reconstruction_errors > FIXED_THRESHOLD).astype(int)
            # acc = accuracy_score(all_true_labels, y_pred, zero_division=0) # zero_division handles cases where precision/recall is 0
            # prec = precision_score(all_true_labels, y_pred, zero_division=0)
            # rec = recall_score(all_true_labels, y_pred, zero_division=0)
            # f1 = f1_score(all_true_labels, y_pred, zero_division=0)
            # print(f"\n(Example) Metrics at Threshold = {FIXED_THRESHOLD:.4f}:")
            # print(f"  Accuracy:  {acc:.4f}")
            # print(f"  Precision: {prec:.4f}")
            # print(f"  Recall:    {rec:.4f}")
            # print(f"  F1-Score:  {f1:.4f}")

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
            plt.show()
            print(f"ROC curve saved to {os.path.join(RESULTS_DIR, 'roc_curve.png')}")
        else:
            print("Warning: Only one class (normal or anomalous) found in true labels. Cannot calculate full metrics.")
            print(f"Total sequences processed: {len(all_reconstruction_errors)}")
            print(f"Total anomalous sequences: {np.sum(all_true_labels)}")
