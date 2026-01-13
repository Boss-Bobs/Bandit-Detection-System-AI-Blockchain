import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import collections
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, HTML
import shutil

# --- Configuration ---
# IMPORTANT: These MUST match your trained model
MODEL_PATH = "/kaggle/input/3dcae-model-project/tensorflow1/default/1/3DCAE_model_to_be_used.h5"
SEQUENCE_LENGTH = 16
RESIZE_DIM = (128, 128)
CHANNELS = 3
# A placeholder threshold. You MUST tune this value based on your specific video data.
ANOMALY_THRESHOLD = 0.015 
# Placeholder for your video file.
VIDEO_SOURCE = "/kaggle/input/bandit-dataset/VID-20250917-WA0008.mp4"
# New output path for the video file
OUTPUT_VIDEO_PATH = "ave2.mp4"
# Limit the number of frames to process for notebook display
MAX_FRAMES_TO_PROCESS = 450

# --- Main Logic ---
def notebook_visualizer():
    """
    Simulates a live CCTV feed, detects anomalies, and displays output in a notebook.
    """
    print("Loading trained autoencoder model...")
    try:
        model = load_model(MODEL_PATH, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model from {MODEL_PATH}. Please ensure it exists.")
        print(f"Error details: {e}")
        return

    print("Opening video source...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source {VIDEO_SOURCE}")
        return

    # Get video properties for the output video writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 format
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    all_reconstruction_errors = []

    print(f"\nStarting anomaly visualization. Processing frames and writing to {OUTPUT_VIDEO_PATH}...")

    try:
        while frame_count < MAX_FRAMES_TO_PROCESS:
            ret, frame = cap.read()
            if not ret:
                print("End of video or read error.")
                break

            processed_frame = cv2.resize(frame, RESIZE_DIM, interpolation=cv2.INTER_AREA)
            processed_frame = processed_frame / 255.0  # Normalize to [0, 1]
            
            sequence_buffer.append(processed_frame)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_sequence = np.expand_dims(np.array(list(sequence_buffer)), axis=0)
                reconstructed_sequence = model.predict(input_sequence, verbose=0)
                
                original_last_frame = input_sequence[0, -1]
                reconstructed_last_frame = reconstructed_sequence[0, -1]
                
                reconstruction_error = np.mean(np.square(original_last_frame - reconstructed_last_frame))
                all_reconstruction_errors.append(reconstruction_error)
                
                display_frame = frame.copy()
                text = f"Score: {reconstruction_error:.4f}"
                
                if reconstruction_error > ANOMALY_THRESHOLD:
                    text_color = (0, 0, 255)  # Red (BGR)
                    alert_text = "ANOMALY DETECTED!"
                    cv2.putText(display_frame, alert_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    text_color = (0, 255, 0)  # Green (BGR)

                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
                
                # Write the frame to the output video file
                video_writer.write(display_frame)

            frame_count += 1
            print(f"Processed frame {frame_count}/{MAX_FRAMES_TO_PROCESS}", end='\r')

        print("\nFinished processing frames. Displaying results...")

    finally:
        cap.release()
        video_writer.release()
        print(f"Video saved to {OUTPUT_VIDEO_PATH}")

    # --- Plot reconstruction error over time ---
    plt.figure(figsize=(12, 6))
    plt.plot(all_reconstruction_errors, label='Reconstruction Error')
    plt.axhline(y=ANOMALY_THRESHOLD, color='r', linestyle='--', label=f'Anomaly Threshold ({ANOMALY_THRESHOLD})')
    plt.xlabel('Frame')
    plt.ylabel('Error')
    plt.title('Reconstruction Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    notebook_visualizer()
