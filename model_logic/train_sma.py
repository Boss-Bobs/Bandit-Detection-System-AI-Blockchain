import numpy as np
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Input, MaxPooling3D, UpSampling3D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
from data_generator import VideoSequenceGenerator
from datetime import datetime

# --- Configuration ---
SEQUENCE_LENGTH = 16
RESIZE_DIM = (128, 128)
CHANNELS = 3
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 0.0001  # Default, will be optimized by SMA
MODEL_SAVE_DIR = "/kaggle/working/3DCAE/3DCAE_processed_data/model"
MODEL_FILENAME = "3DCAE_model_to_be_used.h5" 
LOG_DIR = "/kaggle/working/3DCAE/3DCAE_processed_data/logs"
TRAINING_VIDEO_PATHS_FILE = "/kaggle/working/3DCAE/3DCAE_processed_data/combined_training_video_paths.txt"
VALIDATION_VIDEO_PATHS_FILE = "/kaggle/working/3DCAE/3DCAE_processed_data/combined_validation_video_paths.txt"
TESTING_VIDEO_PATHS_FILE = "/kaggle/working/3DCAE/3DCAE_processed_data/combined_validation_video_paths.txt"
GROUND_TRUTH_DIR = "/kaggle/working/3DCAE/3DCAE_processed_data/ground_truth"
ANNOTATION_THRESHOLD = 0.0015

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Ground Truth Masks ---
def load_ground_truth_masks(gt_base_dir, test_sequence_folder_path, target_resize_dim):
    masks = []
    sequence_folder_name = os.path.basename(test_sequence_folder_path)
    gt_subfolder_name = sequence_folder_name + "_gt"
    gt_folder_path = os.path.join(gt_base_dir, gt_subfolder_name)
    if not os.path.exists(gt_folder_path):
        print(f"Ground truth folder {gt_folder_path} not found, skipping...")
        return None
    mask_filenames = sorted([f for f in os.listdir(gt_folder_path) if f.endswith(('.bmp', '.png', '.jpg'))])
    for filename in mask_filenames:
        mask_path = os.path.join(gt_base_dir, gt_subfolder_name, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8) * 255
        if mask.shape != target_resize_dim:
            mask = cv2.resize(mask, (target_resize_dim[1], target_resize_dim[0]), interpolation=cv2.INTER_NEAREST)
        if CHANNELS == 3:  # Convert grayscale mask to RGB
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        masks.append(mask)
    if not masks:
        print(f"No valid masks found in {gt_folder_path}, skipping...")
        return None
    return np.array(masks)

# --- Model Architecture ---
def build_autoencoder(input_shape, filters_1=64, kernel_size_1=3, filters_2=128, kernel_size_2=3):
    input_layer = Input(shape=input_shape)
    kernel_size_1 = (int(kernel_size_1), int(kernel_size_1), int(kernel_size_1))
    kernel_size_2 = (int(kernel_size_2), int(kernel_size_2), int(kernel_size_2))
    
    # Encoder
    x = Conv3D(filters=int(filters_1), kernel_size=kernel_size_1, activation='relu', padding='same')(input_layer)
    x = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)
    x = Conv3D(filters=int(filters_2), kernel_size=kernel_size_2, activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    encoded = Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = Conv3DTranspose(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D(size=(1, 2, 2))(x)
    decoded = Conv3DTranspose(filters=CHANNELS, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder

# --- Objective Function for SMA ---
def objective_function(params):
    lr, filters_1, kernel_size_1, filters_2, kernel_size_2 = params
    print(f"Evaluating params: lr={lr:.6f}, filters_1={int(filters_1)}, kernel_size_1={int(kernel_size_1)}, filters_2={int(filters_2)}, kernel_size_2={int(kernel_size_2)}")
    input_shape = (SEQUENCE_LENGTH, RESIZE_DIM[0], RESIZE_DIM[1], CHANNELS)
    model = build_autoencoder(input_shape, filters_1, kernel_size_1, filters_2, kernel_size_2)
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    
    train_generator = VideoSequenceGenerator(
        video_paths_file=TRAINING_VIDEO_PATHS_FILE,
        sequence_length=SEQUENCE_LENGTH,
        resize_dim=RESIZE_DIM,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    validation_generator = VideoSequenceGenerator(
        video_paths_file=VALIDATION_VIDEO_PATHS_FILE,
        sequence_length=SEQUENCE_LENGTH,
        resize_dim=RESIZE_DIM,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_generator = VideoSequenceGenerator(
        video_paths_file=TESTING_VIDEO_PATHS_FILE,
        sequence_length=SEQUENCE_LENGTH,
        resize_dim=RESIZE_DIM,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, f"temp_model_{os.getpid()}.h5"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=0
    )
    # Using patience=3 for quick evaluation within the objective function
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        mode='min',
        verbose=0
    )
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        mode='min',
        verbose=0
    )
    
    model.fit(
        train_generator,
        epochs=15, # Limited epochs for hyperparameter search
        validation_data=validation_generator,
        callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback],
        verbose=0
    )
    
    all_reconstruction_errors = []
    all_true_labels = []
    all_test_sequence_folders = test_generator._load_sequence_folder_paths(TESTING_VIDEO_PATHS_FILE)
    for sequence_folder_path in all_test_sequence_folders:
        original_frames_resized = test_generator._get_image_frames_from_folder(sequence_folder_path)
        if not original_frames_resized:
            continue
        gt_pixel_masks_for_sequence = load_ground_truth_masks(GROUND_TRUTH_DIR, sequence_folder_path, RESIZE_DIM)
        sequences_from_current_folder = []
        if len(original_frames_resized) >= SEQUENCE_LENGTH:
            for i in range(0, len(original_frames_resized) - SEQUENCE_LENGTH + 1):
                sequence = original_frames_resized[i: i + SEQUENCE_LENGTH]
                sequences_from_current_folder.append(np.array(sequence, dtype=np.float32))
        for seq_idx_in_folder, sequence in enumerate(sequences_from_current_folder):
            input_batch = np.expand_dims(sequence, axis=0)
            reconstructed_batch = model.predict(input_batch, verbose=0)
            reconstruction_error = np.mean(np.square(sequence - reconstructed_batch[0]))
            middle_frame_idx_in_sequence = SEQUENCE_LENGTH // 2
            global_frame_idx = seq_idx_in_folder + middle_frame_idx_in_sequence
            if gt_pixel_masks_for_sequence is not None and global_frame_idx < len(gt_pixel_masks_for_sequence):
                gt_segment_start = seq_idx_in_folder
                gt_segment_end = seq_idx_in_folder + SEQUENCE_LENGTH
                if gt_segment_end <= len(gt_pixel_masks_for_sequence):
                    sequence_gt_label = np.any(gt_pixel_masks_for_sequence[gt_segment_start: gt_segment_end] > 0).astype(int)
                    all_reconstruction_errors.append(reconstruction_error)
                    all_true_labels.append(sequence_gt_label)
    
    if all_reconstruction_errors and all_true_labels:
        all_reconstruction_errors = np.array(all_reconstruction_errors)
        all_true_labels = np.array(all_true_labels)
        all_true_labels = (all_true_labels > 0).astype(int)
        if len(np.unique(all_true_labels)) > 1:
            predictions = (all_reconstruction_errors > ANNOTATION_THRESHOLD).astype(int)
            f1 = f1_score(all_true_labels, predictions)
            # SMA minimizes, so return negative F1
            return -f1
    print("No valid ground truth data for F1 score, returning high loss...")
    return float('inf')

# --- Slime Mould Algorithm ---
def slime_mould_algorithm(objective_function, PopulationSize=6, MaxIters=3, dimensions=5, lb=[1e-5, 16, 3, 32, 3], ub=[1e-2, 128, 7, 256, 7], z=0.03):
    num_agents = PopulationSize
    max_iters = MaxIters
    # Set the seed for reproducibility
    np.random.seed(42) 
    random.seed(42)
    positions = np.random.uniform(lb, ub, size=(num_agents, dimensions))
    best_pos = None
    best_fitness = float('inf')

    for t in range(max_iters):
        # Calculate fitness for all positions
        # Note: This is computationally intensive as it trains a model per position per iteration
        fitness_value = np.array([objective_function(pos) for pos in positions])
        
        min_idx = np.argmin(fitness_value)
        if fitness_value[min_idx] < best_fitness:
            best_fitness = fitness_value[min_idx]
            best_pos = positions[min_idx]

        sorted_indices = np.argsort(fitness_value)
        positions = positions[sorted_indices]
        fitness_value = fitness_value[sorted_indices]
        X_b = positions[0] # Best position (leader)

        # Update weights (W) and search parameters (a, vb, vc)
        # Weight (W) calculation based on fitness difference
        # Avoid division by zero/very small number
        max_fit = np.max(fitness_value)
        min_fit = np.min(fitness_value)
        
        # Calculate W array for all agents (used in exploration/exploitation)
        # The equation for W in the original script is:
        # w = 1 + random.random() * np.log((fitness_value[0] - fitness_value[j]) / (np.max(fitness_value) - np.min(fitness_value) + 1e-10) + 1)
        # This seems to be done per agent inside the loop, so we'll keep it there.
        
        arctanh_input = np.clip(-(t / max_iters) + 1, -0.999, 0.999)
        a = np.arctanh(arctanh_input) # Global search parameter 'a'

        for j in range(num_agents):
            r = random.random()
            # Probability 'p' of transitioning from exploration to exploitation
            p = np.tanh(np.abs(fitness_value[j] - best_fitness)) 
            
            # Local search parameters
            vb = random.uniform(-a, a)
            vc = random.uniform(-(1 - t / max_iters), 1 - t / max_iters)

            # Weight calculation for the current agent
            if max_fit == min_fit:
                # Handle case where all fitness values are the same
                w = 1.0
            else:
                 # Original weight calculation from the script
                 w = 1 + random.random() * np.log((fitness_value[0] - fitness_value[j]) / (max_fit - min_fit + 1e-10) + 1)

            # Update position (Movement)
            if r < z: # Exploration (Random relocation)
                positions[j] = np.random.uniform(lb, ub, size=dimensions)
            elif r < p: # Exploitation (Movement towards best position influenced by others)
                idx_A, idx_B = random.sample(range(num_agents), 2)
                X_A, X_B = positions[idx_A], positions[idx_B]
                # Note: The original script uses X_A and X_B as randomly selected positions, 
                # which is common in SMA for simulating vein width influence.
                positions[j] = X_b + vb * (w * (X_A - X_B))
            else: # Exploitation (Local search / movement based on current position and 'vc')
                positions[j] = vc * positions[j]

            # Boundary check and constraint handling
            positions[j] = np.clip(positions[j], lb, ub)
            # Ensure odd kernel sizes for stability/consistency
            positions[j][2] = 2 * round(positions[j][2] / 2) + 1  # Ensure odd kernel_size_1
            positions[j][4] = 2 * round(positions[j][4] / 2) + 1  # Ensure odd kernel_size_2

        print(f"Iteration {t+1}/{max_iters} - Best Fitness (-F1): {best_fitness}")

    return best_pos, best_fitness

# --- Main Training Logic ---
if __name__ == '__main__':
    print("Running Slime Mould Algorithm for hyperparameter optimization...")
    lb = [1e-5, 16, 3, 32, 3]  # [lr, filters_1, kernel_size_1, filters_2, kernel_size_2]
    ub = [1e-2, 128, 7, 256, 7]
    best_params, best_fitness = slime_mould_algorithm(
        objective_function,
        PopulationSize=6,
        MaxIters=3,
        lb=lb,
        ub=ub
    )
    print(f"Optimal Hyperparameters: lr={best_params[0]:.6f}, filters_1={int(best_params[1])}, "
          f"kernel_size_1={int(best_params[2])}, filters_2={int(best_params[3])}, kernel_size_2={int(best_params[4])}")
    print(f"Best Objective (-F1): {best_fitness} (F1: {-best_fitness:.4f})")

    print("\nSetting up final model with optimized hyperparameters...")
    input_shape = (SEQUENCE_LENGTH, RESIZE_DIM[0], RESIZE_DIM[1], CHANNELS)
    model = build_autoencoder(best_params[1], best_params[2], best_params[3], best_params[4]) # Pass params to build_autoencoder
    
    optimizer = Adam(learning_rate=best_params[0])
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    # --- Data Generators ---
    train_generator = VideoSequenceGenerator(
        video_paths_file=TRAINING_VIDEO_PATHS_FILE,
        sequence_length=SEQUENCE_LENGTH,
        resize_dim=RESIZE_DIM,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    validation_generator = VideoSequenceGenerator(
        video_paths_file=VALIDATION_VIDEO_PATHS_FILE,
        sequence_length=SEQUENCE_LENGTH,
        resize_dim=RESIZE_DIM,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --- Callbacks ---
    model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    # INTEGRATED: EarlyStopping patience = 5
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5, 
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    # INTEGRATED: ReduceLROnPlateau min_lr = 0.000001
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.000001, 
        mode='min',
        verbose=1
    )
    # Ensure TensorBoard uses tf.keras.callbacks.TensorBoard for compatibility
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")))

    print("\nStarting final model training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback],
        verbose=1
    )

    print("\nTraining finished.")