import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Input, MaxPooling3D, UpSampling3D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from data_generator import VideoSequenceGenerator # Import our custom generator
from datetime import datetime

# --- Configuration ---
# Match these with what was used in data_generator.py
SEQUENCE_LENGTH = 16 # <--- REDUCED SEQUENCE LENGTH (from 16 to 8)
RESIZE_DIM = (128, 128) # <--- REDUCED IMAGE DIMENSIONS (from 128x128 to 64x64)
CHANNELS = 3 # Number of color channels (RGB)
BATCH_SIZE = 2 # <--- REDUCED BATCH SIZE (from 2 to 1)

EPOCHS = 50 # Number of training epochs
LEARNING_RATE = 0.0001
MODEL_SAVE_DIR = "/kaggle/working/3DCAE/3DCAE_processed_data/model" # Directory to save trained models
MODEL_FILENAME = "3DCAE_model_to_be_used.h5" # Name for the saved model
LOG_DIR = "/kaggle/working/3DCAE/3DCAE_processed_data/logs" # Directory for TensorBoard logs

# --- Data Paths (from vid2array.py output) ---
TRAINING_VIDEO_PATHS_FILE = "/kaggle/working/3DCAE/3DCAE_processed_data/combined_training_video_paths.txt"
VALIDATION_VIDEO_PATHS_FILE = "/kaggle/working/3DCAE/3DCAE_processed_data/combined_validation_video_paths.txt"

# Ensure model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# --- Model Architecture (3D CNN Autoencoder) ---
def build_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x) # Pool spatially, not temporally
    
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x) # Pool temporally and spatially

    x = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    
    encoded = Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x) # Bottleneck

    # Decoder
    x = Conv3DTranspose(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D(size=(2, 2, 2))(x)

    x = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)

    x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D(size=(1, 2, 2))(x) # Upsample spatially, not temporally in first dim

    decoded = Conv3DTranspose(filters=CHANNELS, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x) # Output layer

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder

# --- Main Training Logic ---
if __name__ == '__main__':
    print("Setting up model and data generator...")
    input_shape = (SEQUENCE_LENGTH, RESIZE_DIM[0], RESIZE_DIM[1], CHANNELS)
    model = build_autoencoder(input_shape)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse') # Mean Squared Error for reconstruction loss

    model.summary()

    # --- Data Generators ---
    # Create the training data generator
    train_generator = VideoSequenceGenerator(
        video_paths_file=TRAINING_VIDEO_PATHS_FILE,
        sequence_length=SEQUENCE_LENGTH,
        resize_dim=RESIZE_DIM,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Create the validation data generator
    validation_generator = VideoSequenceGenerator(
        video_paths_file=VALIDATION_VIDEO_PATHS_FILE,
        sequence_length=SEQUENCE_LENGTH,
        resize_dim=RESIZE_DIM,
        batch_size=BATCH_SIZE,
        shuffle=False # Don't shuffle validation data
    )

    # Callbacks
    model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME),
        monitor='val_loss', # Changed monitor to validation loss
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss', # Changed monitor to validation loss
        patience=5,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss', # Changed monitor to validation loss
        factor=0.5,
        patience=2,
        min_lr=0.000001,
        mode='min',
        verbose=1
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")))


    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback],
        verbose=1
    )

    print("\nTraining finished.")
