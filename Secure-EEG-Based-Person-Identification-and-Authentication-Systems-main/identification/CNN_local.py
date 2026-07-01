# -*- coding: utf-8 -*-
"""
EEG Person Identification using CNN - Local Version
This script is adapted from the Colab notebook to run locally.
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pyedflib
from tqdm import tqdm
import time
import os
import glob

# ============== CONFIGURATION ==============
# Update these paths to your local dataset and output folders
DATASET_PATH = os.environ.get("EEG_DATASET_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "files"))
OUTPUT_DIR = os.environ.get("EEG_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "output"))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============== MODEL DEFINITION ==============
def CNN():
    input_shape = (160, 64)

    model = models.Sequential()

    model.add(layers.BatchNormalization(input_shape=input_shape, epsilon=.0001))
    
    model.add(layers.Conv1D(input_shape=input_shape, activation='relu', filters=128, kernel_size=2, strides=1, padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Conv1D(input_shape=(80, 128), activation='relu', filters=256, kernel_size=2, strides=1, padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Conv1D(input_shape=(40, 256), activation='relu', filters=512, kernel_size=2, strides=1, padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Conv1D(input_shape=(20, 512), activation='relu', filters=1024, kernel_size=2, strides=1, padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Reshape((-1, 64*160), input_shape=(80, 10, 1024)))
    model.add(layers.Dropout(rate=0.5, input_shape=(80, 10240)))
    
    model.add(tf.keras.layers.Dense(109, activation='softmax'))

    model.summary()
    return model


# ============== DATA LOADING FUNCTIONS ==============
def _read_py_function(filename):
    """Read EDF file and extract EEG data with labels."""
    # Handle both string and bytes
    if isinstance(filename, bytes):
        filename = filename.decode()
    elif hasattr(filename, 'numpy'):
        filename = filename.numpy().decode()
    
    try:
        f = pyedflib.EdfReader(filename)
        n_channels = f.signals_in_file
        eeg_data = np.zeros((n_channels, f.getNSamples()[0]), dtype=np.float32)
        for i in np.arange(n_channels):
            eeg_data[i, :] = f.readSignal(i)
        
        n_samples = f.getNSamples()[0]
        f.close()  # Important: close the file after reading

        reminder = int(n_samples % 160)
        n_samples -= reminder
        seconds = int(n_samples/160)  # 160 is frequency
        
        # Extract person ID from filename - works with Windows paths
        path_parts = filename.replace("\\", "/").split("/")
        person_filename = path_parts[-1]
        person_id = int(person_filename.partition("S")[2].partition("R")[0])
        
        label = np.zeros(109, dtype=bool)  # 109 classes (persons)
        label[person_id-1] = 1
        labels = np.tile(label, (seconds, 1))
        
        eeg_data = eeg_data.transpose()
        if reminder > 0:
            eeg_data = eeg_data[:-reminder, :]
        intervals = np.linspace(0, n_samples, num=seconds, endpoint=False, dtype=int)
        eeg_data = np.split(eeg_data, intervals)
        del eeg_data[0]
        eeg_data = np.array(eeg_data)  # shape = (seconds, frequency, n_channels)
        return eeg_data, labels
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.array([]), np.array([])


def data_generator(files):
    """Generator that yields (sample, label) pairs."""
    for filename in files:
        eeg_data, labels = _read_py_function(filename)
        if len(eeg_data) > 0:
            for i in range(len(eeg_data)):
                # Yield single sample and label with shape (1, 109)
                yield eeg_data[i], np.expand_dims(labels[i], axis=0)


def get_tf_dataset(dataset_type="train", batch_size=80, shuffle=True):
    """
    Create a tf.data.Dataset for training/testing.
    """
    path = DATASET_PATH
    
    if dataset_type == "train":
        files = []
        for i in range(1, 13):
            pattern = os.path.join(path, "S*", f"S*R{i:02d}.edf")
            files.extend(glob.glob(pattern))
    elif dataset_type == "test":
        pattern = os.path.join(path, "S*", "S*R13.edf")
        files = glob.glob(pattern)
    elif dataset_type == "validation":
        pattern = os.path.join(path, "S*", "S*R14.edf")
        files = glob.glob(pattern)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    print(f"Found {len(files)} files for {dataset_type}")
    
    output_signature = (
        tf.TensorSpec(shape=(160, 64), dtype=tf.float32),
        tf.TensorSpec(shape=(1, 109), dtype=tf.bool)
    )
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(files),
        output_signature=output_signature
    )
    
    if shuffle:
        dataset = dataset.shuffle(10000)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ============== MAIN TRAINING SCRIPT ==============
def main():
    print("=" * 50)
    print("EEG Person Identification - Local Version (Memory Optimized)")
    print("=" * 50)
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path does not exist: {DATASET_PATH}")
        return
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = get_tf_dataset(dataset_type="train", batch_size=80, shuffle=True)
    test_dataset = get_tf_dataset(dataset_type="test", batch_size=80, shuffle=False)
    
    # Create model
    print("\nCreating CNN model...")
    model = CNN()
    
    tf.keras.optimizers.Adam(learning_rate=0.00001)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    # Setup checkpoints
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "cp-{epoch:04d}.weights.h5")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    model.save_weights(checkpoint_path.format(epoch=0))
    
    # Train
    print("\nStarting training...")
    history = model.fit(train_dataset, 
                        epochs=50, 
                        validation_data=test_dataset, 
                        callbacks=[cp_callback])
    
    # Save history
    history_path = os.path.join(OUTPUT_DIR, 'history.npy')
    np.save(history_path, history.history)
    print(f"\nHistory saved to: {history_path}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'training_history.png')}")
    
    # Save model
    model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))
    print("Done.")

if __name__ == "__main__":
    main()
