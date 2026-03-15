import tensorflow as tf
import numpy as np
import os
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_test, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

def to_one_hot(y, by_sub=False):
    if by_sub:
        new_array = np.array(["nan" for nan in range(len(y))])
        for index, label in enumerate(y):
            new_array[index] = ''.join([i for i in label if not i.isdigit()])
    else:
        new_array = y.copy()
    total_labels = np.unique(new_array)
    mapping = {}
    for x in range(len(total_labels)):
        mapping[total_labels[x]] = x
    for x in range(len(new_array)):
        new_array[x] = mapping[new_array[x]]

    return tf.keras.utils.to_categorical(new_array)

def load_halt_subject_data(subject_char: str, data_path: str):
    import scipy.io as sio
    files = [f for f in os.listdir(data_path) if f.startswith(f"HaLTSubject{subject_char}") and f.endswith(".mat")]
    
    # 6 classes: 1: Left Hand, 2: Right Hand, 3: Passive, 4: Left Leg, 5: Tongue, 6: Right Leg
    # We only want Right Hand (2) and Left Leg (4)
    classes_to_plot = [2, 4]
    
    xs_all = []
    ys_all = []
    
    # 200 Hz * 3 seconds = 600 samples
    samples_per_trial = 600 

    for file in files:
        mat = sio.loadmat(os.path.join(data_path, file), squeeze_me=True, struct_as_record=False)
        o = mat['o']
        data = o.data # shape (time, 22)
        marker = o.marker # shape (time,)
        
        for cls in classes_to_plot:
            # find where marker changes to cls
            transitions = np.where((marker[1:] == cls) & (marker[:-1] != cls))[0]
            # also check if the very first element is this class
            if marker[0] == cls:
                transitions = np.insert(transitions, 0, 0)
                
            for start_idx in transitions:
                # Align trial start with the X3 sync pulse (channel index 21).
                # The pulse appears as a spike shortly after the marker transition.
                # We search a small window (e.g. 50 samples = 250ms) for the actual sync peak.
                search_window_end = min(start_idx + 50, data.shape[0])
                if search_window_end > start_idx:
                    x3_window = data[start_idx:search_window_end, 21]
                    # Find the local peak in X3 within this window
                    sync_offset = np.argmax(x3_window)
                    aligned_start_idx = start_idx + sync_offset
                else:
                    aligned_start_idx = start_idx

                end_idx = aligned_start_idx + samples_per_trial
                if end_idx <= data.shape[0]:
                    trial_data = data[aligned_start_idx:end_idx, :] # shape (600, 22)
                    # The 22 channels are Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 A1 A2 F7 F8 T3 T4 T5 T6 Fz Cz Pz X3
                    # A1 (10), A2 (11) are ground nodes at earlobes. X3 (21) is sync bipolar lead.
                    # We explicitly select only the 19 actual EEG leads.
                    eeg_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                    trial_data = trial_data[:, eeg_indices]
                    # We transpose to match standard format (channels, time) -> (19, 600)
                    trial_data = trial_data.T
                    xs_all.append(trial_data)
                    ys_all.append(cls)
                    
    if not xs_all:
        return np.array([]), np.array([])
        
    return np.array(xs_all), np.array(ys_all)

def load_halt(path):
    subjects_chars = [chr(i) for i in range(ord('A'), ord('N'))] # A to M
    
    xs = list()
    ys = list()
    for subject in subjects_chars:
        x, y = load_halt_subject_data(subject, path)
        if len(x) > 0:
            print(f"Subject {subject}: {x.shape}")
            xs.append(x)
            ys.append(y)
            
    data_x = np.concatenate(xs)
    data_y = np.concatenate(ys)

    N_tr, N_ch, N_samples = data_x.shape
    # Add empty dimension for Conv2D channel like in original code
    data_x = data_x.reshape(N_tr, 1, N_ch, -1)
    y_one_hot = to_one_hot(data_y, by_sub=False)

    # Create Validation/test
    x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(data_x,
                                                                                y_one_hot,
                                                                                stratify=y_one_hot,
                                                                                test_size=0.10,
                                                                                random_state=42)

    #Scale indipendently train/test
    x_train_scaled_raw, x_test_valid_scaled_raw = standardize_data(x_train_raw, x_valid_test_raw, N_ch)

    print(x_train_scaled_raw.shape, x_test_valid_scaled_raw.shape)
    return x_train_scaled_raw,y_train_raw,x_test_valid_scaled_raw,y_valid_test_raw


def load_halt_raw(path):
    subjects_chars = [chr(i) for i in range(ord('A'), ord('N'))] # A to M

    xs = list()
    ys = list()
    for subject in subjects_chars:
        x, y = load_halt_subject_data(subject, path)
        if len(x) > 0:
            print(f"Subject {subject}: {x.shape}")
            xs.append(x)
            ys.append(y)
            
    data_x = np.concatenate(xs)
    data_y = np.concatenate(ys)

    N_tr, N_ch, N_samples = data_x.shape
    data_x = data_x.reshape(N_tr, 1, N_ch, -1)
    
    y_one_hot = to_one_hot(data_y, by_sub=False)
    
    # Get integer labels for stratification
    y_labels = np.argmax(y_one_hot, axis=1)

    print(f"Total data shape: {data_x.shape}, Labels shape: {y_one_hot.shape}")
    return data_x, y_one_hot, y_labels, N_ch


def get_available_subjects(path):
    """Discover which subject letters have .mat files in the data directory."""
    import re
    subjects = set()
    for f in os.listdir(path):
        m = re.match(r'HaLTSubject([A-Z])', f)
        if m:
            subjects.add(m.group(1))
    return sorted(subjects)


def load_halt_subject_raw(subject_char, path):
    """Load a single subject's data ready for k-fold CV.

    Returns (X, y_onehot, y_labels, n_channels) where:
      X: shape (N_trials, 1, N_ch, N_samples)
      y_onehot: one-hot encoded labels
      y_labels: integer labels for stratification
      n_channels: number of EEG channels
    """
    x, y = load_halt_subject_data(subject_char, path)
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), 0

    N_tr, N_ch, N_samples = x.shape
    x = x.reshape(N_tr, 1, N_ch, -1)

    y_onehot = to_one_hot(y, by_sub=False)
    y_labels = np.argmax(y_onehot, axis=1)

    print(f"Subject {subject_char}: {x.shape}, classes={np.bincount(y_labels)}")
    return x, y_onehot, y_labels, N_ch


def load_halt_subject_by_session(subject_char, path):
    """Load a single subject's data with session IDs for leave-one-session-out CV.

    Returns (X, y_onehot, y_labels, session_ids, n_channels) where:
      X: shape (N_trials, 1, N_ch, N_samples)
      y_onehot: one-hot encoded labels
      y_labels: integer labels for stratification
      session_ids: array of session index per trial (0, 1, 2, ...)
      n_channels: number of EEG channels
    """
    import scipy.io as sio

    files = sorted([f for f in os.listdir(path)
                    if f.startswith(f"HaLTSubject{subject_char}") and f.endswith(".mat")])

    if not files:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0

    classes_to_use = [2, 4]  # Right Hand, Left Leg
    samples_per_trial = 600
    eeg_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    xs_all, ys_all, sess_all = [], [], []

    for sess_idx, file in enumerate(files):
        mat = sio.loadmat(os.path.join(path, file), squeeze_me=True, struct_as_record=False)
        o = mat['o']
        data = o.data
        marker = o.marker

        for cls in classes_to_use:
            transitions = np.where((marker[1:] == cls) & (marker[:-1] != cls))[0]
            if marker[0] == cls:
                transitions = np.insert(transitions, 0, 0)

            for start_idx in transitions:
                search_window_end = min(start_idx + 50, data.shape[0])
                if search_window_end > start_idx:
                    x3_window = data[start_idx:search_window_end, 21]
                    sync_offset = np.argmax(x3_window)
                    aligned_start_idx = start_idx + sync_offset
                else:
                    aligned_start_idx = start_idx

                end_idx = aligned_start_idx + samples_per_trial
                if end_idx <= data.shape[0]:
                    trial_data = data[aligned_start_idx:end_idx, eeg_indices].T
                    xs_all.append(trial_data)
                    ys_all.append(cls)
                    sess_all.append(sess_idx)

    if not xs_all:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0

    X = np.array(xs_all)
    N_tr, N_ch, N_samples = X.shape
    X = X.reshape(N_tr, 1, N_ch, -1)

    y_onehot = to_one_hot(np.array(ys_all), by_sub=False)
    y_labels = np.argmax(y_onehot, axis=1)
    session_ids = np.array(sess_all)

    n_sessions = len(set(sess_all))
    print(f"Subject {subject_char}: {X.shape}, classes={np.bincount(y_labels)}, "
          f"sessions={n_sessions}, trials/session={[np.sum(session_ids == i) for i in range(n_sessions)]}")
    return X, y_onehot, y_labels, session_ids, N_ch
