# %%
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/ezzo/anaconda3/lib/python3.13/site-packages/nvidia/cuda_nvcc'
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

import models
import HALT_DataLoad
load_halt_raw = HALT_DataLoad.load_halt_raw
standardize_data = HALT_DataLoad.standardize_data
load_halt_subject_by_session = HALT_DataLoad.load_halt_subject_by_session
get_available_subjects = HALT_DataLoad.get_available_subjects


# %%
def plot_confusion(cf_matrix, labels, title, out_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=labels)
    disp.plot()
    disp.ax_.set_xticklabels(labels, rotation=12)
    plt.title(title)
    plt.savefig(out_path)
    plt.close()


def plot_learning_curves(history, title, out_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'{title} — Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{title} — Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def getModel(model_name):
    # Select the model
    if (model_name == 'DB_ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.DB_ATCNet(
            # Dataset parameters
            n_classes=2,
            in_chans=19,
            in_samples=600,

            # Attention Dual-branch Convolution block (ADBC) parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            drop1=0.35,
            depth1=2,
            depth2=4,

            # Sliding window (SW) parameter
            n_windows=5,

            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se', 'improved_cbam'

            # Temporal convolutional Fusion Network block (TCFN) parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            drop2=0.1,
            drop3=0.15,
            drop4=0.15,

            tcn_activation='elu',
        )
    elif (model_name == 'ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.ATCNet(
            # Dataset parameters
            n_classes=2,
            in_chans=19,
            in_samples=600,
            # Sliding window (SW) parameter
            n_windows=5,
            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation='elu'
        )
    elif (model_name == 'TCNet_Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes=2)
    elif (model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes=2)
    elif (model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes=2)
    elif (model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps=600, n_features=19, n_outputs=2)
    elif (model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes=2, Chans=19, Samples=600)
    elif (model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes=2, Chans=19, Samples=600)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model

# %%
def get_data_path():
    """Resolve the Physionet dataset directory.

    Priority:
    1. Environment variable PHYSIONET_DATA_DIR (if it exists and is a directory)
    2. ./files directory inside the repo (if it looks like the dataset)
    3. Original fallback path (/root/autodl-tmp/physionet)

    Raises FileNotFoundError with guidance if no valid path is found.
    """
    # 1) Check environment variable
    env_path = os.environ.get('PHYSIONET_DATA_DIR')
    if env_path:
        if os.path.isdir(env_path):
            print(f"Using PHYSIONET_DATA_DIR={env_path}")
            return env_path
        else:
            print(f"PHYSIONET_DATA_DIR is set to '{env_path}' but that path does not exist or is not a directory.")

    # 2) Check ./files in the repo
    repo_files = os.path.join(os.getcwd(), 'files')
    if os.path.isdir(repo_files):
        # Quick heuristic: look for S001 or RECORDS or any subdirectory starting with 'S'
        looks_like_dataset = False
        if os.path.exists(os.path.join(repo_files, 'S001')) or os.path.exists(os.path.join(repo_files, 'RECORDS')):
            looks_like_dataset = True
        else:
            try:
                for name in os.listdir(repo_files):
                    if os.path.isdir(os.path.join(repo_files, name)) and name.startswith('S'):
                        looks_like_dataset = True
                        break
            except Exception:
                # ignore listing errors, keep looks_like_dataset False
                pass

        if looks_like_dataset:
            print(f"Using local dataset folder: {repo_files}")
            return repo_files
        else:
            print(f"Found '{repo_files}' but it doesn't look like the expected Physionet dataset (no S001/RECORDS found).")

    # 3) Fallback hardcoded path
    fallback = os.path.join(os.getcwd(), 'HALT')
    if os.path.isdir(fallback):
        print(f"Using fallback dataset path: {fallback}")
        return fallback

    # Nothing found -> informative error
    raise FileNotFoundError(
        "Physionet dataset not found.\n"
        "Please either: (a) set the PHYSIONET_DATA_DIR environment variable to the dataset folder,\n"
        "or (b) place the dataset inside ./files (repo root).\n"
        f"Checked locations: PHYSIONET_DATA_DIR={env_path}, ./files={repo_files}, fallback={fallback}"
    )

def run():
    # Get dataset path
    data_path = get_data_path()
    print(f"[DEBUG] Resolved data_path={data_path}")

    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Set dataset paramters
    dataset_conf = {'n_classes': 2, 'n_channels': 19, 'data_path': data_path}
    # Set training hyperparamters
    train_conf = {
        'batch_size': 32, 
        'epochs': 500, 
        'patience': 50,
        'lr': 0.0009,   
        'LearnCurves': True, 
        'model': 'DB_ATCNet'
    }

    # Train subject-dependent: each subject trained and evaluated independently
    print("[DEBUG] Starting Subject-Dependent Training")
    train_subject_dependent(dataset_conf, train_conf, results_path)


def train_subject_dependent(dataset_conf, train_conf, results_path):
    """
    Subject-dependent training: train and evaluate each subject independently
    using leave-one-session-out cross-validation.
    Each fold holds out one entire recording session to prevent
    temporal autocorrelation leakage between train and test.
    """
    print("[DEBUG] Entered train_subject_dependent()")
    in_exp = time.time()

    # Get parameters
    data_path = dataset_conf.get('data_path')
    n_classes = dataset_conf.get('n_classes')
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    model_name = train_conf.get('model')

    # Discover available subjects
    subjects = get_available_subjects(data_path)
    print(f"Found {len(subjects)} subjects: {subjects}")

    # Global log
    global_log = open(results_path + "/global_summary.txt", "w")
    global_log.write(f"Subject-Dependent Training — {model_name}\n")
    global_log.write(f"Leave-one-session-out cross-validation per subject\n")
    global_log.write('=' * 80 + '\n\n')

    # Store per-subject results
    all_subject_accs = {}
    all_subject_kappas = {}

    for subj_idx, subj in enumerate(subjects, 1):
        print(f"\n{'#' * 70}")
        print(f"  SUBJECT {subj} ({subj_idx}/{len(subjects)})")
        print(f"{'#' * 70}")
        in_subj = time.time()

        # Create subject output folder
        subj_path = os.path.join(results_path, f'subject_{subj}')
        os.makedirs(subj_path, exist_ok=True)

        # Load this subject's data WITH session IDs
        X_all, y_all_onehot, y_labels, session_ids, n_channels = \
            load_halt_subject_by_session(subj, data_path)
        if len(X_all) == 0:
            msg = f'Subject {subj}: no data found, skipping.\n'
            print(msg)
            continue

        n_sessions = len(np.unique(session_ids))
        subj_log = open(os.path.join(subj_path, 'subject_summary.txt'), 'w')

        fold_accs = []
        fold_kappas = []

        # Single-session subject: use stratified 80/20 train-test split
        if n_sessions == 1:
            from sklearn.model_selection import train_test_split
            subj_log.write(f'Subject {subj} — {model_name} — Single session (80/20 split)\n')
            subj_log.write('=' * 60 + '\n')
            print(f"  Subject {subj} has only 1 session — using 80/20 stratified split")

            train_idx, test_idx = train_test_split(
                np.arange(len(y_labels)), test_size=0.2,
                stratify=y_labels, random_state=42
            )
            splits = [(train_idx, test_idx)]
            fold_names = ['train_test_split']
        else:
            subj_log.write(f'Subject {subj} — {model_name} — LOSO CV ({n_sessions} sessions)\n')
            subj_log.write('=' * 60 + '\n')
            logo = LeaveOneGroupOut()
            splits = list(logo.split(X_all, y_labels, session_ids))
            fold_names = [f'session_{np.unique(session_ids[ti])[0] + 1}_heldout'
                          for _, ti in splits]

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            fold_label = fold_names[fold - 1]
            print(f"\n{'=' * 50}")
            print(f"  Subject {subj} — {fold_label} (fold {fold}/{len(splits)})")
            print(f"{'=' * 50}")
            in_fold = time.time()

            # Split
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train_oh, y_test_oh = y_all_onehot[train_idx], y_all_onehot[test_idx]

            # Standardize (fit on train, transform both)
            X_train_s, X_test_s = standardize_data(
                X_train.copy(), X_test.copy(), n_channels
            )

            print(f"  Train: {X_train_s.shape[0]}  Test: {X_test_s.shape[0]}")
            print(f"  Train dist: {np.bincount(y_labels[train_idx])}")
            print(f"  Test  dist: {np.bincount(y_labels[test_idx])}")

            # Fold output folder
            fold_path = os.path.join(subj_path, fold_label)
            os.makedirs(fold_path, exist_ok=True)
            weights_path = os.path.join(fold_path, 'best_model.weights.h5')

            # Build & compile a fresh model
            model = getModel(model_name)
            model.compile(
                loss=categorical_crossentropy,
                optimizer=Adam(learning_rate=lr),
                metrics=['accuracy']
            )

            callbacks = [
                ModelCheckpoint(weights_path, monitor='val_accuracy', verbose=1,
                                save_best_only=True, save_weights_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]

            # Train
            history = model.fit(
                X_train_s, y_train_oh,
                validation_data=(X_test_s, y_test_oh),
                epochs=epochs, batch_size=batch_size,
                callbacks=callbacks, verbose=1
            )

            # Load best weights & evaluate
            model.load_weights(weights_path)
            y_pred = model.predict(X_test_s).argmax(axis=-1)
            labels = y_test_oh.argmax(axis=-1)

            acc = accuracy_score(labels, y_pred)
            kappa = cohen_kappa_score(labels, y_pred)
            fold_accs.append(acc)
            fold_kappas.append(kappa)

            out_fold = time.time()
            info = (f'  Subject {subj} Fold {fold}: '
                    f'Acc={acc:.5f}  Kappa={kappa:.5f}  '
                    f'Time={((out_fold - in_fold) / 60):.2f}min')
            print(info)
            subj_log.write(info + '\n')

            # Confusion matrix
            cf = confusion_matrix(labels, y_pred, normalize='pred')
            plot_confusion(
                cf,
                ['Right Hand', 'Left Leg'],
                f'Subject {subj} — {fold_label}',
                os.path.join(fold_path, 'confusion_matrix.png'),
            )

            # Learning curves
            if LearnCurves:
                plot_learning_curves(
                    history,
                    f'Subject {subj} {fold_label}',
                    os.path.join(fold_path, 'learning_curves.png'),
                )

            # Free GPU memory
            keras.backend.clear_session()

        # ---- Subject summary ----
        out_subj = time.time()
        subj_time = (out_subj - in_subj) / 60
        avg_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        avg_kappa = np.mean(fold_kappas)
        std_kappa = np.std(fold_kappas)

        all_subject_accs[subj] = fold_accs
        all_subject_kappas[subj] = fold_kappas

        subj_summary = (
            f'\n{"=" * 60}\n'
            f'Subject {subj} Summary (LOSO, {n_sessions} sessions)\n'
            f'  Per-session Acc:   {[f"{a:.4f}" for a in fold_accs]}\n'
            f'  Per-session Kappa: {[f"{k:.4f}" for k in fold_kappas]}\n'
            f'  Avg Acc:   {avg_acc:.5f} ± {std_acc:.5f}\n'
            f'  Avg Kappa: {avg_kappa:.5f} ± {std_kappa:.5f}\n'
            f'  Time: {subj_time:.2f} min\n'
            f'{"=" * 60}\n'
        )
        print(subj_summary)
        subj_log.write(subj_summary)
        subj_log.close()

        # Save per-subject fold metrics
        np.savez(
            os.path.join(subj_path, 'loso_results.npz'),
            accuracies=np.array(fold_accs),
            kappas=np.array(fold_kappas),
            avg_acc=avg_acc, std_acc=std_acc,
            avg_kappa=avg_kappa, std_kappa=std_kappa
        )

        # Per-subject fold bar chart
        plt.figure(figsize=(10, 5))
        x = np.arange(n_sessions)
        w = 0.35
        plt.bar(x - w/2, fold_accs, w, label='Accuracy', color='steelblue')
        plt.bar(x + w/2, fold_kappas, w, label='Kappa', color='coral')
        plt.axhline(y=avg_acc, color='steelblue', linestyle='--', label=f'Avg Acc: {avg_acc:.3f}')
        plt.axhline(y=avg_kappa, color='coral', linestyle='--', label=f'Avg Kappa: {avg_kappa:.3f}')
        plt.xlabel('Held-out Session'); plt.ylabel('Score')
        plt.title(f'Subject {subj} — LOSO CV ({n_sessions} sessions)')
        plt.xticks(x, [f'Sess {i+1}' for i in range(n_sessions)])
        plt.legend(loc='lower right'); plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(subj_path, 'loso_summary.png'), dpi=150)
        plt.close()

        # Write to global log
        global_log.write(f'Subject {subj}: Acc={avg_acc:.5f}±{std_acc:.5f}  '
                         f'Kappa={avg_kappa:.5f}±{std_kappa:.5f}  '
                         f'Time={subj_time:.2f}min\n')
        global_log.flush()

    # ---- Global summary ----
    out_exp = time.time()
    total_time = (out_exp - in_exp) / 60

    subj_avg_accs = [np.mean(v) for v in all_subject_accs.values()]
    subj_avg_kappas = [np.mean(v) for v in all_subject_kappas.values()]

    global_summary = (
        f'\n{"=" * 80}\n'
        f'GLOBAL SUBJECT-DEPENDENT RESULTS (Leave-One-Session-Out)\n'
        f'{"=" * 80}\n'
        f'Model: {model_name}\n'
        f'Subjects: {len(all_subject_accs)}\n'
        f'CV method: Leave-One-Session-Out\n\n'
    )
    for s in all_subject_accs:
        global_summary += (f'  Subject {s}: Acc={np.mean(all_subject_accs[s]):.5f}±'
                           f'{np.std(all_subject_accs[s]):.5f}  '
                           f'Kappa={np.mean(all_subject_kappas[s]):.5f}±'
                           f'{np.std(all_subject_kappas[s]):.5f}\n')
    global_summary += (
        f'\n  Overall Avg Acc:   {np.mean(subj_avg_accs):.5f} ± {np.std(subj_avg_accs):.5f}\n'
        f'  Overall Avg Kappa: {np.mean(subj_avg_kappas):.5f} ± {np.std(subj_avg_kappas):.5f}\n'
        f'  Total Time: {total_time:.2f} min ({total_time/60:.2f} h)\n'
        f'{"=" * 80}\n'
    )
    print(global_summary)
    global_log.write('\n' + global_summary)
    global_log.close()

    # Save all metrics in one archive
    np.savez(
        os.path.join(results_path, 'all_subjects_results.npz'),
        subjects=np.array(list(all_subject_accs.keys())),
        subject_avg_accs=np.array(subj_avg_accs),
        subject_avg_kappas=np.array(subj_avg_kappas),
        overall_avg_acc=np.mean(subj_avg_accs),
        overall_std_acc=np.std(subj_avg_accs),
        overall_avg_kappa=np.mean(subj_avg_kappas),
        overall_std_kappa=np.std(subj_avg_kappas)
    )

    # All-subjects comparison bar chart
    subj_labels = list(all_subject_accs.keys())
    plt.figure(figsize=(max(10, len(subj_labels) * 1.2), 6))
    x = np.arange(len(subj_labels))
    w = 0.35
    plt.bar(x - w/2, subj_avg_accs, w, label='Avg Accuracy', color='steelblue')
    plt.bar(x + w/2, subj_avg_kappas, w, label='Avg Kappa', color='coral')
    plt.axhline(y=np.mean(subj_avg_accs), color='steelblue', linestyle='--',
                label=f'Grand Avg Acc: {np.mean(subj_avg_accs):.3f}')
    plt.axhline(y=np.mean(subj_avg_kappas), color='coral', linestyle='--',
                label=f'Grand Avg Kappa: {np.mean(subj_avg_kappas):.3f}')
    plt.xlabel('Subject'); plt.ylabel('Score')
    plt.title(f'{model_name} — Subject-Dependent LOSO CV')
    plt.xticks(x, [f'Subj {s}' for s in subj_labels])
    plt.legend(loc='lower right'); plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'all_subjects_summary.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to: {results_path}")


def train_kfold(dataset_conf, train_conf, results_path):
    """
    Train model using stratified k-fold cross-validation.
    This ensures class distribution is preserved in each fold.
    """
    print("[DEBUG] Entered train_kfold()")
    in_exp = time.time()
    
    # Create log files
    log_write = open(results_path + "/log_kfold.txt", "w")
    
    # Get dataset paramters
    data_path = dataset_conf.get('data_path')
    n_classes = dataset_conf.get('n_classes')
    
    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    model_name = train_conf.get('model')
    n_splits = train_conf.get('cv_splits', 5)
    
    print(f'Training model: {model_name} with {n_splits}-fold stratified cross-validation')
    log_write.write(f'Model: {model_name}, {n_splits}-fold stratified cross-validation\n')
    log_write.write('='*80 + '\n')
    
    # Load all data (without train/test split)
    print("[DEBUG] Loading all data...")
    X_all, y_all_onehot, y_labels, n_channels = load_halt_raw(data_path)
    print(f"[DEBUG] Data loaded: X_all={X_all.shape}, y_all={y_all_onehot.shape}")
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_accuracies = []
    fold_kappas = []
    fold_histories = []
    
    # Iterate through folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_labels), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*60}")
        in_fold = time.time()
        
        # Split data for this fold
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train_onehot, y_test_onehot = y_all_onehot[train_idx], y_all_onehot[test_idx]
        
        # Standardize data (fit on train, transform both)
        X_train_scaled, X_test_scaled = standardize_data(
            X_train.copy(), X_test.copy(), n_channels
        )
        
        print(f"Train size: {X_train_scaled.shape[0]}, Test size: {X_test_scaled.shape[0]}")
        
        # Check class distribution
        train_dist = np.bincount(y_labels[train_idx])
        test_dist = np.bincount(y_labels[test_idx])
        print(f"Train class distribution: {train_dist}")
        print(f"Test class distribution: {test_dist}")
        
        # Create folder for this fold's saved models
        fold_path = results_path + f'/fold_{fold}'
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        filepath = fold_path + '/best_model.weights.h5'
        
        # Create a fresh model for each fold
        model = getModel(model_name)
        model.compile(
            loss=categorical_crossentropy, 
            optimizer=Adam(learning_rate=lr), 
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                          save_best_only=True, save_weights_only=True, mode='max'),
            EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
        ]
        
        # Train
        history = model.fit(
            X_train_scaled, y_train_onehot, 
            validation_data=(X_test_scaled, y_test_onehot),
            epochs=epochs, batch_size=batch_size, 
            callbacks=callbacks, verbose=1
        )
        fold_histories.append(history)
        
        # Load best weights and evaluate
        model.load_weights(filepath)
        y_pred = model.predict(X_test_scaled).argmax(axis=-1)
        labels = y_test_onehot.argmax(axis=-1)
        
        acc = accuracy_score(labels, y_pred)
        kappa = cohen_kappa_score(labels, y_pred)
        
        fold_accuracies.append(acc)
        fold_kappas.append(kappa)
        
        out_fold = time.time()
        fold_time = (out_fold - in_fold) / 60
        
        # Log fold results
        info = f'Fold {fold}: Accuracy={acc:.5f}, Kappa={kappa:.5f}, Time={fold_time:.2f}min'
        print(info)
        log_write.write(info + '\n')
        
        # Generate and save confusion matrix for this fold
        cf_matrix = confusion_matrix(labels, y_pred, normalize='pred')
        plot_confusion(
            cf_matrix,
            ['Right Hand', 'Left Leg'],
            f'Confusion Matrix - Fold {fold}',
            fold_path + '/confusion_matrix.png',
        )
        
        # Plot learning curves for this fold
        if LearnCurves:
            plot_learning_curves(
                history,
                f'Fold {fold}',
                fold_path + '/learning_curves.png',
            )
        
        # Clear session to free memory
        keras.backend.clear_session()
    
    # Calculate and report final results
    out_exp = time.time()
    total_time = (out_exp - in_exp) / 60
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    avg_kappa = np.mean(fold_kappas)
    std_kappa = np.std(fold_kappas)
    
    results_summary = f"""
    Model: {model_name}
    Number of Folds: {n_splits}
    
    Per-fold Accuracies: {[f'{a:.4f}' for a in fold_accuracies]}
    Per-fold Kappas: {[f'{k:.4f}' for k in fold_kappas]}
    
    Average Accuracy: {avg_acc:.5f} ± {std_acc:.5f}
    Average Kappa: {avg_kappa:.5f} ± {std_kappa:.5f}
    
    Total Training Time: {total_time:.2f} minutes ({total_time/60:.2f} hours)
    """
    
    print(results_summary)
    log_write.write('\n' + '='*80 + '\n')
    log_write.write('FINAL RESULTS\n')
    log_write.write(results_summary)
    log_write.close()
    
    # Save all fold metrics
    np.savez(
        results_path + '/kfold_results.npz',
        accuracies=np.array(fold_accuracies),
        kappas=np.array(fold_kappas),
        avg_acc=avg_acc,
        std_acc=std_acc,
        avg_kappa=avg_kappa,
        std_kappa=std_kappa
    )
    
    # Plot summary bar chart
    plt.figure(figsize=(10, 5))
    x = np.arange(n_splits)
    width = 0.35
    
    plt.bar(x - width/2, fold_accuracies, width, label='Accuracy', color='steelblue')
    plt.bar(x + width/2, fold_kappas, width, label='Kappa', color='coral')
    
    plt.axhline(y=avg_acc, color='steelblue', linestyle='--', label=f'Avg Acc: {avg_acc:.3f}')
    plt.axhline(y=avg_kappa, color='coral', linestyle='--', label=f'Avg Kappa: {avg_kappa:.3f}')
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title(f'{model_name} - {n_splits}-Fold Cross-Validation Results')
    plt.xticks(x, [f'Fold {i+1}' for i in range(n_splits)])
    plt.legend(loc='lower right')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(results_path + '/kfold_summary.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {results_path}")


# %%
if __name__ == "__main__":
    run()