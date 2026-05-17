# %%
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/ezzo/anaconda3/lib/python3.13/site-packages/nvidia/cuda_nvcc'
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
from gumbel_channel_selection import GumbelAnnealingCallback
load_halt_raw = HALT_DataLoad.load_halt_raw
standardize_data = HALT_DataLoad.standardize_data
load_halt_subject_by_session = HALT_DataLoad.load_halt_subject_by_session
get_available_subjects = HALT_DataLoad.get_available_subjects

# ── Channel Ablation Configuration ──────────────────────────────────────────
# Channel names in the order used by HALT_DataLoad (19 EEG leads)
CHANNEL_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
                  'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

# Cross-subject ranked order (based on the mean-std consistency metric)
RANKED_CHANNELS = ['Cz',   # Drop: 0.0156
    'P3',   # Drop: 0.0128
    'T5',   # Drop: 0.0108
    'F7',   # Drop: 0.0106
    'T3',   # Drop: 0.0103
    'O2',   # Drop: 0.0096
    'F8',   # Drop: 0.0090
    'P4',   # Drop: 0.0085
    'Fz',   # Drop: 0.0069
    'C4',   # Drop: 0.0059
    'T6',   # Drop: 0.0036
    'Pz',   # Drop: 0.0017
    'F3',   # Drop: 0.0017
    'F4',   # Drop: 0.0011
    'T4',   # Drop: -0.0001
    'C3',   # Drop: -0.0005
    'Fp1',  # Drop: -0.0005
    'Fp2',   # Drop: -0.0014
    'O1',   # Drop: 0.0119
    ]
RANKED_INDICES = [CHANNEL_NAMES.index(ch) for ch in RANKED_CHANNELS]
# ────────────────────────────────────────────────────────────────────────────

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


def getModel(model_name, n_chans=19, **kwargs):
    # Select the model
    if (model_name == 'DB_ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.DB_ATCNet(
            # Dataset parameters
            n_classes=2,
            in_chans=n_chans,
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
            attention='improved_cbam',  # Options: None, 'mha','mhla', 'cbam', 'se', 'improved_cbam'

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
            in_chans=n_chans,
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
        model = models.EEGNeX_8_32(n_timesteps=600, n_features=n_chans, n_outputs=2)
    elif (model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes=2, Chans=n_chans, Samples=600)
    elif (model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes=2, Chans=n_chans, Samples=600)
    elif (model_name == 'DB_ATCNet_GumbelSelect'):
        # DB-ATCNet with Gumbel-softmax channel selection (Strypsteen & Bertrand 2021)
        # n_chans here is the number of channels to SELECT (K)
        # n_channels_total comes from gumbel_kwargs
        gumbel_kwargs = kwargs.get('gumbel_kwargs', {})
        model = models.DB_ATCNet_GumbelSelect(
            n_classes=2,
            n_channels_total=gumbel_kwargs.get('n_channels_total', 19),
            n_channels_select=n_chans,
            in_samples=600,
            gumbel_lambda=gumbel_kwargs.get('gumbel_lambda', 0.2),
            # Same DB-ATCNet hyperparameters as the baseline
            eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7,
            eegn_dropout=0.3, drop1=0.35, depth1=2, depth2=4,
            n_windows=5,
            attention='improved_cbam',
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32,
            tcn_dropout=0.3, drop2=0.1, drop3=0.15, drop4=0.15,
            tcn_activation='elu',
        )
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


def train_subject_dependent(dataset_conf, train_conf, results_path,
                             channel_indices=None):
    """
    Subject-dependent training: train and evaluate each subject independently
    using leave-one-session-out cross-validation.
    Each fold holds out one entire recording session to prevent
    temporal autocorrelation leakage between train and test.

    channel_indices: list of int, optional. If provided, only these channel
        indices (into the 19-channel array) are kept. None = use all 19.
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

        # ── Channel subset selection ──────────────────────────────────────
        if channel_indices is not None:
            X_all = X_all[:, :, channel_indices, :]   # (N, 1, n_ch, 600)
            n_channels = len(channel_indices)
            ch_names = [CHANNEL_NAMES[i] for i in channel_indices]
            print(f"  Using {n_channels} channels: {ch_names}")
        # ─────────────────────────────────────────────────────────────────

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
            model = getModel(model_name, n_chans=n_channels)
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

            # Free GPU and System memory
            del model, history, X_train, X_test, X_train_s, X_test_s, y_train_oh, y_test_oh, y_pred, labels
            keras.backend.clear_session()
            import gc
            gc.collect()

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
        
        # Explicit free of subject large arrays
        del X_all, y_all_onehot, y_labels, session_ids
        keras.backend.clear_session()
        gc.collect()



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
        del model, history, X_train, X_test, X_train_scaled, X_test_scaled, y_train_onehot, y_test_onehot, y_pred, labels
        keras.backend.clear_session()
        import gc
        gc.collect()
    
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


def train_gumbel_selection(dataset_conf, train_conf, results_path,
                           n_select_channels, n_channels_total=19):
    """Train DB-ATCNet with Gumbel-softmax channel selection.

    Follows Strypsteen & Bertrand (2021): the selection layer and network
    weights are trained jointly in a single end-to-end phase.  Uses the
    existing subject-dependent LOSO CV pipeline.

    Parameters
    ----------
    dataset_conf : dict
        Dataset configuration.
    train_conf : dict
        Training configuration (will be modified to use Gumbel model).
    results_path : str
        Output directory for results.
    n_select_channels : int
        Number of channels K to select.
    n_channels_total : int
        Total number of input channels N (default: 19).
    """
    print(f"\n{'#' * 70}")
    print(f"  GUMBEL CHANNEL SELECTION: selecting {n_select_channels} from {n_channels_total} channels")
    print(f"{'#' * 70}")
    in_exp = time.time()

    # Get parameters
    data_path = dataset_conf.get('data_path')
    n_classes = dataset_conf.get('n_classes')
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')

    # Discover available subjects
    subjects = get_available_subjects(data_path)
    print(f"Found {len(subjects)} subjects: {subjects}")

    # Global log
    global_log = open(results_path + "/global_summary.txt", "w")
    global_log.write(f"Gumbel Channel Selection — DB_ATCNet_GumbelSelect\n")
    global_log.write(f"Selecting {n_select_channels} from {n_channels_total} channels\n")
    global_log.write(f"Leave-one-session-out cross-validation per subject\n")
    global_log.write('=' * 80 + '\n\n')

    all_subject_accs = {}
    all_subject_kappas = {}
    all_subject_selections = {}

    for subj_idx, subj in enumerate(subjects, 1):
        print(f"\n{'#' * 70}")
        print(f"  SUBJECT {subj} ({subj_idx}/{len(subjects)})")
        print(f"{'#' * 70}")
        in_subj = time.time()

        subj_path = os.path.join(results_path, f'subject_{subj}')
        os.makedirs(subj_path, exist_ok=True)

        # Load this subject's data WITH session IDs
        X_all, y_all_onehot, y_labels, session_ids, n_channels = \
            load_halt_subject_by_session(subj, data_path)
        if len(X_all) == 0:
            print(f'Subject {subj}: no data found, skipping.')
            continue

        # NOTE: no channel slicing — the Gumbel layer handles selection
        print(f"  Input channels: {n_channels} → selecting {n_select_channels}")

        n_sessions = len(np.unique(session_ids))
        subj_log = open(os.path.join(subj_path, 'subject_summary.txt'), 'w')

        fold_accs = []
        fold_kappas = []
        fold_selections = []

        # Single-session subject: use stratified 80/20 train-test split
        if n_sessions == 1:
            from sklearn.model_selection import train_test_split
            subj_log.write(f'Subject {subj} — Gumbel K={n_select_channels} — Single session (80/20 split)\n')
            subj_log.write('=' * 60 + '\n')
            print(f"  Subject {subj} has only 1 session — using 80/20 stratified split")

            train_idx, test_idx = train_test_split(
                np.arange(len(y_labels)), test_size=0.2,
                stratify=y_labels, random_state=42
            )
            splits = [(train_idx, test_idx)]
            fold_names = ['train_test_split']
        else:
            subj_log.write(f'Subject {subj} — Gumbel K={n_select_channels} — LOSO CV ({n_sessions} sessions)\n')
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

            # Build model with Gumbel selection layer
            model = getModel('DB_ATCNet_GumbelSelect', n_chans=n_select_channels,
                             gumbel_kwargs={
                                 'n_channels_total': n_channels_total,
                                 'gumbel_lambda': 1.0,
                             })
            model.compile(
                loss=categorical_crossentropy,
                optimizer=Adam(learning_rate=lr),
                metrics=['accuracy']
            )

            # Get the Gumbel selection layer for the callback
            gumbel_layer = model.get_layer('gumbel_selection')

            # Callbacks: standard + Gumbel annealing
            gumbel_callback = GumbelAnnealingCallback(
                gumbel_layer=gumbel_layer,
                total_epochs=epochs,
                start_temp=10.0, end_temp=0.1,
                start_thresh=3.0, end_thresh=1.0,
                temp_anneal_fraction=0.25,
                thresh_anneal_fraction=0.25,
                channel_names=CHANNEL_NAMES,
                verbose=True
            )

            callbacks = [
                ModelCheckpoint(weights_path, monitor='val_accuracy', verbose=1,
                                save_best_only=True, save_weights_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience),
                gumbel_callback,
            ]

            # Train
            history = model.fit(
                X_train_s, y_train_oh,
                validation_data=(X_test_s, y_test_oh),
                epochs=epochs, batch_size=batch_size,
                callbacks=callbacks, verbose=1
            )

            # Load best weights & evaluate (inference mode = hard selection)
            model.load_weights(weights_path)
            y_pred = model.predict(X_test_s).argmax(axis=-1)
            labels = y_test_oh.argmax(axis=-1)

            acc = accuracy_score(labels, y_pred)
            kappa = cohen_kappa_score(labels, y_pred)
            fold_accs.append(acc)
            fold_kappas.append(kappa)

            # Extract selected channels
            selected_indices = gumbel_layer.get_selected_channels()
            selected_names = [CHANNEL_NAMES[i] for i in selected_indices]
            n_unique = len(set(selected_indices.tolist()))
            fold_selections.append(selected_indices.tolist())

            out_fold = time.time()
            info = (f'  Subject {subj} Fold {fold}: '
                    f'Acc={acc:.5f}  Kappa={kappa:.5f}  '
                    f'Time={((out_fold - in_fold) / 60):.2f}min\n'
                    f'    Selected channels ({n_unique} unique): {selected_names}')
            print(info)
            subj_log.write(info + '\n')

            # Save selected channels info
            sel_info = {
                'selected_indices': selected_indices.tolist(),
                'selected_names': selected_names,
                'n_unique': n_unique,
                'selection_probabilities': gumbel_layer.get_selection_probabilities().tolist(),
                'final_entropy': gumbel_layer.get_normalized_entropy().tolist(),
                'entropy_history': gumbel_callback.entropy_history,
            }
            import json
            with open(os.path.join(fold_path, 'selected_channels.json'), 'w') as f:
                json.dump(sel_info, f, indent=2)

            # Confusion matrix
            cf = confusion_matrix(labels, y_pred, normalize='pred')
            plot_confusion(
                cf,
                ['Right Hand', 'Left Leg'],
                f'Subject {subj} — {fold_label} (Gumbel K={n_select_channels})',
                os.path.join(fold_path, 'confusion_matrix.png'),
            )

            # Learning curves
            if LearnCurves:
                plot_learning_curves(
                    history,
                    f'Subject {subj} {fold_label} (Gumbel K={n_select_channels})',
                    os.path.join(fold_path, 'learning_curves.png'),
                )

            # Free GPU and system memory
            del model, history, X_train, X_test, X_train_s, X_test_s, y_train_oh, y_test_oh, y_pred, labels
            keras.backend.clear_session()
            import gc
            gc.collect()

        # ---- Subject summary ----
        out_subj = time.time()
        subj_time = (out_subj - in_subj) / 60
        avg_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        avg_kappa = np.mean(fold_kappas)
        std_kappa = np.std(fold_kappas)

        all_subject_accs[subj] = fold_accs
        all_subject_kappas[subj] = fold_kappas
        all_subject_selections[subj] = fold_selections

        subj_summary = (
            f'\n{"=" * 60}\n'
            f'Subject {subj} Summary (Gumbel K={n_select_channels}, LOSO, {n_sessions} sessions)\n'
            f'  Per-session Acc:   {[f"{a:.4f}" for a in fold_accs]}\n'
            f'  Per-session Kappa: {[f"{k:.4f}" for k in fold_kappas]}\n'
            f'  Avg Acc:   {avg_acc:.5f} ± {std_acc:.5f}\n'
            f'  Avg Kappa: {avg_kappa:.5f} ± {std_kappa:.5f}\n'
            f'  Selected channels per fold: {fold_selections}\n'
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

        # Write to global log
        global_log.write(f'Subject {subj}: Acc={avg_acc:.5f}±{std_acc:.5f}  '
                         f'Kappa={avg_kappa:.5f}±{std_kappa:.5f}  '
                         f'Selections={fold_selections}  '
                         f'Time={subj_time:.2f}min\n')
        global_log.flush()

        # Explicit free of subject large arrays
        del X_all, y_all_onehot, y_labels, session_ids
        keras.backend.clear_session()
        gc.collect()

    # ---- Global summary ----
    out_exp = time.time()
    total_time = (out_exp - in_exp) / 60

    subj_avg_accs = [np.mean(v) for v in all_subject_accs.values()]
    subj_avg_kappas = [np.mean(v) for v in all_subject_kappas.values()]

    global_summary = (
        f'\n{"=" * 80}\n'
        f'GLOBAL GUMBEL SELECTION RESULTS (K={n_select_channels})\n'
        f'{"=" * 80}\n'
        f'Selecting {n_select_channels} from {n_channels_total} channels\n'
        f'Subjects: {len(all_subject_accs)}\n'
        f'CV method: Leave-One-Session-Out\n\n'
    )
    for s in all_subject_accs:
        sel_str = str(all_subject_selections.get(s, []))
        global_summary += (f'  Subject {s}: Acc={np.mean(all_subject_accs[s]):.5f}±'
                           f'{np.std(all_subject_accs[s]):.5f}  '
                           f'Kappa={np.mean(all_subject_kappas[s]):.5f}±'
                           f'{np.std(all_subject_kappas[s]):.5f}  '
                           f'Sel={sel_str}\n')
    global_summary += (
        f'\n  Overall Avg Acc:   {np.mean(subj_avg_accs):.5f} ± {np.std(subj_avg_accs):.5f}\n'
        f'  Overall Avg Kappa: {np.mean(subj_avg_kappas):.5f} ± {np.std(subj_avg_kappas):.5f}\n'
        f'  Total Time: {total_time:.2f} min ({total_time/60:.2f} h)\n'
        f'{"=" * 80}\n'
    )
    print(global_summary)
    global_log.write('\n' + global_summary)
    global_log.close()

    # Save all metrics
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

    print(f"\nResults saved to: {results_path}")


def run_gumbel(k_channels_list=(3, 5, 7, 10)):
    """Run Gumbel-softmax channel selection for multiple K values."""
    data_path = get_data_path()
    gumbel_root = os.path.join(os.getcwd(), 'results', 'gumbel_selection')
    os.makedirs(gumbel_root, exist_ok=True)

    train_conf = {
        'batch_size': 32,
        'epochs': 500,
        'patience': 50,
        'lr': 0.0009,
        'LearnCurves': True,
        'model': 'DB_ATCNet_GumbelSelect',
    }
    dataset_conf = {'n_classes': 2, 'n_channels': 19, 'data_path': data_path}

    for k in k_channels_list:
        print(f"\n{'#' * 70}")
        print(f"  GUMBEL SELECTION: K={k}")
        print(f"{'#' * 70}")

        k_path = os.path.join(gumbel_root, f'k_{k}')
        os.makedirs(k_path, exist_ok=True)

        train_gumbel_selection(
            dataset_conf, train_conf, k_path,
            n_select_channels=k,
            n_channels_total=19
        )

        # Free memory between K values
        import gc
        keras.backend.clear_session()
        gc.collect()

    print(f"\nAll Gumbel selection results saved to: {gumbel_root}")


def run_ablation(channel_counts=(3, 5, 7, 10)):
    """Ablation study: retrain DB-ATCNet for each channel subset
    and compare accuracy vs the 19-channel baseline.

    Results are saved to results/ablation/<config_name>/
    A final comparison table is printed and saved to results/ablation/summary.txt
    """
    data_path = get_data_path()
    ablation_root = os.path.join(os.getcwd(), 'results', 'ablation')
    os.makedirs(ablation_root, exist_ok=True)

    train_conf = {
        'batch_size': 32,
        'epochs': 500,
        'patience': 50,
        'lr': 0.0009,
        'LearnCurves': True,
        'model': 'DB_ATCNet',
    }
    dataset_conf = {'n_classes': 2, 'n_channels': 19, 'data_path': data_path}

    # Store per-config per-subject averages for the comparison table
    # {config_name: {subject: avg_acc}}
    all_results = {}

    configs_to_run = {}
    for count in channel_counts:
        configs_to_run[f'top_{count}'] = RANKED_INDICES[:count]

    for config_name, ch_indices in configs_to_run.items():
        n_ch = 19 if ch_indices is None else len(ch_indices)
        ch_label = 'all 19' if ch_indices is None else \
                   str([CHANNEL_NAMES[i] for i in ch_indices])
        print(f"\n{'#'*70}")
        print(f"  ABLATION CONFIG: {config_name}  ({n_ch} channels)")
        print(f"  Channels: {ch_label}")
        print(f"{'#'*70}")

        config_path = os.path.join(ablation_root, config_name)
        os.makedirs(config_path, exist_ok=True)

        train_subject_dependent(
            dataset_conf, train_conf, config_path,
            channel_indices=ch_indices
        )

        # Read back the per-subject accuracies from the saved .npz files
        config_accs = {}
        for subj_dir in sorted(os.listdir(config_path)):
            npz_path = os.path.join(config_path, subj_dir, 'loso_results.npz')
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                config_accs[subj_dir.replace('subject_', '')] = float(data['avg_acc'])
        all_results[config_name] = config_accs

        # ---- Free RAM explicitly before next configuration ----
        import gc
        keras.backend.clear_session()
        gc.collect()

    # ── Print comparison table ────────────────────────────────────────────────
    subjects = sorted(set(s for res in all_results.values() for s in res))
    config_names = list(all_results.keys())
    header_cols = ['Subject'] + config_names + ['Best config']
    col_w = 14

    lines = []
    lines.append('\n' + '='*80)
    lines.append('ABLATION STUDY RESULTS — Accuracy per subject')
    lines.append('='*80)
    lines.append('  '.join(h.ljust(col_w) for h in header_cols))
    lines.append('-'*80)

    grand_avgs = {cn: [] for cn in config_names}
    for subj in subjects:
        row = [subj.ljust(col_w)]
        accs = {}
        for cn in config_names:
            v = all_results[cn].get(subj, None)
            accs[cn] = v
            row.append((f'{v:.4f}' if v is not None else 'N/A').ljust(col_w))
            if v is not None:
                grand_avgs[cn].append(v)
        best = max((cn for cn in config_names if accs[cn] is not None),
                   key=lambda cn: accs[cn], default='N/A')
        row.append(best)
        lines.append('  '.join(row))

    lines.append('-'*80)
    avg_row = ['MEAN'.ljust(col_w)]
    for cn in config_names:
        v = np.mean(grand_avgs[cn]) if grand_avgs[cn] else float('nan')
        avg_row.append(f'{v:.4f}'.ljust(col_w))
    avg_row.append('')
    lines.append('  '.join(avg_row))
    lines.append('='*80)

    # Accuracy drop from 19-ch baseline (if baseline was run)
    if 'full_19ch' in all_results:
        lines.append('\nAccuracy drop vs full 19-channel baseline:')
        baseline_avgs = {s: all_results['full_19ch'].get(s, None) for s in subjects}
    for cn in config_names:
        if cn == 'full_19ch':
            continue
        drops = []
        for s in subjects:
            base = baseline_avgs.get(s)
            abl = all_results[cn].get(s)
            if base is not None and abl is not None:
                drops.append(abl - base)
        mean_drop = np.mean(drops) if drops else float('nan')
        sign = '+' if mean_drop >= 0 else ''
        lines.append(f'  {cn:<12}: {sign}{mean_drop:.4f} ({sign}{mean_drop*100:.2f}%)')

    summary = '\n'.join(lines)
    print(summary)

    summary_path = os.path.join(ablation_root, 'ablation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\n  Summary saved to: {summary_path}")


# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DB-ATCNet Training on HALT Dataset")
    parser.add_argument("--single-run", action="store_true", 
                        help="Run standard 19-channel training (Baseline)")
    parser.add_argument("--ablation", action="store_true", 
                        help="Run multi-channel ablation study")
    parser.add_argument("--ablation-channels", type=int, nargs="+", default=[3, 5, 7, 10],
                        help="List of channel counts for ablation (e.g. --ablation-channels 3 5 7 10)")
    parser.add_argument("--gumbel-select", action="store_true",
                        help="Run Gumbel-softmax channel selection (Strypsteen & Bertrand 2021)")
    parser.add_argument("--gumbel-k", type=int, nargs="+", default=[3, 5, 7, 10],
                        help="Number of channels to select (e.g. --gumbel-k 3 5 7 10)")
    args = parser.parse_args()

    if args.gumbel_select:
        run_gumbel(k_channels_list=args.gumbel_k)
    elif args.ablation:
        run_ablation(channel_counts=args.ablation_channels)
    elif args.single_run:
        run()
    else:
        print("Please specify a run mode. Examples:")
        print("  python HALT_main.py --single-run")
        print("  python HALT_main.py --ablation")
        print("  python HALT_main.py --ablation --ablation-channels 2 4 8 16")
        print("  python HALT_main.py --gumbel-select")
        print("  python HALT_main.py --gumbel-select --gumbel-k 3 5 7 10")
        print("")
        parser.print_help()