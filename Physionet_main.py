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
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold

import models
from Physionet_DataLoad import load_physionet, load_physionet_raw, standardize_data


# %%
def draw_learning_curves(history, results_path):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/ACC_' + '.png')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/LOSS_' + '.png')
    plt.show()
    plt.close()


def draw_confusion_matrix(cf_matrix, results_path):
    # Generate confusion matrix plot
    # display_labels = ['Left fist','Right fist']#2 class
    display_labels = ['Both feet', 'Left fist', 'Both fists', 'Right fist']#4 class

    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                  display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix ' )
    plt.savefig(results_path + '/subject_'+ '.png')
    plt.show()


def draw_performance_barChart(metric, label, results_path):
    fig, ax = plt.subplots()
    x = list(range(1, 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model ' + label + ' per subject')
    ax.set_ylim([0, 1])
    plt.savefig(results_path + '/' + label + '.png')

# %% Training
def train(dataset_conf, train_conf, results_path):
    print("[DEBUG] Entered train()")
    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz file (zipped archive) to store the accuracy and kappa metrics
    # for all runs (to calculate average accuracy/kappa over all runs)
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')

    # Get dataset paramters
    data_path = dataset_conf.get('data_path')
    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')  # Plot Learning Curves?
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')
    print('Training model: ', model_name)

    # Get the current 'IN' time to calculate the subject training time
    in_sub = time.time()
    # Initiating variables to save the best subject accuracy among multiple runs.
    BestSubjAcc = 0
    bestTrainingHistory = []
    # Get training and test data
    X_train, y_train_onehot,X_test, y_test_onehot = load_physionet(data_path)
    print(f"[DEBUG] Loaded data: X_train={None if X_train is None else getattr(X_train, 'shape', 'unknown')} ", end='')
    try:
        print(f" y_train={getattr(y_train_onehot, 'shape', 'unknown')} X_test={getattr(X_test, 'shape', 'unknown')} y_test={getattr(y_test_onehot, 'shape', 'unknown')}")
    except Exception:
        print()

    # Get the current 'IN' time to calculate the 'run' training time
    in_run = time.time()
    # Create folders and files to save trained models for all runs
    filepath = results_path + '/saved models/run-{}'.format(1)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath = filepath + '/subject-{}.weights.h5'.format(1)

    # Create the model
    model = getModel(model_name)
    # Compile and train the model
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                        save_best_only=True, save_weights_only=True, mode='max'),
        EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
    ]
    history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # Evaluate the performance of the trained model.
    # Here we load the Trained weights from the file saved in the hard
    # disk, which should be the same as the weights of the current model.
    model.load_weights(filepath)
    y_pred = model.predict(X_test).argmax(axis=-1)
    labels = y_test_onehot.argmax(axis=-1)
    acc= accuracy_score(labels, y_pred)
    kappa= cohen_kappa_score(labels, y_pred)

    # Get the current 'OUT' time to calculate the 'run' training time
    out_run = time.time()
    # Print & write performance measures for each run
    info = 'Subject: {}   Train no. {}   Time: {:.3f} m   '.format(1, 1,
                                                                   ((out_run - in_run) / 60))
    info = info + 'Test_acc: {:.5f}   Test_kappa: {:.5f}'.format(acc, kappa)
    print(info)
    log_write.write(info + '\n')
    # If current training run is better than previous runs, save the history.
    if (BestSubjAcc < acc):
        BestSubjAcc = acc
        bestTrainingHistory = history

    # Store the path of the best model among several runs
    best_run = np.argmax(acc)
    filepath = '/saved models/run-{}/subject-{}.weights.h5'.format(best_run + 1, 1) + '\n'
    best_models.write(filepath)
    # Get the current 'OUT' time to calculate the subject training time
    out_sub = time.time()
    # Print & write the best subject performance among multiple runs
    info = '----------\n'
    info = info + 'Subject: {}   best_run: {}   Time: {:.3f} m   '.format(1, best_run + 1,
                                                                          ((out_sub - in_sub) / 60))
    info = info + 'acc: {:.5f}   avg_acc: {:.5f} +- {:.5f}   '.format(acc, np.average(acc),
                                                                      np.std(acc))
    info = info + 'kappa: {:.5f}   avg_kappa: {:.5f} +- {:.5f}'.format(kappa,
                                                                       np.average(kappa),
                                                                       np.std(kappa))
    info = info + '\n----------'
    print(info)
    log_write.write(info + '\n')
    # Plot Learning curves
    if (LearnCurves == True):
        print('Plot Learning Curves ....... ')
        draw_learning_curves(bestTrainingHistory, results_path)

    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.3f} h   '.format((out_exp - in_exp) / (60 * 60))
    print(info)
    log_write.write(info + '\n')

    # Store the accuracy and kappa metrics as arrays for all runs into a .npz
    # file format, which is an uncompressed zipped archive, to calculate average
    # accuracy/kappa over all runs.
    np.savez(perf_allRuns, acc=acc, kappa=kappa)

    # Close open files
    best_models.close()
    log_write.close()
    perf_allRuns.close()

    # Evaluation
    model = getModel(model_name)
    evaluate(model, dataset_conf, results_path, X_test, y_test_onehot)

# %% Evaluation
# Renamed from `test` to `evaluate` to avoid pytest treating this as a test function.
def evaluate(model, dataset_conf, results_path,X_test,y_test_onehot, allRuns=True):
     # Open the  "Log" file to write the evaluation results
     log_write = open(results_path + "/log.txt", "a")
     # Open the file that stores the path of the best models among several random runs.
     best_models = open(results_path + "/best models.txt", "r")

     # Get dataset paramters
     n_classes = dataset_conf.get('n_classes')

     # Initialize variables
     cf_matrix = np.zeros([n_classes, n_classes])

     # Calculate the average performance (average accuracy and K-score) for
     # all runs (experiments) for each subject.
     if (allRuns):
         # Load the test accuracy and kappa metrics as arrays for all runs from a .npz
         # file format, which is an uncompressed zipped archive, to calculate average
         # accuracy/kappa over all runs.
         perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
         perf_arrays = np.load(perf_allRuns)
         acc_allRuns = perf_arrays['acc']
         kappa_allRuns = perf_arrays['kappa']

         # Load data
         X_test, y_test_onehot = X_test, y_test_onehot

         # Load the best model out of multiple random runs (experiments).
         filepath = best_models.readline()
         model.load_weights(results_path + filepath[:-1])

         # Predict MI task
         y_pred = model.predict(X_test).argmax(axis=-1)

         # Calculate accuracy and K-score
         labels = y_test_onehot.argmax(axis=-1)
         acc_bestRun = accuracy_score(labels, y_pred)
         kappa_bestRun = cohen_kappa_score(labels, y_pred)
         # Calculate and draw confusion matrix
         cf_matrix[:, :] = confusion_matrix(labels, y_pred, normalize='pred')
         draw_confusion_matrix(cf_matrix[ :, :], results_path)

         # Print & write performance measures for each subject
         info = 'Subject: {}   best_run: {:2}  '.format(1,
                                                        (filepath[filepath.find('run-') + 4:filepath.find('/sub')]))
         info = info + 'acc: {:.5f}   kappa: {:.5f}   '.format(acc_bestRun, kappa_bestRun)
         if (allRuns):
             info = info + 'avg_acc: {:.5f} +- {:.5f}   avg_kappa: {:.5f} +- {:.5f}'.format(
                 np.average(acc_allRuns), np.std(acc_allRuns),
                 np.average(kappa_allRuns), np.std(kappa_allRuns))
         print(info)
         log_write.write('\n' + info)

         # t-sne visualization
         y_pred1 = model.predict(X_test)
         tsne = TSNE(n_components=2, random_state=33, verbose=1, n_iter=1000)
         tsne_results2 = tsne.fit_transform(y_pred1)
         color_map = np.argmax(y_test_onehot, axis=1)
         print(color_map.shape)
         plt.figure(figsize=(8, 8))
         display_labels = ['Left fist', 'Right fist']
         for cl in range(2):#2 class
         # for cl in range(4):#4 class
             indices = np.where(color_map == cl)
             print(indices[0].shape)
             indices = indices[0]
             plt.scatter(tsne_results2[indices, 0], tsne_results2[indices, 1], s=8, label=display_labels[cl])
         plt.legend()
         plt.show()
         plt.savefig(results_path + '/t-sne_allsubject' + '.png')

     # Print & write the average performance measures for all subjects
     info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.5f}   Kappa = {:.5f}\n'.format(1,
         np.average(acc_bestRun), np.average(kappa_bestRun))

     print(info)
     log_write.write(info)

     # Close open files
     log_write.close()


# %%
def getModel(model_name):
    # Select the model
    if (model_name == 'DB_ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.DB_ATCNet(
            # Dataset parameters
            n_classes=4,
            in_chans=64,
            in_samples=640,

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
            n_classes=4,
            in_chans=64,
            in_samples=640,
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
        model = models.TCNet_Fusion(n_classes=4)
    elif (model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes=4)
    elif (model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes=4)
    elif (model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps=640, n_features=64, n_outputs=4)
    elif (model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes=4, Chans=64, Samples=640)
    elif (model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes=4, Chans=64, Samples=640)
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
    fallback = "/root/autodl-tmp/physionet"
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
    # Resolve dataset path (checks env var PHYSIONET_DATA_DIR, then ./files, then fallback)
    data_path = get_data_path()
    print(f"[DEBUG] Resolved data_path={data_path}")

    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)  # Create a new directory if it does not exist

    # Set dataset paramters
    dataset_conf = {'n_classes': 4, 'n_channels': 64, 'data_path': data_path}
    # Set training hyperparamters
    train_conf = {
        'batch_size': 32, 
        'epochs': 500, 
        'patience': 100, 
        'lr': 0.0009,   
        'LearnCurves': True, 
        'model': 'DB_ATCNet',
        'n_folds': 5  # Number of folds for cross-validation
    }

    # Train the model using stratified k-fold cross-validation
    print("[DEBUG] Starting Stratified K-Fold Cross-Validation")
    train_kfold(dataset_conf, train_conf, results_path)


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
    n_folds = train_conf.get('n_folds', 5)
    
    print(f'Training model: {model_name} with {n_folds}-fold stratified cross-validation')
    log_write.write(f'Model: {model_name}, {n_folds}-fold stratified cross-validation\n')
    log_write.write('='*80 + '\n')
    
    # Load all data (without train/test split)
    print("[DEBUG] Loading all data...")
    X_all, y_all_onehot, y_labels, n_channels = load_physionet_raw(data_path)
    print(f"[DEBUG] Data loaded: X_all={X_all.shape}, y_all={y_all_onehot.shape}")
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_accuracies = []
    fold_kappas = []
    fold_histories = []
    
    # Iterate through folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_labels), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_folds}")
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
        plt.figure(figsize=(8, 6))
        display_labels = ['Both feet', 'Left fist', 'Both fists', 'Right fist']
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=display_labels)
        disp.plot()
        disp.ax_.set_xticklabels(display_labels, rotation=12)
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.savefig(fold_path + '/confusion_matrix.png')
        plt.close()
        
        # Plot learning curves for this fold
        if LearnCurves:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title(f'Fold {fold} - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title(f'Fold {fold} - Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(fold_path + '/learning_curves.png')
            plt.close()
        
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
    Number of Folds: {n_folds}
    
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
    x = np.arange(n_folds)
    width = 0.35
    
    plt.bar(x - width/2, fold_accuracies, width, label='Accuracy', color='steelblue')
    plt.bar(x + width/2, fold_kappas, width, label='Kappa', color='coral')
    
    plt.axhline(y=avg_acc, color='steelblue', linestyle='--', label=f'Avg Acc: {avg_acc:.3f}')
    plt.axhline(y=avg_kappa, color='coral', linestyle='--', label=f'Avg Kappa: {avg_kappa:.3f}')
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title(f'{model_name} - {n_folds}-Fold Cross-Validation Results')
    plt.xticks(x, [f'Fold {i+1}' for i in range(n_folds)])
    plt.legend(loc='lower right')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(results_path + '/kfold_summary.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {results_path}")


# %%
if __name__ == "__main__":
    run()