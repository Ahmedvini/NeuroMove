import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import HALT_DataLoad
from models import DB_ATCNet

# 1. Setup GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/ezzo/anaconda3/lib/python3.13/site-packages/nvidia/cuda_nvcc'

# Disable oneDNN optimizations to ensure consistent results across runs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration
n_classes = 2
in_chans = 19
in_samples = 600
data_path = "./HALT"
results_dir = "./results"
vis_base_dir = os.path.join(results_dir, "visualizations")
TARGET_SUBJECTS = ['A', 'B', 'C', 'E', 'F']

# Correct labels from HALT_DataLoad (cls 2=Right Hand, cls 4=Left Leg)
# Mapping: 2 -> 0, 4 -> 1
class_names = ['Right Hand', 'Left Leg']
# Channel names mapping for 19 leads selected in HALT_DataLoad
channel_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

def load_subject_data(subject_id, path):
    """Load data for a specific subject, standardize all trials, and return the full set."""
    x_raw, y_raw = HALT_DataLoad.load_halt_subject_data(subject_id, path)
    if x_raw.size == 0:
        raise ValueError(f"No HALT data found for Subject {subject_id}.")

    x_raw = x_raw.reshape(x_raw.shape[0], 1, x_raw.shape[1], -1)
    y_one_hot = HALT_DataLoad.to_one_hot(y_raw, by_sub=False)

    # Standardize entire subject set (fit per-channel on full data)
    for j in range(x_raw.shape[2]):
        scaler = StandardScaler()
        scaler.fit(x_raw[:, 0, j, :])
        x_raw[:, 0, j, :] = scaler.transform(x_raw[:, 0, j, :])

    return x_raw, y_one_hot

# 4. Helper Function: Plot Filters (Kernels & Channels)
def plot_filters(layer, layer_name, detailed_dir):
    weights = layer.get_weights()
    if not weights:
        return
    
    W = weights[0] # Shape depends on layer type
    
    # 4a. Temporal Convolution Kernels: (kernel_size, 1, 1, filters) or similar
    # In DB-ATCNet, initial Conv2D has shape (64, 1, 1, 16)
    if isinstance(layer, keras.layers.Conv2D) and W.shape[1] == 1:
        # It's a temporal filter. W shape: (time_steps, 1, in_channels, out_channels)
        time_steps = W.shape[0]
        out_filters = W.shape[3]
        
        # Plot up to 16 filters
        n_plots = min(out_filters, 16)
        cols = 4
        rows = int(np.ceil(n_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        fig.suptitle(f'Temporal Kernels: {layer_name}', fontsize=16)
        axes = axes.flatten()
        
        for i in range(n_plots):
            # Plot the 1D waveform learned by the kernel
            kernel_1d = W[:, 0, 0, i]
            axes[i].plot(kernel_1d, color='blue')
            axes[i].set_title(f'Filter {i+1}')
            axes[i].set_xlabel('Filter Time Window (Samples)')
            axes[i].set_ylabel('Weight Magnitude')
            axes[i].grid(True)
        
        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(detailed_dir, f"kernels_{layer_name.replace('/', '_')}.png"))
        plt.close()
        
    # 4b. Spatial/Channel Filters: DepthwiseConv2D (1, channels, 1, depth_multiplier)
    elif isinstance(layer, keras.layers.DepthwiseConv2D) and W.shape[0] == 1:
        # It's a spatial filter. W shape: (1, 19 channels, 1 in_depth, out_depth_multiplier)
        num_channels = W.shape[1]
        depth_mult = W.shape[3]
        
        current_channel_names = globals().get('channel_names', ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'])
        
        # We will plot a bar chart showing the absolute magnitude/importance of each channel
        # for a specific spatial filter branch.
        n_plots = min(depth_mult, 4)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        if n_plots == 1: axes = [axes]
        fig.suptitle(f'Spatial Channel Importance: {layer_name}', fontsize=16)
        
        for i in range(n_plots):
            spatial_weights = np.abs(W[0, :, 0, i]) # Magnitude of weights for the 19 channels
            
            axes[i].bar(current_channel_names, spatial_weights, color='coral')
            axes[i].set_title(f'Spatial Map {i+1} (Absolute Weights)')
            axes[i].set_xlabel('EEG Channel Sensors')
            axes[i].set_ylabel('Weight Magnitude')
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        plt.savefig(os.path.join(detailed_dir, f"channels_{layer_name.replace('/', '_')}.png"))
        plt.close()

    # 4c. Deep Temporal Kernels (Conv1D in TCFN): (kernel_size, in_channels, filters)
    elif isinstance(layer, keras.layers.Conv1D):
        out_filters = W.shape[2]
        n_plots = min(out_filters, 16)
        cols = 4
        rows = int(np.ceil(n_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        fig.suptitle(f'Deep 1D Temporal Kernels: {layer_name}', fontsize=16)
        axes = axes.flatten()
        
        for i in range(n_plots):
            # Plot the sum across input channels to show the general temporal shape
            kernel_1d = np.mean(W[:, :, i], axis=1) 
            axes[i].plot(kernel_1d, color='green')
            axes[i].set_title(f'Filter {i+1}')
            axes[i].set_xlabel('Filter Time Window (Samples)')
            axes[i].set_ylabel('Weight Magnitude')
            axes[i].grid(True)
        
        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(detailed_dir, f"kernels_1d_{layer_name.replace('/', '_')}.png"))
        plt.close()

for TARGET_SUBJECT in TARGET_SUBJECTS:
    print(f"\n\n{'='*60}")
    print(f"   PROCESSING SUBJECT: {TARGET_SUBJECT}")
    print(f"{'='*60}")

    vis_dir = os.path.join(vis_base_dir, f"subject_{TARGET_SUBJECT}")
    detailed_dir = os.path.join(vis_dir, "detailed")
    presentation_dir = os.path.join(vis_dir, "presentation")

    print(f"[INFO] Creating directories for {TARGET_SUBJECT}...")
    for d in [vis_dir, detailed_dir, presentation_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Get the Model and Load Best Weights
    print(f"[INFO] Rebuilding DB_ATCNet architecture for Subject {TARGET_SUBJECT}...")
    model = DB_ATCNet(
        n_classes=n_classes,
        in_chans=in_chans,
        in_samples=in_samples,
        eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
        drop1=0.35, depth1=2, depth2=4, n_windows=5, attention='mha',
        tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
        drop2=0.1, drop3=0.15, drop4=0.15, tcn_activation='elu'
    )

    # Look for the best weights for this subject
    weights_path = None
    subj_dir = os.path.join(results_dir, f"subject_{TARGET_SUBJECT}")

    if os.path.exists(subj_dir):
        folders = sorted([f for f in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, f))])
        for f in folders:
            candidate = os.path.join(subj_dir, f, "best_model.weights.h5")
            if os.path.exists(candidate):
                weights_path = candidate
                break

    if weights_path is None:
        print(f"[ERROR] Could not find best_model.weights.h5 for Subject {TARGET_SUBJECT} in results/subject_{TARGET_SUBJECT}/*/.")
        continue

    print(f"[INFO] Loading Subject {TARGET_SUBJECT} weights from {weights_path}")
    model.load_weights(weights_path)


    # 3. Load a sample from validation/test data for EACH class to pass through the network
    print(f"[INFO] Loading Subject {TARGET_SUBJECT} from HALT dataset...")
    X_data, y_data = load_subject_data(TARGET_SUBJECT, data_path)
    print(f"[INFO] Subject {TARGET_SUBJECT} data shape: {X_data.shape}")

    label_indices = np.argmax(y_data, axis=1)
    label_counts = {class_names[i]: int((label_indices == i).sum()) for i in range(len(class_names))}
    print(f"[INFO] Subject {TARGET_SUBJECT} class counts: {label_counts}")

    # Select exactly one CORRECTLY classified sample per class
    dataset_sample_indices = {}
    selected_samples = []
    
    # We need to predict on all data to find correct samples, or just iterate
    print(f"[INFO] Searching for correctly classified samples for Subject {TARGET_SUBJECT}...")
    all_preds = model.predict(X_data, batch_size=32, verbose=0)
    all_pred_labels = np.argmax(all_preds, axis=1)

    for class_idx, class_name in enumerate(class_names):
        # Find indices where ground truth matches class_idx AND prediction matches ground truth
        correct_positions = np.where((label_indices == class_idx) & (all_pred_labels == class_idx))[0]
        
        if len(correct_positions) == 0:
            print(f"[WARNING] Could not find any CORRECTLY classified samples for class: {class_name} for Subject {TARGET_SUBJECT}. Skipping class.")
            continue
            
        chosen_idx = int(correct_positions[0])  # Use the first correctly classified sample
        dataset_sample_indices[class_name] = chosen_idx
        selected_samples.append(X_data[chosen_idx:chosen_idx+1])
        print(f"  - Found correct sample for {class_name} at index {chosen_idx} (Confidence: {all_preds[chosen_idx][class_idx]:.4f})")

    if not selected_samples:
        print(f"[ERROR] No correctly classified classes found for Subject {TARGET_SUBJECT}. Skipping.")
        continue

    # Build a compact batch with one correctly classified sample per class
    selected_batch = np.concatenate(selected_samples, axis=0)
    current_sample_indices = {name: i for i, name in enumerate(dataset_sample_indices.keys())}

    print(f"[INFO] Using verified correct samples for visualization: {current_sample_indices}")

    print(f"[INFO] Extracting kernels and channel weights for {TARGET_SUBJECT} (limited to key layers)...")
    target_layers = []
    found_conv2d = False
    found_dw = False
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, keras.Model):
            for sub_idx, sub_layer in enumerate(layer.layers):
                if not found_conv2d and isinstance(sub_layer, keras.layers.Conv2D):
                    target_layers.append((sub_layer, f"{layer.name}_{sub_layer.name}"))
                    found_conv2d = True
                if not found_dw and isinstance(sub_layer, keras.layers.DepthwiseConv2D):
                    target_layers.append((sub_layer, f"{layer.name}_{sub_layer.name}"))
                    found_dw = True
        elif isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D, keras.layers.Conv1D)) and len(target_layers) < 4:
            target_layers.append((layer, layer.name))

    for layer, name in target_layers:
        plot_filters(layer, name, detailed_dir)

    # 5. Extract Activations (Feature Maps) at various levels
    activation_targets = []
    # Just grab 3 key layers for summary
    for name in [t[1] for t in target_layers]:
        if "conv2d" in name.lower() and "depthwise" not in name.lower() and len(activation_targets) == 0:
            activation_targets.append(name)
        elif "depthwise" in name.lower() and len(activation_targets) == 1:
            activation_targets.append(name)
    
    # Add one high-level TCN layer manually if possible
    for layer in model.layers:
        if "tcn" in layer.name.lower() and hasattr(layer, 'output'):
            activation_targets.append(layer.name)
            break

    top_level_outputs = [layer.output for layer in model.layers if hasattr(layer, 'output')]
    activation_model = Model(inputs=model.input, outputs=top_level_outputs)

    # Loop over available classes (limit to 2 for speed)
    for class_name, idx in list(current_sample_indices.items())[:2]:
        print(f"\n[INFO] --- EXTRACTING FEATURES FOR CLASS: {class_name} ({TARGET_SUBJECT}) ---")
        
        class_dir = os.path.join(presentation_dir, class_name.replace(' ', '_'))
        levels_dir = os.path.join(class_dir, 'model_levels')
        channels_dir = os.path.join(class_dir, 'channels_and_temporal')
        attention_dir = os.path.join(class_dir, 'attention_focus')
        os.makedirs(levels_dir, exist_ok=True)
        os.makedirs(channels_dir, exist_ok=True)
        os.makedirs(attention_dir, exist_ok=True)

        sample_x = selected_batch[idx:idx+1]
        preds = activation_model.predict(sample_x, verbose=0)
        
        # 5a. Detailed Activation Heatmaps (Only for top level summary, not every layer)
        # To save time, we only save the specific layers used in the summary
        for i, layer in enumerate(model.layers):
            if layer.name in activation_targets:
                act = preds[i] if isinstance(preds, list) else preds
                act_2d = np.squeeze(act)
                if len(act_2d.shape) > 2:
                    act_2d = np.mean(act_2d, axis=1)
                
                if len(act_2d.shape) == 2:
                    plt.figure(figsize=(10, 4))
                    plt.imshow(act_2d.T, aspect='auto', cmap='viridis', interpolation='nearest')
                    plt.title(f"Key Activation: {layer.name} ({class_name})")
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(os.path.join(detailed_dir, f"{class_name.replace(' ', '_')}_key_activation_{layer.name}.png"))
                    plt.close()

        # 6. Generate a Unified Presentation Summary for THIS class
        print(f"[INFO] Generating consolidated Presentation Summary for {class_name}...")
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        fig.suptitle(f'DB-ATCNet Internal Feature Progression\nClass: {class_name} (Subject {TARGET_SUBJECT})', fontsize=20, fontweight='bold')

        raw_signal = np.squeeze(sample_x) 
        time_axis = np.linspace(0, 3.0, 600)  
        current_channel_names = globals().get('channel_names', ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'])

        spacing = np.max(np.abs(raw_signal)) * 1.5 
        for ch in range(19):
            axes[0].plot(time_axis, raw_signal[ch, :] + (ch * spacing), color='cyan', linewidth=0.5)
        
        axes[0].set_title('Raw Input EEG Signal', fontsize=14)
        axes[0].set_yticks(np.arange(19) * spacing)
        axes[0].set_yticklabels(current_channel_names)
        axes[0].set_xlabel('Time (Seconds)', fontsize=10)
        axes[0].set_ylabel('EEG Sensors', fontsize=10)
        axes[0].set_ylim(-spacing, 19 * spacing)
        axes[0].grid(True, linestyle=':', alpha=0.6)

        low_act, mid_act, high_act = None, None, None
        for i, layer in enumerate(model.layers):
            if not hasattr(layer, 'output'): continue
            act = preds[i] if isinstance(preds, list) else preds
            # Preds already contains only one sample
            act_for_class = np.squeeze(act) if act.shape[0] == 1 else act[0]
            
            if len(act_for_class.shape) < 2 or np.prod(act_for_class.shape) <= 10: continue
            
            act_2d = act_for_class
            if len(act_2d.shape) < 2: continue
            
            if len(act_2d.shape) > 2: 
                act_2d = np.mean(act_2d, axis=1)
            
            if len(act_2d.shape) != 2: continue
            if min(act_2d.shape) < 2: continue
            
            t_dim, f_dim = act_2d.shape
            
            if low_act is None and t_dim >= 500:
                low_act = act_2d.T
            elif mid_act is None and 5 < t_dim < 100 and f_dim >= 16:
                mid_act = act_2d.T
            elif t_dim <= 10 and f_dim >= 16:
                high_act = act_2d.T

        if low_act is not None:
            low_t = low_act.shape[1]
            im1 = axes[1].imshow(low_act, aspect='auto', cmap='viridis', origin='lower')
            axes[1].set_title(f'Low-Level Feature Map', fontsize=14)
            axes[1].set_ylabel('Rows = 16 Distinct Filters', fontsize=10)
            axes[1].set_xlabel(f'Columns = Time Window Progressions (0 to {low_t})', fontsize=10)

        if mid_act is not None:
            mid_t = mid_act.shape[1]
            im2 = axes[2].imshow(mid_act, aspect='auto', cmap='plasma', origin='lower')
            axes[2].set_title('Mid-Level Feature Map', fontsize=14)
            axes[2].set_ylabel('Rows = 32 Spatial Combinations', fontsize=10)
            axes[2].set_xlabel(f'Columns = Time Window Progressions (0 to {mid_t})', fontsize=10)

        if high_act is not None:
            high_t = high_act.shape[1]
            im3 = axes[3].imshow(high_act, aspect='auto', cmap='inferno', origin='lower')
            axes[3].set_title('High-Level Feature Map', fontsize=14)
            axes[3].set_ylabel('Rows = 32 Deep Temporal Blocks', fontsize=10)
            axes[3].set_xlabel(f'Columns = Time Window Progressions (0 to {high_t})', fontsize=10)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.6])
        if high_act is not None:
            cbar = fig.colorbar(im3, cax=cbar_ax)
            cbar.set_label('Activation Strength', rotation=270, labelpad=15)

        plt.tight_layout(rect=[0, 0.03, 0.9, 0.96])
        plt.savefig(os.path.join(levels_dir, "summary_overview.png"), dpi=300)
        plt.close()

        # 5b. Class-Specific Spatial Channel Importance Bar Chart
        print(f"[INFO] Generating Class-Specific Spatial Importance for {class_name}...")
        total_importance = np.zeros(19)
        spatial_layers_found = 0
        current_input_act = None
        
        for i, layer in enumerate(model.layers):
            if not hasattr(layer, 'output'): continue
            act = preds[i] if isinstance(preds, list) else preds
            if len(act.shape) == 4 and act.shape[2] == 19:
                current_input_act = act[0]
                
            if isinstance(layer, keras.layers.DepthwiseConv2D):
                W = layer.get_weights()[0]
                if current_input_act is not None:
                    act_4d = np.expand_dims(current_input_act, axis=-1)
                    importance_map = np.abs(act_4d * W)
                    channel_imp = np.mean(importance_map, axis=0)
                    channel_imp = np.sum(channel_imp, axis=(1, 2))
                    total_importance += channel_imp
                    spatial_layers_found += 1

        if spatial_layers_found > 0:
            total_importance /= spatial_layers_found
            fig_sp, ax_sp = plt.subplots(figsize=(12, 6))
            bars = ax_sp.bar(current_channel_names, total_importance, color='tomato', edgecolor='white')
            ax_sp.set_title(f'Topographical Channel Importance: {class_name}', fontsize=16, fontweight='bold')
            ax_sp.set_xlabel('EEG Channel Sensors', fontsize=12)
            ax_sp.set_ylabel('Aggregated Contribution Score', fontsize=12)
            ax_sp.grid(axis='y', linestyle='--', alpha=0.5)
            top_indices = np.argsort(total_importance)[-3:][::-1]
            for idx_top in top_indices:
                bars[idx_top].set_color('gold')
                bars[idx_top].set_edgecolor('black')
                bars[idx_top].set_linewidth(2)
            for i_bar, v in enumerate(total_importance):
                ax_sp.text(i_bar, v, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(channels_dir, "spatial_importance.png"), dpi=300)
            plt.close()

        # 5c. Temporal Kernels Presentation Plot
        print(f"[INFO] Generating Temporal Kernels for {class_name}...")
        temporal_layer = None
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D):
                temporal_layer = layer
                break
        if temporal_layer:
            W = temporal_layer.get_weights()[0]
            out_filters = min(W.shape[3], 4)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Learned Temporal Filters (Subject {TARGET_SUBJECT})', fontsize=16, fontweight='bold', y=0.98)
            axes = axes.flatten()
            for i in range(out_filters):
                axes[i].plot(W[:, 0, 0, i], color='cyan', linewidth=2)
                axes[i].set_title(f'Temporal Kernel Pattern #{i+1}', fontsize=12)
                axes[i].grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.88])
            plt.savefig(os.path.join(channels_dir, "temporal_kernels.png"), dpi=300)
            plt.close()

        # 5d. TCN Activation Focus
        print(f"[INFO] Generating TCN Activation Focus for {class_name}...")
        if high_act is not None:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(f'TCN Activation Focus: {class_name} (Subject {TARGET_SUBJECT})', fontsize=16, fontweight='bold', y=0.96)
            for ch in range(19):
                axes[0].plot(time_axis, raw_signal[ch, :] + (ch * spacing), color='cyan', linewidth=0.5)
            axes[0].set_xlim([0, 3.0])
            axes[0].set_yticks([])
            axes[0].set_ylabel('19 Electrode Channels', fontsize=12)
            
            focus_curve = np.mean(high_act, axis=0)
            focus_curve = (focus_curve - np.min(focus_curve)) / (np.max(focus_curve) - np.min(focus_curve) + 1e-9)
            downsampled_time = np.linspace(0, 3.0, high_act.shape[1])
            axes[1].fill_between(downsampled_time, 0, focus_curve, color='magenta', alpha=0.6)
            axes[1].plot(downsampled_time, focus_curve, color='white', linewidth=2)
            axes[1].set_xlim([0, 3.0])
            axes[1].set_ylim([0, 1.1])
            axes[1].set_xlabel('Time (Seconds)', fontsize=12, fontweight='bold')
            axes[1].grid(True, linestyle='--', alpha=0.4)
            
            max_idx = np.argmax(focus_curve)
            peak_time = downsampled_time[max_idx]
            axes[0].axvline(peak_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
            axes[1].axvline(peak_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            plt.savefig(os.path.join(attention_dir, "tcn_activation_focus.png"), dpi=300)
            plt.close()

        # 5e. MHA Attention Scores
        print(f"[INFO] Extracting real MHA Q·K attention scores for {class_name}...")
        mha_windows = []
        temp_mha_input_idx = None
        for i, layer in enumerate(model.layers):
            if not hasattr(layer, 'output'): continue
            if isinstance(layer, keras.layers.LayerNormalization): temp_mha_input_idx = i
            if isinstance(layer, keras.layers.MultiHeadAttention):
                if temp_mha_input_idx is not None: mha_windows.append((layer, temp_mha_input_idx))
        
        if mha_windows:
            mha_layer, mha_input_idx = mha_windows[-1] 
            mha_input = preds[mha_input_idx]
            _, attn_scores = mha_layer(tf.constant(mha_input), tf.constant(mha_input), return_attention_scores=True, training=False)
            attn_np = attn_scores.numpy()[0]
            num_heads = attn_np.shape[0]
            seq_len = attn_np.shape[1]
            fig, axes = plt.subplots(1, num_heads + 1, figsize=(6 * (num_heads + 1), 5))
            fig.suptitle(f'MHA Scores (Subject {TARGET_SUBJECT}): {class_name}', fontsize=16, fontweight='bold', y=1.05)
            for h in range(num_heads):
                im = axes[h].imshow(attn_np[h], cmap='hot', vmin=0, vmax=1, origin='upper')
                axes[h].set_title(f'Head {h+1}', fontsize=14)
                axes[h].set_xticks(range(seq_len))
                axes[h].set_yticks(range(seq_len))
                for qi in range(seq_len):
                    for ki in range(seq_len):
                        val = attn_np[h, qi, ki]
                        color = 'black' if val > 0.5 else 'white'
                        axes[h].text(ki, qi, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
            avg_attn = np.mean(attn_np, axis=0)
            im = axes[num_heads].imshow(avg_attn, cmap='hot', vmin=0, vmax=1, origin='upper')
            axes[num_heads].set_title('Averaged Across Heads', fontsize=14)
            axes[num_heads].set_xticks(range(seq_len))
            axes[num_heads].set_yticks(range(seq_len))
            for qi in range(seq_len):
                for ki in range(seq_len):
                    val = avg_attn[qi, ki]
                    color = 'black' if val > 0.5 else 'white'
                    axes[num_heads].text(ki, qi, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
            plt.tight_layout()
            plt.savefig(os.path.join(attention_dir, "mha_attention_scores.png"), dpi=300, bbox_inches='tight')
            plt.close()




print("[INFO] Visualization script completed successfully.")
print(f"       Exhaustive analysis saved to: {detailed_dir}")
print(f"       Presentation summary saved to: {presentation_dir}")
