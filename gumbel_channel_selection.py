"""
Gumbel-Softmax EEG Channel Selection Layer for Keras/TensorFlow.

Implements the concrete selector layer from:
    Strypsteen & Bertrand (2021), "End-to-end learnable EEG channel selection
    for deep neural networks with Gumbel-softmax", arXiv:2102.09050v3

The layer learns to select K channels from N input channels by jointly
training the selection parameters with the downstream network weights.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
from keras.callbacks import Callback


class GumbelChannelSelection(Layer):
    """Concrete selector layer using Gumbel-softmax for EEG channel selection.

    This layer is placed before a downstream EEG model. It contains K
    "selection neurons", each of which learns to select one channel from
    the N input channels.

    During training, each neuron samples soft weights from the Concrete
    distribution (Gumbel-softmax), producing a weighted combination of
    input channels. As the temperature β is annealed towards 0, these
    weights converge to one-hot vectors and the layer transitions from
    mixing to selecting channels.

    During inference, the softmax is replaced by a deterministic argmax,
    giving hard one-hot channel selection.

    Input shape:  (batch, 1, N, T)  — N input channels, T time samples
    Output shape: (batch, 1, K, T)  — K selected/mixed channels

    Parameters
    ----------
    n_channels : int
        Number of input channels (N).
    n_select : int
        Number of channels to select (K).
    initial_temperature : float
        Starting temperature for the Gumbel-softmax (default: 10.0).

    Reference equations from the paper:
        Eq. 2: Gumbel-softmax sampling
        Eq. 3: Selection probabilities
        Eq. 4: Test-time argmax selection
        Eq. 5: Duplicate-avoidance regularization
        Eq. 6: Normalized entropy for convergence monitoring
    """

    def __init__(self, n_channels, n_select, initial_temperature=10.0,
                 gumbel_lambda=0.2, ranked_init=None, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels  # N
        self.n_select = n_select      # K
        self.gumbel_lambda = gumbel_lambda
        # ranked_init: list of K channel indices in order of preference.
        # If provided, logits are warm-started so neuron k prefers channel ranked_init[k].
        self.ranked_init = list(ranked_init[:n_select]) if ranked_init is not None else None
        self.temperature = tf.Variable(
            initial_temperature, trainable=False, dtype=tf.float32,
            name='temperature'
        )
        self.threshold = tf.Variable(
            3.0, trainable=False, dtype=tf.float32, name='threshold'
        )

    def build(self, input_shape):
        # Learnable logits α (shape: N × K), initialized small random
        # Following the reference implementation: randn / 100
        self.logits = self.add_weight(
            name='selection_logits',
            shape=(self.n_channels, self.n_select),
            initializer=keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
        )
        super().build(input_shape)
        # Warm-start logits from ranked list if provided
        if self.ranked_init is not None:
            self._init_ranked_logits()

    def _init_ranked_logits(self):
        """Initialize logits so neuron k starts preferring ranked_init[k].

        A small positive bias (init_strength) is added to position
        (ranked_init[k], k), while all other entries stay near zero.
        This is a soft prior — gradients can still move the logits freely.
        """
        import numpy as np
        init_strength = 1.0
        values = np.zeros((self.n_channels, self.n_select), dtype=np.float32)
        for k, ch_idx in enumerate(self.ranked_init):
            values[ch_idx, k] = init_strength
        self.logits.assign(values)

    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs: Tensor of shape (batch, 1, N, T)
            training: Boolean flag for training vs inference mode.

        Returns:
            Tensor of shape (batch, 1, K, T)
        """
        if training:
            # --- Training: Gumbel-softmax sampling (Eq. 2) ---
            batch_size = tf.shape(inputs)[0]

            # Sample Gumbel noise: G = -log(-log(U)), U ~ Uniform(eps, 1-eps)
            eps = 1e-6
            u = tf.random.uniform(
                shape=(batch_size, self.n_channels, self.n_select),
                minval=eps, maxval=1.0 - eps
            )
            gumbel_noise = -tf.math.log(-tf.math.log(u))

            # Gumbel-softmax: w = softmax((logits + G) / β)
            # logits is (N, K), broadcast over batch
            # Result w is (batch, N, K) — per-sample weight vectors
            w = tf.nn.softmax(
                (self.logits[tf.newaxis, :, :] + gumbel_noise) / self.temperature,
                axis=1  # softmax over channels (N)
            )

            # Apply selection: z = w^T @ X
            # inputs: (batch, 1, N, T) → squeeze to (batch, N, T)
            x = inputs[:, 0, :, :]           # (batch, N, T)
            # w: (batch, N, K) → transpose to (batch, K, N)
            w_t = tf.transpose(w, perm=[0, 2, 1])  # (batch, K, N)
            # Matmul: (batch, K, N) @ (batch, N, T) = (batch, K, T)
            z = tf.matmul(w_t, x)

            # Add the duplicate-avoidance regularization loss (Eq. 5)
            self.add_loss(self.gumbel_lambda * self.regularization_loss())

            # Restore the singleton dimension: (batch, 1, K, T)
            return z[:, tf.newaxis, :, :]

        else:
            # --- Inference: deterministic argmax selection (Eq. 4) ---
            # Find the channel with the highest logit per selection neuron
            selected_indices = tf.argmax(self.logits, axis=0)  # shape (K,)

            # Gather selected channels from input
            # inputs: (batch, 1, N, T)
            x = inputs[:, 0, :, :]  # (batch, N, T)
            # Gather along the channel axis
            z = tf.gather(x, selected_indices, axis=1)  # (batch, K, T)
            return z[:, tf.newaxis, :, :]

    def get_selection_probabilities(self):
        """Get the probability matrix P (Eq. 3).

        Returns softmax of (logits/temperature) over the channel axis (dim=0),
        so each column k gives the probability distribution over channels for
        selection neuron k.

        Returns:
            numpy array of shape (N, K)
        """
        p = tf.nn.softmax(self.logits / self.temperature, axis=0)
        return p.numpy()

    def get_selected_channels(self):
        """Get the currently selected channel indices (test-time argmax).

        Returns:
            numpy array of shape (K,) with channel indices
        """
        return tf.argmax(self.logits, axis=0).numpy()

    def get_normalized_entropy(self):
        """Compute normalized entropy of each selection neuron (Eq. 6).

        H(α_k) = -1/log(N) * Σ_j p_jk * log(p_jk)

        where p = softmax(logits / temperature, axis=0).

        Returns:
            numpy array of shape (K,) with normalized entropies in [0, 1]
        """
        eps = 1e-10
        p = tf.clip_by_value(tf.nn.softmax(self.logits / self.temperature, axis=0), eps, 1.0)
        H = -tf.reduce_sum(p * tf.math.log(p), axis=0) / math.log(self.n_channels)
        return H.numpy()

    def regularization_loss(self):
        """Compute the duplicate-avoidance regularization (Eq. 5).

        L(P) = Σ_n ReLU(Σ_k p_nk - τ)

        where P is obtained by softmax(logits / temperature, axis=0), and τ is
        the current threshold.

        Returns:
            Scalar tensor — the regularization loss (without λ weighting)
        """
        eps = 1e-10
        # p_nk: probability of selection neuron k selecting channel n
        # self.logits shape: (N, K)
        # We want softmax over N for each K
        p = tf.nn.softmax(self.logits / self.temperature, axis=0) # shape (N, K)
        p = tf.clip_by_value(p, eps, 1.0)
        
        # row_sums[n] = Σ_k p_nk
        row_sums = tf.reduce_sum(p, axis=1)  # shape (N,)
        
        # ReLU penalty when row sum exceeds threshold
        penalty = tf.reduce_sum(tf.nn.relu(row_sums - self.threshold))
        return penalty

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_channels': self.n_channels,
            'n_select': self.n_select,
            'initial_temperature': float(self.temperature.numpy()),
            'gumbel_lambda': self.gumbel_lambda,
            'ranked_init': self.ranked_init,
        })
        return config


class GumbelAnnealingCallback(Callback):
    """Keras callback for temperature and threshold annealing.

    Implements the exponential decay schedules from the paper:
        β(t) = β_s * (β_e / β_s)^(t / T_anneal)
        τ(t) = τ_s * (τ_e / τ_s)^(t / T)

    Also monitors the normalized entropy of the selection neurons and
    logs the current channel selection each epoch.

    Parameters
    ----------
    gumbel_layer : GumbelChannelSelection
        The selection layer whose temperature and threshold to update.
    total_epochs : int
        Maximum number of training epochs (T).
    start_temp : float
        Starting temperature β_s (default: 10.0).
    end_temp : float
        Ending temperature β_e (default: 0.1).
    start_thresh : float
        Starting regularization threshold τ_s (default: 3.0).
    end_thresh : float
        Ending regularization threshold τ_e (default: 1.1).
    anneal_fraction : float
        Fraction of total epochs at which temperature reaches end_temp
        (default: 0.75, i.e. 3/4 of training as in paper).
    channel_names : list of str, optional
        Human-readable channel names for logging.
    verbose : bool
        Whether to print annealing info each epoch.
    """

    def __init__(self, gumbel_layer, total_epochs,
                 start_temp=10.0, end_temp=0.1,
                 start_thresh=3.0, end_thresh=1.0,
                 temp_anneal_fraction=0.75,
                 thresh_anneal_fraction=0.5,
                 channel_names=None, verbose=True):
        super().__init__()
        self.gumbel_layer = gumbel_layer
        self.total_epochs = total_epochs
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.start_thresh = start_thresh
        self.end_thresh = end_thresh
        self.temp_anneal_epoch = int(total_epochs * temp_anneal_fraction)
        self.thresh_anneal_epoch = 75
        self.channel_names = channel_names
        self.verbose = verbose

        # Track history for analysis
        self.temp_history = []
        self.thresh_history = []
        self.entropy_history = []
        self.selection_history = []

    def on_epoch_begin(self, epoch, logs=None):
        # --- Temperature schedule: β(t) = β_s * (β_e/β_s)^(t/T_temp_anneal) ---
        p_temp = min(epoch / max(self.temp_anneal_epoch, 1), 1.0)
        new_temp = self.start_temp * (self.end_temp / self.start_temp) ** p_temp

        # --- Threshold schedule: τ(t) = τ_s * (τ_e/τ_s)^(t/T_thresh_anneal) ---
        p_thresh = min(epoch / max(self.thresh_anneal_epoch, 1), 1.0)
        new_thresh = self.start_thresh * (self.end_thresh / self.start_thresh) ** p_thresh

        # Update the layer
        self.gumbel_layer.temperature.assign(new_temp)
        self.gumbel_layer.threshold.assign(new_thresh)

        self.temp_history.append(new_temp)
        self.thresh_history.append(new_thresh)

    def on_epoch_end(self, epoch, logs=None):
        # Monitor entropy and selection
        entropy = self.gumbel_layer.get_normalized_entropy()
        selected = self.gumbel_layer.get_selected_channels()
        mean_entropy = float(np.mean(entropy))

        self.entropy_history.append(mean_entropy)
        self.selection_history.append(selected.tolist())

        if self.verbose:
            ch_str = ', '.join(
                self.channel_names[i] if self.channel_names is not None
                else str(i)
                for i in selected
            )
            n_unique = len(set(selected.tolist()))
            print(
                f'  [Gumbel] Epoch {epoch+1}: '
                f'temp={float(self.gumbel_layer.temperature.numpy()):.4f}, '
                f'thresh={float(self.gumbel_layer.threshold.numpy()):.4f}, '
                f'mean_entropy={mean_entropy:.4f}, '
                f'selected=[{ch_str}] '
                f'({n_unique} unique)'
            )

    def get_summary(self):
        """Return a dict with the final annealing state."""
        return {
            'final_temperature': self.temp_history[-1] if self.temp_history else None,
            'final_threshold': self.thresh_history[-1] if self.thresh_history else None,
            'final_entropy': self.entropy_history[-1] if self.entropy_history else None,
            'final_selection': self.selection_history[-1] if self.selection_history else None,
            'entropy_history': self.entropy_history,
        }
