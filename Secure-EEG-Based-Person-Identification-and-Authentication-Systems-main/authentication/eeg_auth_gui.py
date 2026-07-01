"""
EEG Biometric Security Suite - Professional Edition
A high-end security terminal for neural authentication
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import mne
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
import tensorflow as tf
import threading
import time
from datetime import datetime

class SecuritySuiteGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Biometric Security Terminal")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0a0e1a')
        
        # State variables
        self.model = None
        self.fingerprint_model = None
        self.enrolled_users = {}
        self.dataset_path = os.environ.get("EEG_DATASET_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "files"))
        self.auth_state = "STANDBY"  # STANDBY, SCANNING, AUTHENTICATED, DENIED
        
        # Configuration
        self.config = {
            'RUN_ID': 1,
            'TARGET_CHANNELS': ['Oz', 'T7', 'Cz'],
            'WINDOW_SIZE': 160,
            'STRIDE': 4,
            'SAMPLE_RATE': 160,
            'THRESHOLD': 0.8148
        }
        
        # Colors
        self.colors = {
            'bg_dark': '#0a0e1a',
            'bg_panel': '#141b2d',
            'accent_cyan': '#00f0ff',
            'accent_green': '#00ff88',
            'accent_orange': '#ff6b35',
            'accent_red': '#ff3366',
            'text_primary': '#ffffff',
            'text_secondary': '#8892b0',
            'border': '#1e3a5f'
        }
        
        self.create_ui()
        
    def create_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(fill='both', expand=True)
        
        # Top bar - Security header
        self.create_header(main_frame)
        
        # Content area
        content = tk.Frame(main_frame, bg=self.colors['bg_dark'])
        content.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left: Control Terminal
        left_panel = tk.Frame(content, bg=self.colors['bg_panel'], width=400)
        left_panel.pack(side='left', fill='y', padx=(0, 15))
        left_panel.pack_propagate(False)
        self.create_control_terminal(left_panel)
        
        # Center: Main Display
        center_panel = tk.Frame(content, bg=self.colors['bg_dark'])
        center_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        self.create_main_display(center_panel)
        
        # Right: System Monitor
        right_panel = tk.Frame(content, bg=self.colors['bg_panel'], width=350)
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)
        self.create_system_monitor(right_panel)
        
    def create_header(self, parent):
        header = tk.Frame(parent, bg='#0d1117', height=100)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # Logo and title
        title_frame = tk.Frame(header, bg='#0d1117')
        title_frame.pack(expand=True)
        
        tk.Label(
            title_frame,
            text="⬡",
            font=('Consolas', 40),
            bg='#0d1117',
            fg=self.colors['accent_cyan']
        ).pack(side='left', padx=(0, 15))
        
        title_container = tk.Frame(title_frame, bg='#0d1117')
        title_container.pack(side='left')
        
        tk.Label(
            title_container,
            text="NEURAL BIOMETRIC",
            font=('Consolas', 24, 'bold'),
            bg='#0d1117',
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        tk.Label(
            title_container,
            text="SECURITY TERMINAL v2.0",
            font=('Consolas', 12),
            bg='#0d1117',
            fg=self.colors['accent_cyan']
        ).pack(anchor='w')
        
        # Status indicator
        self.header_status = tk.Label(
            header,
            text="● SYSTEM READY",
            font=('Consolas', 11, 'bold'),
            bg='#0d1117',
            fg=self.colors['accent_green']
        )
        self.header_status.place(relx=0.95, rely=0.5, anchor='e')
        
    def create_control_terminal(self, parent):
        # Terminal header
        tk.Label(
            parent,
            text="┌─ CONTROL TERMINAL",
            font=('Consolas', 12, 'bold'),
            bg=self.colors['bg_panel'],
            fg=self.colors['accent_cyan'],
            anchor='w'
        ).pack(fill='x', padx=15, pady=(15, 5))
        
        # System Status
        status_frame = self.create_terminal_section(parent, "SYSTEM STATUS")
        
        self.model_status = tk.Label(
            status_frame,
            text="⚠ MODEL NOT LOADED",
            font=('Consolas', 10),
            bg=self.colors['bg_panel'],
            fg=self.colors['accent_orange']
        )
        self.model_status.pack(pady=5)
        
        self.create_terminal_button(
            status_frame,
            "► LOAD MODEL",
            self.load_model,
            self.colors['accent_cyan']
        ).pack(pady=5, fill='x')
        
        # Identity Input
        identity_frame = self.create_terminal_section(parent, "IDENTITY INPUT")
        
        tk.Label(
            identity_frame,
            text="Subject ID:",
            font=('Consolas', 9),
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary']
        ).pack(pady=(5, 0))
        
        input_container = tk.Frame(identity_frame, bg=self.colors['bg_dark'])
        input_container.pack(pady=5, padx=10, fill='x')
        
        self.subject_entry = tk.Entry(
            input_container,
            font=('Consolas', 16, 'bold'),
            bg='#1a2332',
            fg=self.colors['accent_cyan'],
            insertbackground=self.colors['accent_cyan'],
            relief='flat',
            justify='center',
            bd=2,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['accent_cyan']
        )
        self.subject_entry.pack(fill='x', ipady=8)
        self.subject_entry.insert(0, "90")
        
        # Action buttons
        btn_frame = tk.Frame(identity_frame, bg=self.colors['bg_panel'])
        btn_frame.pack(pady=10, fill='x', padx=10)
        
        self.create_terminal_button(
            btn_frame,
            "✓ ENROLL",
            self.enroll_user,
            self.colors['accent_green']
        ).pack(side='left', expand=True, fill='x', padx=(0, 5))
        
        self.create_terminal_button(
            btn_frame,
            "⚡ AUTHENTICATE",
            self.authenticate_user,
            self.colors['accent_orange']
        ).pack(side='right', expand=True, fill='x', padx=(5, 0))
        
        # Enrolled database
        db_frame = self.create_terminal_section(parent, "ENROLLED DATABASE")
        
        self.db_listbox = tk.Listbox(
            db_frame,
            font=('Consolas', 9),
            bg='#1a2332',
            fg=self.colors['accent_green'],
            selectbackground=self.colors['accent_cyan'],
            selectforeground='#000000',
            relief='flat',
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors['border'],
            height=8
        )
        self.db_listbox.pack(pady=5, padx=10, fill='both', expand=True)
        
    def create_main_display(self, parent):
        # Authentication display
        auth_display = tk.Frame(parent, bg=self.colors['bg_panel'], height=400)
        auth_display.pack(fill='x', pady=(0, 15))
        auth_display.pack_propagate(False)
        
        tk.Label(
            auth_display,
            text="┌─ AUTHENTICATION STATUS",
            font=('Consolas', 11, 'bold'),
            bg=self.colors['bg_panel'],
            fg=self.colors['accent_cyan'],
            anchor='w'
        ).pack(fill='x', padx=15, pady=(15, 10))
        
        # Large status display
        self.auth_status_label = tk.Label(
            auth_display,
            text="STANDBY",
            font=('Consolas', 48, 'bold'),
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary']
        )
        self.auth_status_label.pack(pady=40)
        
        # Similarity meter
        meter_frame = tk.Frame(auth_display, bg=self.colors['bg_panel'])
        meter_frame.pack(pady=10)
        
        tk.Label(
            meter_frame,
            text="SIMILARITY INDEX:",
            font=('Consolas', 10),
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary']
        ).pack(side='left', padx=(0, 10))
        
        self.similarity_value = tk.Label(
            meter_frame,
            text="---.----",
            font=('Consolas', 20, 'bold'),
            bg=self.colors['bg_panel'],
            fg=self.colors['accent_cyan']
        )
        self.similarity_value.pack(side='left')
        
        # Progress bar
        self.progress_canvas = tk.Canvas(
            auth_display,
            height=8,
            bg=self.colors['bg_dark'],
            highlightthickness=0
        )
        self.progress_canvas.pack(fill='x', padx=50, pady=10)
        
        # Neural signal visualization
        signal_frame = tk.Frame(parent, bg=self.colors['bg_panel'])
        signal_frame.pack(fill='both', expand=True)
        
        tk.Label(
            signal_frame,
            text="┌─ NEURAL SIGNATURE ANALYSIS",
            font=('Consolas', 11, 'bold'),
            bg=self.colors['bg_panel'],
            fg=self.colors['accent_cyan'],
            anchor='w'
        ).pack(fill='x', padx=15, pady=(15, 5))
        
        # Create matplotlib figure
        self.signal_fig = Figure(figsize=(10, 5), facecolor=self.colors['bg_panel'])
        self.signal_axes = []
        
        for i, channel in enumerate(self.config['TARGET_CHANNELS']):
            ax = self.signal_fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('#0d1117')
            ax.set_ylabel(channel, color=self.colors['accent_cyan'], fontweight='bold', fontsize=10)
            ax.tick_params(colors=self.colors['text_secondary'], labelsize=8)
            ax.grid(True, alpha=0.1, color=self.colors['accent_cyan'])
            
            for spine in ax.spines.values():
                spine.set_color(self.colors['border'])
                spine.set_linewidth(1)
                
            self.signal_axes.append(ax)
        
        self.signal_fig.tight_layout()
        
        self.signal_canvas = FigureCanvasTkAgg(self.signal_fig, signal_frame)
        self.signal_canvas.get_tk_widget().pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
    def create_system_monitor(self, parent):
        tk.Label(
            parent,
            text="┌─ SYSTEM MONITOR",
            font=('Consolas', 11, 'bold'),
            bg=self.colors['bg_panel'],
            fg=self.colors['accent_cyan'],
            anchor='w'
        ).pack(fill='x', padx=15, pady=(15, 10))
        
        # Metrics
        metrics_frame = tk.Frame(parent, bg=self.colors['bg_panel'])
        metrics_frame.pack(fill='x', padx=15, pady=5)
        
        self.create_metric_display(metrics_frame, "THRESHOLD", "0.8148")
        self.create_metric_display(metrics_frame, "ACCURACY", "96.0%")
        self.create_metric_display(metrics_frame, "EER", "4.26%")
        
        # System log
        tk.Label(
            parent,
            text="┌─ ACTIVITY LOG",
            font=('Consolas', 10, 'bold'),
            bg=self.colors['bg_panel'],
            fg=self.colors['accent_cyan'],
            anchor='w'
        ).pack(fill='x', padx=15, pady=(20, 5))
        
        log_frame = tk.Frame(parent, bg='#0d1117')
        log_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        self.log_text = tk.Text(
            log_frame,
            font=('Consolas', 8),
            bg='#0d1117',
            fg=self.colors['accent_green'],
            relief='flat',
            wrap='word',
            state='disabled',
            height=20
        )
        self.log_text.pack(fill='both', expand=True)
        
        self.log("[SYSTEM] Neural Biometric Security Terminal initialized")
        self.log("[SYSTEM] Awaiting model configuration...")
        
    def create_terminal_section(self, parent, title):
        tk.Label(
            parent,
            text=f"├─ {title}",
            font=('Consolas', 9, 'bold'),
            bg=self.colors['bg_panel'],
            fg=self.colors['text_secondary'],
            anchor='w'
        ).pack(fill='x', padx=15, pady=(15, 5))
        
        frame = tk.Frame(parent, bg=self.colors['bg_panel'])
        frame.pack(fill='both', padx=15, pady=5)
        return frame
        
    def create_terminal_button(self, parent, text, command, color):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Consolas', 10, 'bold'),
            bg=color,
            fg='#000000',
            activebackground=color,
            activeforeground='#000000',
            relief='flat',
            cursor='hand2',
            padx=15,
            pady=8
        )
        return btn
        
    def create_metric_display(self, parent, label, value):
        container = tk.Frame(parent, bg=self.colors['bg_dark'])
        container.pack(fill='x', pady=3)
        
        tk.Label(
            container,
            text=label,
            font=('Consolas', 8),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary']
        ).pack(side='left', padx=5)
        
        tk.Label(
            container,
            text=value,
            font=('Consolas', 10, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_cyan']
        ).pack(side='right', padx=5)
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')
        self.log_text.config(state='disabled')
        
    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Neural Model",
            filetypes=[("Keras Model", "*.keras"), ("H5 Model", "*.h5")]
        )
        
        if model_path:
            try:
                self.log("[MODEL] Loading neural network...")
                self.model = tf.keras.models.load_model(model_path)
                self.fingerprint_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer('fingerprint_layer').output
                )
                self.model_status.config(text="✓ MODEL ACTIVE", fg=self.colors['accent_green'])
                self.header_status.config(text="● SYSTEM ARMED", fg=self.colors['accent_green'])
                self.log("[MODEL] Neural network loaded successfully")
                self.log("[MODEL] Fingerprint extractor initialized")
                messagebox.showinfo("Success", "Neural model loaded and ready")
            except Exception as e:
                self.log(f"[ERROR] Model loading failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                
    def enroll_user(self):
        if self.fingerprint_model is None:
            messagebox.showerror("Error", "Load model first!")
            return
            
        try:
            subject_id = int(self.subject_entry.get())
            
            def enroll_thread():
                try:
                    self.log(f"[ENROLL] Initiating enrollment for Subject {subject_id:03d}")
                    segments, raw_data = self.load_subject_data(subject_id)
                    
                    self.log(f"[ENROLL] Extracting neural fingerprints...")
                    fingerprints = self.fingerprint_model.predict(segments, verbose=0)
                    
                    template = np.mean(fingerprints, axis=0)
                    self.enrolled_users[subject_id] = {
                        'template': template,
                        'fingerprints': fingerprints
                    }
                    
                    self.root.after(0, lambda: self.complete_enrollment(subject_id, raw_data))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.log(f"[ERROR] Enrollment failed: {str(e)}"))
                    self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                    
            threading.Thread(target=enroll_thread, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid subject ID!")
            
    def complete_enrollment(self, subject_id, raw_data):
        self.db_listbox.insert(tk.END, f"Subject {subject_id:03d} [ACTIVE]")
        self.plot_signal(raw_data)
        self.log(f"[ENROLL] Subject {subject_id:03d} enrolled successfully")
        self.log(f"[DATABASE] Total enrolled: {len(self.enrolled_users)}")
        messagebox.showinfo("Success", f"Subject {subject_id:03d} enrolled!")
        
    def authenticate_user(self):
        if self.fingerprint_model is None:
            messagebox.showerror("Error", "Load model first!")
            return
            
        if not self.enrolled_users:
            messagebox.showerror("Error", "No enrolled users!")
            return
            
        try:
            subject_id = int(self.subject_entry.get())
            
            if subject_id not in self.enrolled_users:
                messagebox.showerror("Error", f"Subject {subject_id:03d} not enrolled!")
                return
                
            def auth_thread():
                try:
                    self.root.after(0, lambda: self.set_auth_state("SCANNING"))
                    self.log(f"[AUTH] Initiating authentication for Subject {subject_id:03d}")
                    self.log(f"[AUTH] Capturing neural signature...")
                    
                    segments, raw_data = self.load_subject_data(subject_id)
                    
                    self.log(f"[AUTH] Processing biometric data...")
                    fingerprints = self.fingerprint_model.predict(segments, verbose=0)
                    
                    template = self.enrolled_users[subject_id]['template']
                    similarities = [1 - cosine(template, fp) for fp in fingerprints]
                    avg_similarity = np.mean(similarities)
                    
                    self.log(f"[AUTH] Similarity index: {avg_similarity:.4f}")
                    self.log(f"[AUTH] Threshold: {self.config['THRESHOLD']:.4f}")
                    
                    authenticated = avg_similarity >= self.config['THRESHOLD']
                    
                    self.root.after(0, lambda: self.complete_authentication(
                        authenticated, avg_similarity, raw_data
                    ))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.log(f"[ERROR] Authentication failed: {str(e)}"))
                    self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                    self.root.after(0, lambda: self.set_auth_state("STANDBY"))
                    
            threading.Thread(target=auth_thread, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid subject ID!")
            
    def complete_authentication(self, authenticated, similarity, raw_data):
        if authenticated:
            self.set_auth_state("AUTHENTICATED")
            self.log(f"[AUTH] ✓ ACCESS GRANTED - Similarity: {similarity:.4f}")
        else:
            self.set_auth_state("DENIED")
            self.log(f"[AUTH] ✗ ACCESS DENIED - Similarity: {similarity:.4f}")
            
        self.similarity_value.config(text=f"{similarity:.4f}")
        self.plot_signal(raw_data)
        self.animate_progress(similarity)
        
    def set_auth_state(self, state):
        self.auth_state = state
        
        if state == "STANDBY":
            self.auth_status_label.config(
                text="STANDBY",
                fg=self.colors['text_secondary']
            )
            self.header_status.config(text="● SYSTEM ARMED", fg=self.colors['accent_green'])
            
        elif state == "SCANNING":
            self.auth_status_label.config(
                text="SCANNING...",
                fg=self.colors['accent_cyan']
            )
            self.header_status.config(text="● PROCESSING", fg=self.colors['accent_cyan'])
            
        elif state == "AUTHENTICATED":
            self.auth_status_label.config(
                text="✓ ACCESS GRANTED",
                fg=self.colors['accent_green']
            )
            self.header_status.config(text="● AUTHENTICATED", fg=self.colors['accent_green'])
            
        elif state == "DENIED":
            self.auth_status_label.config(
                text="✗ ACCESS DENIED",
                fg=self.colors['accent_red']
            )
            self.header_status.config(text="● DENIED", fg=self.colors['accent_red'])
            
    def animate_progress(self, value):
        width = self.progress_canvas.winfo_width()
        if width <= 1:
            width = 500
            
        fill_width = int(width * value)
        
        color = self.colors['accent_green'] if value >= self.config['THRESHOLD'] else self.colors['accent_red']
        
        self.progress_canvas.delete('all')
        self.progress_canvas.create_rectangle(
            0, 0, fill_width, 8,
            fill=color,
            outline=''
        )
        
    def load_subject_data(self, subject_id):
        path = f"{self.dataset_path}/S{subject_id:03d}/S{subject_id:03d}R{self.config['RUN_ID']:02d}.edf"
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        if raw.info['sfreq'] != self.config['SAMPLE_RATE']:
            raw.resample(self.config['SAMPLE_RATE'], npad="auto", verbose=False)
        
        mne.rename_channels(raw.info, lambda x: x.strip('.'))
        raw.pick(self.config['TARGET_CHANNELS'])
        
        data = raw.get_data().T
        data = self.gram_schmidt(data)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        
        segments = self.create_segments(data, self.config['WINDOW_SIZE'], self.config['STRIDE'])
        
        return np.array(segments, dtype=np.float32), data
        
    def gram_schmidt(self, vectors):
        basis = np.zeros_like(vectors)
        for i in range(vectors.shape[1]):
            v = vectors[:, i]
            u = v.copy()
            for j in range(i):
                prev_u = basis[:, j]
                norm_prev = np.dot(prev_u, prev_u)
                if norm_prev > 1e-10:
                    projection = (np.dot(v, prev_u) / norm_prev) * prev_u
                    u -= projection
            basis[:, i] = u
        return basis
        
    def create_segments(self, data, window_size, stride):
        n_samples = data.shape[0]
        segments = []
        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size
            segment = data[start:end, :]
            if segment.shape == (window_size, data.shape[1]):
                segments.append(segment)
        return segments
        
    def plot_signal(self, data):
        for i, ax in enumerate(self.signal_axes):
            ax.clear()
            ax.plot(data[:500, i], color=self.colors['accent_cyan'], linewidth=1.5, alpha=0.8)
            ax.set_facecolor('#0d1117')
            ax.set_ylabel(
                self.config['TARGET_CHANNELS'][i],
                color=self.colors['accent_cyan'],
                fontweight='bold',
                fontsize=10
            )
            ax.tick_params(colors=self.colors['text_secondary'], labelsize=8)
            ax.grid(True, alpha=0.1, color=self.colors['accent_cyan'])
            
            for spine in ax.spines.values():
                spine.set_color(self.colors['border'])
                spine.set_linewidth(1)
                
        self.signal_canvas.draw()

def main():
    root = tk.Tk()
    app = SecuritySuiteGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
