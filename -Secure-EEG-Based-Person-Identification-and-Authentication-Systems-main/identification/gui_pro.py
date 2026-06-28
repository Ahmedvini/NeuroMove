import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import pyedflib
import os
import threading
import time
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

# ============== CONFIGURATION ==============
CHECKPOINT_PATH = os.environ.get("EEG_CHECKPOINT_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "checkpoints", "cp-0050.weights.h5"))
DATASET_PATH = os.environ.get("EEG_DATASET_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "files"))
THRESHOLD = 0.85

class ProfessionalAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroID: Biometric Analytics Dashboard")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f5f7fa") # Light professional gray
        
        # Data Buffers for Graph
        self.history_len = 50
        self.time_history = deque(np.linspace(0, self.history_len, self.history_len), maxlen=self.history_len)
        self.conf_history = deque([0]*self.history_len, maxlen=self.history_len)
        
        # State
        self.model = None
        self.is_monitoring = False
        self.current_file_data = None
        
        self.setup_styles()
        self.setup_ui()
        
        # Initial Plot update
        self.update_plot()
        
        # Background Load
        self.log_event("System", "Initializing Neural Network Engine...")
        threading.Thread(target=self.load_model, daemon=True).start()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Card style
        style.configure("Card.TFrame", background="white", relief="ridge", borderwidth=1)
        style.configure("Header.TLabel", background="#f5f7fa", foreground="#2c3e50", font=('Segoe UI', 24, 'bold'))
        style.configure("Sub.TLabel", background="white", foreground="#7f8c8d", font=('Segoe UI', 10))
        
        # Button styles
        style.configure("Primary.TButton", font=('Segoe UI', 11, 'bold'), background="#3498db", foreground="white")
        style.map("Primary.TButton", background=[('active', '#2980b9')])
        
        style.configure("Action.TButton", font=('Segoe UI', 11, 'bold'), background="#27ae60", foreground="white")
        style.map("Action.TButton", background=[('active', '#2ecc71')])
        
        style.configure("Stop.TButton", font=('Segoe UI', 11, 'bold'), background="#e74c3c", foreground="white")
        style.map("Stop.TButton", background=[('active', '#c0392b')])

    def setup_ui(self):
        # 1. Header
        header_frame = tk.Frame(self.root, bg="#f5f7fa", height=80)
        header_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(header_frame, text="NeuroID Analytics Dashboard", style="Header.TLabel").pack(side="left")
        self.status_pill = tk.Label(header_frame, text="SYSTEM OFFLINE", bg="#95a5a6", fg="white", 
                                  font=('Segoe UI', 10, 'bold'), padx=15, pady=5, relief="flat")
        self.status_pill.pack(side="right", pady=10)

        # 2. Main Content Grid
        content_frame = tk.Frame(self.root, bg="#f5f7fa")
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left Column: Controls & Stats
        left_col = ttk.Frame(content_frame)
        left_col.pack(side="left", fill="y", padx=(0, 20))
        
        # Control Card
        control_card = ttk.Frame(left_col, style="Card.TFrame", padding=20)
        control_card.pack(fill="x", pady=(0, 20))
        
        ttk.Label(control_card, text="SESSION CONTROLS", font=('Segoe UI', 10, 'bold'), background="white").pack(anchor="w")
        ttk.Separator(control_card).pack(fill="x", pady=10)
        
        self.btn_load = ttk.Button(control_card, text="📂 Load Dataset", style="Primary.TButton", command=self.load_file)
        self.btn_load.pack(fill="x", pady=5)
        
        self.btn_start = ttk.Button(control_card, text="▶ Start Session", style="Action.TButton", 
                                  command=self.toggle_monitoring, state="disabled")
        self.btn_start.pack(fill="x", pady=5)

        # Identity Card (Dynamic Result)
        id_card = ttk.Frame(left_col, style="Card.TFrame", padding=20)
        id_card.pack(fill="x", pady=(0, 20))
        
        ttk.Label(id_card, text="CURRENT IDENTITY", font=('Segoe UI', 10, 'bold'), background="white").pack(anchor="w")
        self.lbl_id_big = tk.Label(id_card, text="---", font=('Segoe UI', 36, 'bold'), bg="white", fg="#bdc3c7")
        self.lbl_id_big.pack(pady=10)
        
        self.lbl_conf_text = tk.Label(id_card, text="Confidence: 0.0%", font=('Segoe UI', 12), bg="white", fg="#7f8c8d")
        self.lbl_conf_text.pack()

        # Right Column: Graph & Logs
        right_col = ttk.Frame(content_frame)
        right_col.pack(side="right", fill="both", expand=True)
        
        # Graph Card
        graph_card = ttk.Frame(right_col, style="Card.TFrame", padding=10)
        graph_card.pack(fill="both", expand=True, pady=(0, 20))
        
        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor('white')
        
        self.line, = self.ax.plot([], [], lw=2, color="#3498db")
        self.ax.axhline(y=THRESHOLD*100, color='#e74c3c', linestyle='--', alpha=0.5, label='Security Threshold')
        self.ax.set_ylim(0, 105)
        self.ax.set_xlim(0, self.history_len)
        self.ax.set_title("Real-Time Confidence Metrics", fontsize=10)
        self.ax.set_ylabel("Confidence (%)")
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.legend(loc="lower right", fontsize=8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Log Card
        log_card = ttk.Frame(right_col, style="Card.TFrame", padding=10)
        log_card.pack(fill="x", pady=10)
        
        ttk.Label(log_card, text="SECURITY AUDIT LOG", font=('Segoe UI', 10, 'bold'), background="white").pack(anchor="w")
        
        self.log_tree = ttk.Treeview(log_card, columns=("Time", "Event", "Status"), show="headings", height=6)
        self.log_tree.heading("Time", text="Timestamp")
        self.log_tree.heading("Event", text="Event Type")
        self.log_tree.heading("Status", text="Status")
        self.log_tree.column("Time", width=100)
        self.log_tree.column("Event", width=300)
        self.log_tree.column("Status", width=100)
        self.log_tree.pack(fill="x", pady=5)

    def log_event(self, event_type, message, status="INFO"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_tree.insert("", 0, values=(ts, message, status))
        
        # Color coding
        tag = "norm"
        if status == "GRANTED": tag = "good"
        elif status == "DENIED": tag = "bad"
        elif status == "INFO": tag = "info"
        
        self.log_tree.tag_configure("good", foreground="#27ae60")
        self.log_tree.tag_configure("bad", foreground="#c0392b")
        self.log_tree.item(self.log_tree.get_children()[0], tags=(tag,))

    def load_model(self):
        try:
            from tensorflow.keras import models, layers
            # Silent import
            input_shape = (160, 64)
            model = models.Sequential([
                layers.BatchNormalization(input_shape=input_shape, epsilon=.0001),
                layers.Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(filters=512, kernel_size=2, activation='relu', padding='same'),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(filters=1024, kernel_size=2, activation='relu', padding='same'),
                layers.MaxPooling1D(pool_size=2),
                layers.Reshape((-1, 64*160)),
                layers.Dropout(0.5),
                layers.Dense(109, activation='softmax')
            ])
            
            if os.path.exists(CHECKPOINT_PATH):
                model.load_weights(CHECKPOINT_PATH)
                self.model = model
                self.root.after(0, lambda: self.set_system_status("STANDBY", "#f39c12"))
                self.log_event("System", "Model weights loaded successfully.", "INFO")
            else:
                self.log_event("Error", "Checkpoint file not found.", "CRITICAL")
        except Exception as e:
            self.log_event("Error", str(e), "CRITICAL")

    def set_system_status(self, text, color):
        self.status_pill.config(text=text, bg=color)

    def load_file(self):
        path = filedialog.askopenfilename(initialdir=DATASET_PATH, title="Select EEG Recording", filetypes=[("EDF", "*.edf")])
        if path:
            self.current_filename = os.path.basename(path)
            self.log_event("IO", f"Loaded file: {self.current_filename}", "INFO")
            
            # Read
            try:
                f = pyedflib.EdfReader(path)
                n = f.getNSamples()[0]
                data = np.zeros((f.signals_in_file, n), dtype=np.float32)
                for i in range(f.signals_in_file): data[i, :] = f.readSignal(i)
                f.close()
                self.current_file_data = data.transpose()
                self.btn_start.config(state="normal")
            except:
                self.log_event("IO", "Failed to read file", "CRITICAL")

    def toggle_monitoring(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.btn_start.config(text="⏹ Stop Session", style="Stop.TButton")
            self.btn_load.config(state="disabled")
            self.set_system_status("MONITORING ACTIVE", "#27ae60")
            threading.Thread(target=self.monitor_loop, daemon=True).start()
        else:
            self.is_monitoring = False
            self.btn_start.config(text="▶ Start Session", style="Action.TButton")
            self.btn_load.config(state="normal")
            self.set_system_status("STANDBY", "#f39c12")
            self.lbl_id_big.config(text="---", fg="#bdc3c7")
            self.lbl_conf_text.config(text="Confidence: 0.0%", fg="#7f8c8d")

    def monitor_loop(self):
        idx = 0
        while self.is_monitoring and idx + 160 < len(self.current_file_data):
            # Process
            chunk = self.current_file_data[idx:idx+160]
            input_tensor = np.expand_dims(chunk, axis=0)
            
            preds = self.model.predict(input_tensor, verbose=0)
            conf = np.max(preds)
            uid = np.argmax(preds) + 1
            
            # Update Data
            self.conf_history.append(conf * 100)
            
            # UI Update
            self.root.after(0, lambda u=uid, c=conf: self.update_dashboard(u, c))
            
            idx += 160
            time.sleep(1.0)
        
        self.is_monitoring = False
        self.root.after(0, lambda: self.btn_start.config(text="▶ Restart", style="Action.TButton"))

    def update_dashboard(self, uid, conf):
        # 1. Update Graph
        self.line.set_data(range(len(self.conf_history)), self.conf_history)
        self.canvas.draw()
        
        # 2. Update Identity Card
        self.lbl_conf_text.config(text=f"Confidence: {conf*100:.1f}%")
        
        if conf >= THRESHOLD:
            self.lbl_id_big.config(text=f"S{uid:03d}", fg="#2c3e50")
            self.lbl_conf_text.config(fg="#27ae60")
            self.log_event("Access Control", f"User S{uid:03d} verified", "GRANTED")
        else:
            self.lbl_id_big.config(text="UNKNOWN", fg="#c0392b")
            self.lbl_conf_text.config(fg="#c0392b")
            self.log_event("Access Control", f"Authentication failed ({conf*100:.1f}%)", "DENIED")

    def update_plot(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalAuthApp(root)
    root.mainloop()
