# EEG Authentication System - GUI Application

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib mne scikit-learn tensorflow scipy
```

### Running the Application
```bash
python eeg_auth_gui.py
```

## 📋 Features

### 1. **Professional Dashboard**
- Modern dark theme interface
- Real-time EEG signal visualization
- Authentication status display
- Metrics dashboard

### 2. **User Management**
- **Enroll Users**: Create biometric templates from EEG data
- **Verify Users**: Authenticate against enrolled templates
- **User List**: View all enrolled subjects

### 3. **Visualizations**
- **3-Channel EEG Display**: Real-time plotting of Oz, T7, Cz channels
- **ROC Curve**: System performance visualization
- **Similarity Scores**: Live authentication metrics

## 🎯 How to Use

### Step 1: Load the Model
1. Click **"📁 Load Trained Model"**
2. Select your trained `.keras` or `.h5` model file
3. Wait for confirmation

### Step 2: Enroll a User
1. Enter a **Subject ID** (e.g., 1, 2, 3...)
2. Click **"✅ Enroll User"**
3. The system will:
   - Load the subject's EEG data
   - Process and create a biometric template
   - Add to the enrolled users list

### Step 3: Verify a User
1. Enter the **Subject ID** of an enrolled user
2. Click **"🔐 Verify User"**
3. View the result:
   - ✅ **AUTHENTICATED** (Green) - User verified
   - ❌ **REJECTED** (Red) - Authentication failed
   - **Similarity Score** - Confidence metric

## 📊 Understanding the Results

### Similarity Score
- **Range**: 0.0 to 1.0
- **Threshold**: 0.8148 (from training)
- **Above threshold** = Authenticated
- **Below threshold** = Rejected

### Visual Indicators
- **Green** = Success
- **Red** = Failure
- **Blue** = System information

## 🔧 Configuration

The system uses these default parameters:
- **Channels**: Oz, T7, Cz
- **Window Size**: 160 samples (1 second @ 160 Hz)
- **Stride**: 4 samples
- **Sample Rate**: 160 Hz

## 📁 Dataset Structure

Your dataset should be organized as:
```
files/
├── S001/
│   └── S001R01.edf
├── S002/
│   └── S002R01.edf
└── ...
```

## 🎨 GUI Components

### Left Panel (Controls)
- System status indicator
- Model loading
- Dataset path configuration
- User enrollment/verification
- Enrolled users list
- Performance metrics

### Right Panel (Visualizations)
- Real-time EEG signal plots (3 channels)
- Authentication result display
- Similarity score
- ROC curve (when available)

## 🔐 Security Notes

- The system uses **cosine similarity** for matching
- Templates are stored in memory (not persistent)
- Threshold of **0.8148** provides ~96% accuracy
- Best used with fresh EEG recordings

## 🐛 Troubleshooting

### "Model Not Loaded" Error
- Ensure you've loaded a valid trained model
- Check that the model has a 'fingerprint_layer'

### "File Not Found" Error
- Verify the dataset path is correct
- Check that subject folders exist (S001, S002, etc.)
- Ensure .edf files are present

### "Subject Not Enrolled" Error
- Enroll the user first before verification
- Check the subject ID is correct

## 💡 Tips

1. **First Time Use**: Load the model before any operations
2. **Testing**: Enroll subjects 1-89, verify with 90-109
3. **Performance**: Processing may take a few seconds per subject
4. **Visualization**: Signal plots update after each operation

## 📝 Technical Details

- **Framework**: Tkinter (Python GUI)
- **Plotting**: Matplotlib
- **EEG Processing**: MNE-Python
- **Deep Learning**: TensorFlow/Keras
- **Threading**: Background processing for smooth UI

---

**Developed for EEG Biometric Authentication Research**
