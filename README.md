# Power Line Fault Detection System

A commercial-grade, production-ready AI web application designed to automatically detect and classify defects on power line transmission insulators. The application utilizes a customized **Ultralytics YOLO11** model for real-time computer vision inference and presents findings in a gorgeous, fully responsive **Glassmorphism** dashboard.

## 🚀 Key Features
- **Accurate Fault Classification**: Automatically detects and classifies insulators into three states:
  1. **Broken Insulator** (Structural shell damages/fractures)
  2. **Pollution Flashover** (Surface salt/dust deposits)
  3. **Normal Insulator** (Healthy, clean string)
- **High-Performance Backend**: Built with Flask, running asynchronous prediction pipelines.
- **Hardware Acceleration**: Automatic CPU/GPU fallback depending on CUDA availability.
- **Glassmorphism Interface**: Premium design incorporating neon borders, micro-animations, loading scanners, and dual light/dark themes.
- **Visual Analytics**: Interactive side-by-side view (Original vs. Detection Overlay), dynamic counts animation, and bounding box tables.
- **Security-First Design**: Secure file validation (max 20MB, allowed image types only), path traversal prevention, and local database file-rotation (cleans temporary folders automatically).

---

## 📂 Project Directory Structure

```
Power-Line-Fault-Detection-System/
├── app.py                 # Flask server & routes definitions
├── config.py              # Application settings (directory paths, upload sizes)
├── utils.py               # Inference helper, OpenCV canvas drawer & file cleanup
├── verify_system.py       # Dependency check & mock image prediction tester
├── best.pt                # Trained YOLO11 model weights file
├── requirements.txt       # Python environment package dependencies
├── README.md              # Technical system documentation
├── app_server.log         # Server logging outputs (generated after start)
├── templates/             # HTML Templates
│   ├── index.html         # Main analysis dashboard template
│   └── about.html         # Problem description & model metrics template
└── static/                # Web assets
    ├── css/
    │   └── style.css      # Custom styling, dark/light themes, & keyframe scanners
    └── js/
        └── main.js        # Drag-and-drop handles, async REST calls, & DOM updates
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8 to 3.12 (Python 3.12 recommended)
- Git (optional)
- NVIDIA CUDA Toolkit & cuDNN (optional, for GPU acceleration)

### Step 1: Clone or Navigate to Directory
Ensure you are inside the project folder:
```bash
cd "Power Line Fault Detection System"
```

### Step 2: Create a Virtual Environment (Recommended)
Isolate the project packages:
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
Install all package requirements via pip:
```bash
pip install -r requirements.txt
```

---

## 🧪 System Verification

To ensure your Python environment is configured correctly, run the custom verification script:
```bash
python verify_system.py
```
This script will test:
1. Necessary library imports.
2. CUDA GPU availability.
3. Model loading of `best.pt`.
4. End-to-end mock prediction on a synthetic noise matrix.

---

## ⚡ Running the Application

Start the Flask server locally:
```bash
python app.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Usage Steps:
1. **Toggle Theme**: Use the circular button in the navbar to switch between Light and Dark mode.
2. **Select Image**: Drag & drop any `.jpg`, `.jpeg`, `.png`, or `.bmp` survey photograph into the dashboard dropzone, or click inside to browse local files.
3. **Inference**: Click **Detect Faults** to run the YOLO11 model. A scanning overlay will animate while processing.
4. **Inspect Results**: View the detection coordinates table, summary metrics cards, and download the annotated file.

---

## 🤖 YOLO11 Model & Performance
- **Input Dimensions**: 640x640 pixels (resized dynamically).
- **Latency**: ~15ms on NVIDIA RTX GPUs / ~120ms on modern multi-core CPUs.
- **Inference Mode**: Dynamically selects CUDA if a compatible card is detected, otherwise falling back to CPU.
- **Overlay Rendering**: Customized OpenCV canvas draw routine highlighting faults with distinct colored border tags (Red = Broken, Orange = Pollution, Green = Normal).
