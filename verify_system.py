import os
import sys
import numpy as np
import cv2

print("=== Power Line Fault Detection System Verification ===")

# 1. Test Imports
print("\n[1/4] Verifying library imports...")
try:
    import flask
    import ultralytics
    import torch
    import PIL
    print("  - Flask: OK")
    print("  - Ultralytics (YOLO): OK")
    print("  - PyTorch: OK")
    print("  - Pillow: OK")
    print("  - OpenCV & NumPy: OK")
except ImportError as e:
    print(f"  - ERROR: Missing dependency: {e}")
    print("  Please run: pip install -r requirements.txt")
    sys.exit(1)

# 2. Test CUDA
print("\n[2/4] Verifying hardware acceleration...")
cuda_avail = torch.cuda.is_available()
print(f"  - CUDA GPU available: {cuda_avail}")
if cuda_avail:
    print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("  - Falling back to CPU.")

# 3. Test Model Load
print("\n[3/4] Testing YOLOv11 model loading...")
model_path = "best.pt"
if not os.path.exists(model_path):
    print(f"  - ERROR: Model file '{model_path}' not found in current directory.")
    sys.exit(1)

try:
    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"  - Model loaded successfully from {model_path}.")
    print(f"  - Detected Classes: {model.names}")
except Exception as e:
    print(f"  - ERROR: Failed to load model: {e}")
    sys.exit(1)

# 4. Test Mock Inference
print("\n[4/4] Running end-to-end inference test on dummy image...")
try:
    # Create a dummy image (RGB noise, size 640x640)
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    dummy_path = "test_dummy.jpg"
    cv2.imwrite(dummy_path, dummy_img)
    print(f"  - Mock image created at {dummy_path}.")
    
    # Import detect_faults from utils
    from utils import detect_faults
    
    # Ensure static dirs exist
    from config import Config
    Config.init_app()
    
    # Run test
    output_path, metadata = detect_faults(dummy_path, Config.PREDICTION_FOLDER)
    print(f"  - Detection completed in {metadata['detection_time_ms']} ms.")
    print(f"  - Inference Device: {metadata['device']}")
    print(f"  - Resolution: {metadata['resolution']}")
    print(f"  - Total objects detected: {metadata['summary']['Total Faults'] + metadata['summary']['Normal Insulator']}")
    print("  - Result image saved successfully.")
    
    # Clean up dummy test file
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    print("  - Cleaned up mock test files.")
    
    print("\n=== SYSTEM VERIFICATION SUCCESSFUL ===")
    
except Exception as e:
    print(f"  - ERROR: End-to-end test failed: {e}")
    sys.exit(1)
