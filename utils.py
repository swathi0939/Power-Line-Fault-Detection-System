import os
import time
import logging
import cv2
import numpy as np
import torch
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLOv11 model once on import
MODEL_FILE = "best.pt"
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), MODEL_FILE)

logger.info(f"Loading YOLOv11 model from {model_path}...")
try:
    model = YOLO(model_path)
    logger.info("Model loaded successfully.")
    # Log model.names to verify dataset classes match expected classes
    if model:
        logger.info(f"Model original classes (model.names): {model.names}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Auto detect GPU or CPU
device = "0" if torch.cuda.is_available() else "cpu"
logger.info(f"YOLO will run on device: {device} (PyTorch CUDA available: {torch.cuda.is_available()})")

def cleanup_old_files(folder, max_age_seconds=600):
    """
    Deletes files in folder that are older than max_age_seconds (default 10 minutes).
    Helps prevent disk space exhaustion.
    """
    if not os.path.exists(folder):
        return

    now = time.time()
    deleted_count = 0
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        # Avoid deleting .gitkeep or other system files
        if filename.startswith('.'):
            continue
        try:
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    deleted_count += 1
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} files older than {max_age_seconds}s from {folder}")

def detect_faults(image_path, output_dir):
    """
    Runs YOLOv11 inference on the image at image_path,
    saves the prediction image with bounding boxes to output_dir,
    and returns inference metadata.
    """
    # 10. Add exception handling: Handle missing model
    if model is None:
        raise ValueError("YOLO model is not initialized.")

    try:
        start_time = time.time()

        # 10. Add exception handling: Verify image file existence
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Source image file not found: {image_path}")

        # Read the image size for metadata
        try:
            with Image.open(image_path) as img_pil:
                width, height = img_pil.size
        except Exception as e:
            logger.error(f"Failed to open image for reading dimensions: {e}")
            width, height = 0, 0

        # Run prediction
        # 11. Improve performance: Automatically use device configured on startup (CUDA if available, else CPU)
        try:
            results = model.predict(
                source=image_path,
                device=device,
                imgsz=640,
                conf=0.25,
                iou=0.45,
                agnostic_nms=True,
                max_det=20,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {str(e)}")

        detection_time = time.time() - start_time

        # Process detections
        predictions = []
        summary = {
            "Total Faults": 0,
            "Broken Insulator": 0,
            "Pollution Flashover": 0,
            "Normal Insulator": 0
        }

        # 10. Add exception handling: Load original image using OpenCV for custom rendering
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image via OpenCV from: {image_path}")

        # Result object details
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            logger.info("No detections found by YOLO.")
        else:
            logger.info(f"YOLO found {len(boxes)} detections.")

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                confidence = float(box.conf[0])
                class_idx = int(box.cls[0])

                # 2. Fix class mapping: Read classes directly from model.names and format
                try:
                    raw_class = model.names[class_idx]
                except KeyError:
                    raw_class = f"class_{class_idx}"

                mapped_class = raw_class.replace("_", " ").replace("-", " ").title()

                # Align class names with expected frontend categories
                if mapped_class == "Broken":
                    mapped_class = "Broken Insulator"
                elif mapped_class == "Insulator":
                    mapped_class = "Normal Insulator"

                # Box coords
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 4. Add detailed logging: For every detection print/log Class, Confidence, Box
                logger.info(f"{mapped_class}\nConfidence: {confidence:.2f}\nBox: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")

                # 1 & 5. Only discard detections if confidence is below 0.25
                if confidence < 0.25:
                    continue

                # 6. Fix prediction list generation: Every accepted detection is appended to predictions
                predictions.append({
                    "class_name": mapped_class,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

                # 7. Ensure counters update correctly: increment summary counts for accepted detections
                if mapped_class == "Broken Insulator":
                    summary["Broken Insulator"] += 1
                    summary["Total Faults"] += 1
                elif mapped_class == "Pollution Flashover":
                    summary["Pollution Flashover"] += 1
                    summary["Total Faults"] += 1
                elif mapped_class == "Normal Insulator":
                    summary["Normal Insulator"] += 1
                else:
                    summary[mapped_class] = summary.get(mapped_class, 0) + 1

                # 8. Draw bounding boxes only for accepted detections: never draw boxes that are rejected
                # Colors: RGB to BGR
                if mapped_class == "Broken Insulator":
                    color = (68, 68, 239)        # Red BGR
                elif mapped_class == "Pollution Flashover":
                    color = (11, 158, 245)       # Orange BGR
                elif mapped_class == "Normal Insulator":
                    color = (129, 185, 16)       # Green BGR
                else:
                    color = (241, 102, 99)       # Indigo BGR

                # Draw premium bounding box (thickness 3)
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), color, 3)

                # Draw custom corner accents (visual elegance)
                length = min(30, int((x2_i - x1_i) * 0.2), int((y2_i - y1_i) * 0.2))
                if length > 2:
                    # Top-Left
                    cv2.line(img, (x1_i, y1_i), (x1_i + length, y1_i), color, 5)
                    cv2.line(img, (x1_i, y1_i), (x1_i, y1_i + length), color, 5)
                    # Top-Right
                    cv2.line(img, (x2_i, y1_i), (x2_i - length, y1_i), color, 5)
                    cv2.line(img, (x2_i, y1_i), (x2_i, y1_i + length), color, 5)
                    # Bottom-Left
                    cv2.line(img, (x1_i, y2_i), (x1_i + length, y2_i), color, 5)
                    cv2.line(img, (x1_i, y2_i), (x1_i, y2_i - length), color, 5)
                    # Bottom-Right
                    cv2.line(img, (x2_i, y2_i), (x2_i - length, y2_i), color, 5)
                    cv2.line(img, (x2_i, y2_i), (x2_i, y2_i - length), color, 5)

                # Text label details
                label = f"{mapped_class} {confidence:.1%}"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = max(0.5, min(0.8, (x2_i - x1_i) / 300))
                thickness = 1

                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Badge coordinates (above the box, check bounds)
                badge_y1 = max(0, y1_i - text_h - 10)
                badge_y2 = y1_i if y1_i - text_h - 10 >= 0 else y1_i + text_h + 10

                cv2.rectangle(img, (x1_i, badge_y1), (x1_i + text_w + 10, badge_y2), color, -1)
                text_y = badge_y2 - 5 if y1_i - text_h - 10 >= 0 else badge_y2 - 5

                # Write white text on badge
                cv2.putText(img, label, (x1_i + 5, text_y + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 9. Verify image saving: Ensure prediction image is saved inside static/predictions is exactly the processed image
        filename = os.path.basename(image_path)
        predicted_path = os.path.join(output_dir, filename)
        
        # 10. Add exception handling: Wrap OpenCV write
        try:
            success = cv2.imwrite(predicted_path, img)
            if not success:
                raise ValueError("cv2.imwrite failed to save the image.")
        except Exception as e:
            raise RuntimeError(f"Failed to save predicted image via OpenCV: {str(e)}")

        # 12. Preserve existing API responses
        metadata = {
            "filename": filename,
            "detection_time_ms": int(detection_time * 1000),
            "device": device,
            "resolution": f"{width}x{height}",
            "predictions": predictions,
            "summary": summary
        }

        return predicted_path, metadata

    except Exception as e:
        logger.exception("Error in detect_faults backend logic.")
        # Re-raise the exception to be handled by Flask request handler
        raise e
