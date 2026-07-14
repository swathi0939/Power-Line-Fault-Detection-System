import os
import time
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename
from config import Config
from utils import detect_faults, cleanup_old_files

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app_server.log")
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Run config directory initialization
Config.init_app()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    """
    Renders the beautiful landing dashboard.
    """
    logger.info("Serving home page.")
    return render_template('index.html')

@app.route('/about')
def about():
    """
    Renders the informative details page.
    """
    logger.info("Serving about page.")
    return render_template('about.html')

@app.route('/detect', methods=['POST'])
def detect():
    """
    API endpoint to upload an image and run YOLO11 detection.
    """
    logger.info("Detect endpoint called.")
    
    # Run cleanup of files older than 10 minutes (600s) on each prediction request
    try:
        cleanup_old_files(app.config['UPLOAD_FOLDER'], 600)
        cleanup_old_files(app.config['PREDICTION_FOLDER'], 600)
    except Exception as e:
        logger.error(f"Error during scheduled file cleanup: {e}")

    # Check if files is in the request
    if 'file' not in request.files:
        logger.warning("No file part in the request.")
        return jsonify({"success": False, "error": "No file part in the request."}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("No file selected.")
        return jsonify({"success": False, "error": "No file selected."}), 400
        
    if not allowed_file(file.filename):
        logger.warning(f"File type not allowed: {file.filename}")
        allowed_str = ", ".join(app.config['ALLOWED_EXTENSIONS']).upper()
        return jsonify({"success": False, "error": f"Invalid file format. Allowed types: {allowed_str}."}), 400

    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Prevent duplicate filenames by appending timestamp
        base, ext = os.path.splitext(filename)
        unique_filename = f"{base}_{int(time.time() * 1000)}{ext}"
        
        # Save upload file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        logger.info(f"File saved successfully to {upload_path}")
        
        # Run detection
        predicted_path, metadata = detect_faults(upload_path, app.config['PREDICTION_FOLDER'])
        logger.info(f"Fault detection successful. Output saved to {predicted_path}")
        
        # Build relative URLs for frontend display
        metadata["original_url"] = f"/static/uploads/{unique_filename}"
        metadata["predicted_url"] = f"/static/predictions/{unique_filename}"
        metadata["success"] = True
        
        # Check if the request is AJAX or expects JSON
        accept_header = request.headers.get('Accept', '')
        is_ajax = (
            'application/json' in accept_header or 
            request.args.get('ajax') == 'true' or 
            request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        )
        
        if is_ajax:
            return jsonify(metadata)
            
        # If traditional form POST, render the result template
        return render_template(
            'result.html',
            original_url=metadata["original_url"],
            predicted_url=metadata["predicted_url"],
            summary=metadata["summary"],
            predictions=metadata["predictions"],
            detection_time_ms=metadata["detection_time_ms"],
            device=metadata["device"],
            resolution=metadata["resolution"],
            filename=metadata["filename"]
        )
        
    except Exception as e:
        logger.exception("An error occurred during image processing or detection.")
        return jsonify({"success": False, "error": f"Inference failed: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """
    Downloads the predicted result file.
    """
    filename = secure_filename(filename)
    logger.info(f"Downloading predicted file: {filename}")
    
    # Check traversal safety by comparing directory path
    prediction_dir = os.path.abspath(app.config['PREDICTION_FOLDER'])
    target_path = os.path.abspath(os.path.join(prediction_dir, filename))
    
    if not target_path.startswith(prediction_dir):
        logger.warning("Directory traversal attempt blocked.")
        return abort(403)
        
    if not os.path.exists(target_path):
        logger.warning(f"Requested download file not found: {filename}")
        return abort(404)
        
    return send_from_directory(prediction_dir, filename, as_attachment=True)

@app.errorhandler(413)
def request_entity_too_large(error):
    logger.warning("Upload exceeds maximum allowed size.")
    return jsonify({
        "success": False, 
        "error": f"File is too large. Maximum size allowed is {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)} MB."
    }), 413

if __name__ == '__main__':
    # Run development server
    app.run(host='0.0.0.0', port=5000, debug=True)
