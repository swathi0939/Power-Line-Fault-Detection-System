import os

class Config:
    # Secret key for Flask session/security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'powerline_fault_detection_secret_key_2026')
    
    # Workspace base directory
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Upload and prediction folders
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    PREDICTION_FOLDER = os.path.join(BASE_DIR, 'static', 'predictions')
    
    # Allowed file formats
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
    
    # Max file size: 20 MB
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024
    
    # Ensure directories exist
    @classmethod
    def init_app(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.PREDICTION_FOLDER, exist_ok=True)
