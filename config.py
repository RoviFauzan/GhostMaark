import os

class Config:
    """Konfigurasi aplikasi Watermarker"""
    
    # Folder paths
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    OUTPUT_FOLDER = os.path.join(os.getcwd(), 'outputs')
    MODELS_FOLDER = os.path.join(os.getcwd(), 'models')
    
    # Max file size (50MB)
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    
    # Debug mode
    DEBUG = True
    
    # Port and host
    PORT = int(os.environ.get('PORT', 5000))
    HOST = '0.0.0.0'
    
    @staticmethod
    def create_folders():
        """Membuat folder jika belum ada"""
        for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER, Config.MODELS_FOLDER]:
            os.makedirs(folder, exist_ok=True)
