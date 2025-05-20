from flask import Flask
import os

from config import Config
from routes.main_routes import setup_main_routes
from routes.watermark_routes import setup_watermark_routes

def create_app():
    """Buat dan konfigurasi aplikasi Flask"""
    app = Flask(__name__)
    
    # Konfigurasi aplikasi dari objek Config
    app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    app.config['OUTPUT_FOLDER'] = Config.OUTPUT_FOLDER
    app.config['MODELS_FOLDER'] = Config.MODELS_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    
    # Buat folder jika belum ada
    Config.create_folders()
    
    # Dictionary untuk menyimpan progres task
    tasks = {}
    
    # Setup rute-rute aplikasi
    setup_main_routes(app, Config, tasks)
    setup_watermark_routes(app, Config, tasks)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
