import os
import cv2
from werkzeug.utils import secure_filename

def get_mime_type(file_path):
    """Mengembalikan MIME type berdasarkan ekstensi file"""
    if file_path.lower().endswith('.png'):
        return "image/png"
    elif file_path.lower().endswith(('.jpg', '.jpeg')):
        return "image/jpeg"
    elif file_path.lower().endswith('.bmp'):
        return "image/bmp"
    elif file_path.lower().endswith('.gif'):
        return "image/gif"
    elif file_path.lower().endswith('.mp4'):
        return "video/mp4"
    elif file_path.lower().endswith('.avi'):
        return "video/x-msvideo"
    elif file_path.lower().endswith('.mov'):
        return "video/quicktime"
    elif file_path.lower().endswith('.mkv'):
        return "video/x-matroska"
    elif file_path.lower().endswith('.flv'):
        return "video/x-flv"
    else:
        return "application/octet-stream"

def is_image(file_path):
    """Cek apakah file adalah gambar"""
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

def is_gif(file_path):
    """Cek apakah file adalah GIF"""
    return file_path.lower().endswith('.gif')

def is_video(file_path):
    """Cek apakah file adalah video"""
    return file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))

def save_metadata(output_folder, output_path, metadata):
    """Menyimpan metadata watermark ke file tersembunyi"""
    try:
        # Create a signature file for each watermark
        sig_path = os.path.join(output_folder, ".sig_" + os.path.basename(output_path))
        with open(sig_path, 'w') as f:
            f.write(str(metadata))
        
        # Also create a non-hidden version as backup
        backup_path = os.path.join(output_folder, "sig_" + os.path.basename(output_path))
        with open(backup_path, 'w') as f:
            f.write(str(metadata))
        
        return sig_path
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return None

def find_metadata(file_path):
    """Find metadata for a file"""
    try:
        base_name = os.path.basename(file_path)
        folder = os.path.dirname(file_path)
        
        # Try hidden version
        sig_path = os.path.join(folder, ".sig_" + base_name)
        if os.path.exists(sig_path):
            with open(sig_path, 'r') as f:
                return eval(f.read().strip())
        
        # Try non-hidden version
        sig_path = os.path.join(folder, "sig_" + base_name)
        if os.path.exists(sig_path):
            with open(sig_path, 'r') as f:
                return eval(f.read().strip())
    except Exception as e:
        print(f"Error finding metadata: {e}")
    
    return None

def clean_temp_files(directory, prefix):
    """Membersihkan file sementara dengan prefix tertentu"""
    for file in os.listdir(directory):
        if file.startswith(prefix):
            try:
                os.remove(os.path.join(directory, file))
            except:
                pass  # Abaikan error saat membersihkan
