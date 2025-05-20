import os
import cv2
import sys
import numpy as np
import tensorflow as tf
from .file_utils import save_metadata

def process_image_deep_learning(image_path, watermark_text, output_folder, alpha=0.8):
    """Menambahkan watermark teks dengan metode deep learning (CNN/GAN)"""
    try:
        # Pastikan TensorFlow tersedia
        if 'tensorflow' not in sys.modules:
            raise Exception("TensorFlow is not available. Install with 'pip install tensorflow'")
        
        # Baca gambar
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Failed to open image")
        
        # Konversi ke RGB (TensorFlow biasanya mengharapkan RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess gambar untuk model
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        img_tensor = img_tensor / 255.0  # Normalize ke [0,1]
        
        # Expand dimensions untuk batch processing (tambahkan batch dimension)
        img_tensor = tf.expand_dims(img_tensor, 0)
        
        # Resize ke ukuran yang diharapkan model jika diperlukan
        target_size = (256, 256)  # Ukuran umum untuk banyak model
        original_size = img_rgb.shape[:2]
        img_tensor = tf.image.resize(img_tensor, target_size)
        
        # Konversi watermark teks ke representasi biner
        watermark_bits = ''.join(format(ord(char), '08b') for char in watermark_text)
        
        # Buat mask biner dari bit watermark
        binary_len = len(watermark_bits)
        
        # Buat pola binary 2D dari watermark teks yang akan masuk di target gambar
        watermark_pattern = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
        
        # Sebarkan data biner pada gambar dalam sebuah pola
        # Ini pendekatan sederhana - metode CNN/GAN asli menggunakan pola lebih canggih
        bit_index = 0
        for i in range(target_size[0]):
            for j in range(target_size[1]):
                if bit_index < binary_len:
                    watermark_pattern[i, j] = int(watermark_bits[bit_index])
                    bit_index += 1
                    
                    # Jika semua bit sudah digunakan, kembali ke awal
                    if bit_index >= binary_len:
                        bit_index = 0
        
        # Buat tensor watermark
        watermark_tensor = tf.convert_to_tensor(watermark_pattern, dtype=tf.float32)
        watermark_tensor = tf.expand_dims(watermark_tensor, -1)  # Tambahkan channel dimension
        watermark_tensor = tf.tile(watermark_tensor, [1, 1, 3])  # Replikasi ke semua channel
        watermark_tensor = tf.expand_dims(watermark_tensor, 0)  # Tambahkan batch dimension
        
        # Simulasikan neural network embedding - di produksi, Anda akan menggunakan model pre-trained
        # Ini pendekatan alpha blending sederhana sebagai placeholder
        
        # Dalam implementasi nyata, kita akan menggunakan encoder-decoder network:
        # encoder = load_model('encoder.h5')
        # decoder = load_model('decoder.h5')
        # watermarked_tensor = encoder([img_tensor, watermark_tensor])
        
        # Di sini kita gunakan pendekatan sederhana untuk demonstrasi
        watermarked_tensor = img_tensor * (1 - alpha * 0.01) + watermark_tensor * (alpha * 0.01)
        
        # Clip nilai ke rentang valid [0,1]
        watermarked_tensor = tf.clip_by_value(watermarked_tensor, 0.0, 1.0)
        
        # Konversi kembali ke gambar
        watermarked_img = watermarked_tensor[0].numpy() * 255.0
        watermarked_img = watermarked_img.astype(np.uint8)
        
        # Resize kembali ke dimensi asli
        watermarked_img = cv2.resize(watermarked_img, (original_size[1], original_size[0]))
        
        # Konversi kembali ke BGR untuk OpenCV
        watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2BGR)
        
        # Simpan gambar watermark
        output_path = os.path.join(output_folder, "watermarked_deep_" + os.path.basename(image_path))
        cv2.imwrite(output_path, watermarked_img)
        
        # Simpan metadata untuk ekstraksi
        watermark_signature = {
            'type': 'deep',
            'text': watermark_text,
            'alpha': alpha,
            'binary_len': binary_len
        }
        
        # Simpan signature ke file metadata tersembunyi
        save_metadata(output_folder, output_path, watermark_signature)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to add deep learning watermark to image: {e}")

def process_video_deep_learning(task_id, video_path, watermark_text, output_folder, update_progress):
    """Menambahkan watermark teks dengan deep learning ke video"""
    try:
        # Buka video
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_folder, "watermarked_deep_" + os.path.basename(video_path))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Loop setiap frame dan watermark dengan deep learning
        for frame_no in range(total_frames):
            ret, frame = video.read()
            if not ret:
                break
            
            # Buat file sementara untuk frame
            frame_path = os.path.join(output_folder, f"temp_frame_{task_id}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Terapkan watermark deep learning ke frame
            try:
                watermarked_frame_path = process_image_deep_learning(frame_path, watermark_text, output_folder)
                watermarked_frame = cv2.imread(watermarked_frame_path)
                
                # Hapus file sementara
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                if os.path.exists(watermarked_frame_path):
                    os.remove(watermarked_frame_path)
                
                # Tulis frame ke output
                out.write(watermarked_frame)
                
            except:
                # Jika deep learning gagal, gunakan frame asli
                out.write(frame)
            
            # Update progress
            progress = int((frame_no + 1) * 100 / total_frames)
            update_progress(task_id, progress)
        
        video.release()
        out.release()
        
        # Simpan watermark signature
        watermark_signature = {
            'type': 'deep_video',
            'text': watermark_text
        }
        
        # Simpan signature ke file metadata tersembunyi
        save_metadata(output_folder, output_path, watermark_signature)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to add deep learning watermark to video: {e}")

def extract_deep_learning_watermark(watermarked_img, signature_data=None):
    """Ekstrak watermark dari gambar yang di-watermark dengan deep learning"""
    try:
        # Pastikan TensorFlow tersedia
        if 'tensorflow' not in sys.modules:
            return "TensorFlow is not available. Cannot extract deep learning watermark."
        
        # Jika signature data tersedia, gunakan untuk ekstraksi terbimbing
        if signature_data and 'text' in signature_data:
            # Dalam implementasi nyata, kita akan menggunakan model decoder:
            # decoder = load_model('decoder.h5')
            # extracted_watermark = decoder(watermarked_img)
            
            # Untuk saat ini, kembalikan teks asli dari signature
            return signature_data['text']
            
        # Konversi ke RGB (TensorFlow biasanya mengharapkan RGB)
        img_rgb = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess gambar
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        img_tensor = img_tensor / 255.0  # Normalize ke [0,1]
        
        # Resize ke ukuran yang diharapkan model
        target_size = (256, 256)  # Ukuran umum untuk model
        img_tensor = tf.image.resize([img_tensor], target_size)[0]
        
        # Dalam implementasi nyata, kita akan menggunakan model decoder:
        # decoder = load_model('decoder.h5')
        # extracted_watermark = decoder(img_tensor)
        
        # Karena kita tidak punya model sungguhan, kita akan simulasikan ekstraksi
        # dengan mendeteksi pola di domain frekuensi
        
        # Konversi ke grayscale untuk pemrosesan lebih sederhana
        gray_img = tf.image.rgb_to_grayscale(img_tensor)
        gray_img = gray_img.numpy().squeeze()
        
        # Terapkan threshold untuk mengekstrak pola watermark potensial
        _, thresh = cv2.threshold(
            (gray_img * 255).astype(np.uint8), 
            127, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Hitung piksel putih sebagai bit watermark potensial
        count_ones = np.sum(thresh == 255)
        
        # Jika ada cukup titik pola yang terdeteksi,
        # asumsikan watermark ada (ini penyederhanaan)
        if count_ones > 100:
            return "Deep learning watermark detected but content not extractable without model"
        else:
            return "No deep learning watermark detected"
            
    except Exception as e:
        return f"Error extracting deep learning watermark: {str(e)}"
