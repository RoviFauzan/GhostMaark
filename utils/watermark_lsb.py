import os
import cv2
import numpy as np
from .file_utils import save_metadata

def process_image_lsb(image_path, watermark_text, output_folder):
    """Menambahkan watermark teks dengan metode LSB (Least Significant Bit)"""
    try:
        # Konversi teks ke binary
        binary_watermark = ''.join(format(ord(char), '08b') for char in watermark_text)
        
        # Buka gambar
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Cek apakah gambar cukup piksel untuk menyimpan watermark
        height, width, _ = img.shape
        if len(binary_watermark) > height * width:
            raise Exception("The image is too small to hide this watermark")
        
        # Flatten gambar untuk bekerja dengan piksel secara linear
        img_flat = img.flatten()
        
        # Tambahkan panjang watermark di awal (32 bit untuk panjang)
        watermark_length = format(len(binary_watermark), '032b')
        binary_data = watermark_length + binary_watermark
        
        # Pastikan kita punya cukup ruang
        if len(binary_data) > len(img_flat):
            raise Exception("The image is too small to hide this watermark")
        
        # Embed data biner di LSB setiap komponen piksel
        for i in range(len(binary_data)):
            # Dapatkan nilai piksel
            pixel_value = img_flat[i]
            
            # Hapus LSB dengan AND dengan 11111110
            pixel_value &= ~1
            
            # Set LSB sesuai bit data biner
            pixel_value |= int(binary_data[i])
            
            # Update gambar
            img_flat[i] = pixel_value
        
        # Reshape array flatten kembali ke bentuk aslinya
        img = img_flat.reshape(height, width, 3)
        
        # Simpan gambar watermark
        output_path = os.path.join(output_folder, "watermarked_lsb_" + os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        
        # Simpan watermark signature untuk ekstraksi
        watermark_signature = {
            'type': 'lsb',
            'text': watermark_text,
            'length': len(binary_watermark)
        }
        
        # Simpan signature ke file metadata tersembunyi
        save_metadata(output_folder, output_path, watermark_signature)
            
        return output_path
    except Exception as e:
        raise Exception(f"Failed to add LSB watermark to image: {e}")

def process_video_steganography(task_id, video_path, watermark_text, output_folder, update_progress):
    """Menambahkan watermark teks dengan steganografi ke video"""
    try:
        # Konversi teks ke binary
        binary_watermark = ''.join(format(ord(char), '08b') for char in watermark_text)
        
        # Buka video
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_folder, "watermarked_stego_" + os.path.basename(video_path))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Hitung berapa bit yang bisa kita embed di setiap frame
        frame_capacity = (width * height * 3) // 100  # Gunakan hanya 1% dari piksel
        
        # Hitung berapa frame yang kita butuhkan untuk embed watermark
        frames_needed = (len(binary_watermark) // frame_capacity) + 1
        
        # Pastikan kita punya cukup frame
        if frames_needed > total_frames:
            raise Exception("The video doesn't have enough frames to hide this watermark")
        
        # Tambahkan panjang watermark (32 bit) di awal
        watermark_length = format(len(binary_watermark), '032b')
        binary_data = watermark_length + binary_watermark
        
        # Embed watermark
        frame_index = 0
        bit_index = 0
        
        for frame_no in range(total_frames):
            ret, frame = video.read()
            if not ret:
                break
            
            # Untuk frame tertentu, embed bit watermark
            if frame_index < frames_needed:
                # Flatten frame
                frame_flat = frame.flatten()
                
                # Jumlah bit untuk di-embed di frame ini
                bits_to_embed = min(frame_capacity, len(binary_data) - bit_index)
                
                if bits_to_embed > 0:
                    # Embed bit di frame ini
                    for i in range(bits_to_embed):
                        pixel_index = np.random.randint(0, len(frame_flat))
                        
                        # Hapus LSB dengan AND dengan 11111110
                        frame_flat[pixel_index] &= ~1
                        
                        # Set LSB sesuai bit data biner
                        if bit_index < len(binary_data):
                            frame_flat[pixel_index] |= int(binary_data[bit_index])
                            bit_index += 1
                    
                    # Reshape frame kembali
                    frame = frame_flat.reshape(height, width, 3)
                    frame_index += 1
            
            # Tulis frame (dimodifikasi atau tidak)
            out.write(frame)
            
            # Update progress
            progress = int((frame_no + 1) * 100 / total_frames)
            update_progress(task_id, progress)
        
        video.release()
        out.release()
        
        # Simpan watermark signature untuk ekstraksi
        watermark_signature = {
            'type': 'video_stego',
            'text': watermark_text,
            'length': len(binary_watermark),
            'frame_capacity': frame_capacity,
            'frames_needed': frames_needed
        }
        
        # Simpan signature ke file metadata tersembunyi
        save_metadata(output_folder, output_path, watermark_signature)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to add steganographic watermark to video: {e}")
