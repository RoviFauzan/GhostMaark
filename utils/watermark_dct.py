import os
import cv2
import numpy as np
from .file_utils import save_metadata

def process_image_dct(image_path, watermark_text, output_folder, alpha=0.1):
    """Menambahkan watermark teks dengan metode DCT (Discrete Cosine Transform)"""
    try:
        # Baca gambar
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Failed to open image")
            
        # Konversi ke YUV (DCT bekerja lebih baik di channel luminance)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        # Dapatkan channel Y (luminance)
        y_channel = img_yuv[:, :, 0]
        
        # Konversi watermark teks ke binary
        binary_watermark = ''.join(format(ord(char), '08b') for char in watermark_text)
        watermark_length = len(binary_watermark)
        
        # Tambahkan informasi panjang di awal (32 bit)
        length_bits = format(watermark_length, '032b')
        binary_data = length_bits + binary_watermark
        
        # Dimensi asli
        height, width = y_channel.shape
        
        # Pastikan gambar cukup besar untuk watermark
        if height < 8 or width < 8:
            raise Exception("Image too small for DCT watermarking")
        
        # Hitung jumlah blok 8x8
        h_blocks = height // 8
        w_blocks = width // 8
        total_blocks = h_blocks * w_blocks
        
        # Pastikan kita punya cukup blok untuk menyimpan watermark
        if total_blocks < len(binary_data):
            raise Exception("Image too small to hide this watermark")
        
        # Terapkan DCT dan embed watermark
        bit_index = 0
        dct_blocks = []
        
        # Simpan signature untuk ekstraksi
        watermark_signature = {
            'type': 'dct',
            'length': len(binary_data),
            'alpha': alpha,
            'text': watermark_text
        }
        
        # Proses setiap blok 8x8
        for i in range(h_blocks):
            for j in range(w_blocks):
                if bit_index >= len(binary_data):
                    break
                    
                # Ekstrak blok 8x8
                block = y_channel[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                
                # Terapkan DCT
                dct_block = cv2.dct(block)
                
                # Embed bit di koefisien frekuensi menengah (4,3)
                # Posisi ini dipilih untuk menyeimbangkan ketahanan dan ketersembunyian
                if binary_data[bit_index] == '1':
                    dct_block[4, 3] = abs(dct_block[4, 3]) + alpha * abs(dct_block[0, 0])
                else:
                    dct_block[4, 3] = -abs(dct_block[4, 3]) - alpha * abs(dct_block[0, 0])
                
                # Terapkan inverse DCT
                block = cv2.idct(dct_block)
                
                # Ganti blok di channel Y
                y_channel[i*8:(i+1)*8, j*8:(j+1)*8] = block
                
                bit_index += 1
                dct_blocks.append((i, j))  # Simpan posisi blok untuk signature
        
        # Update channel Y di gambar YUV
        img_yuv[:, :, 0] = y_channel
        
        # Konversi kembali ke BGR
        watermarked_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        # Simpan gambar watermark
        output_path = os.path.join(output_folder, "watermarked_dct_" + os.path.basename(image_path))
        cv2.imwrite(output_path, watermarked_img)
        
        # Tambahkan posisi blok ke signature untuk ekstraksi
        watermark_signature['blocks'] = dct_blocks[:len(binary_data)]
        
        # Simpan signature ke file metadata tersembunyi
        save_metadata(output_folder, output_path, watermark_signature)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to add DCT watermark to image: {e}")

def extract_dct_watermark(watermarked_img, signature_data=None):
    """Ekstrak watermark teks dari gambar yang di-watermark dengan DCT"""
    try:
        # Konversi ke YUV
        img_yuv = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YUV)
        
        # Dapatkan channel Y
        y_channel = img_yuv[:, :, 0]
        
        if signature_data and 'blocks' in signature_data:
            # Ekstraksi terbimbing menggunakan data signature
            blocks = signature_data.get('blocks', [])
            alpha = signature_data.get('alpha', 0.1)
            
            # Ekstrak 32 bit pertama untuk mendapatkan panjang
            length_bits = ''
            for idx, (i, j) in enumerate(blocks[:32]):
                block = y_channel[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Cek apakah koefisien positif atau negatif
                if dct_block[4, 3] > 0:
                    length_bits += '1'
                else:
                    length_bits += '0'
            
            # Konversi panjang biner ke integer
            try:
                watermark_length = int(length_bits, 2)
            except:
                return "Invalid length data in DCT watermark"
                
            # Sanity check panjangnya
            if watermark_length <= 0 or watermark_length > 10000:
                return "Invalid DCT watermark length"
                
            # Ekstrak bit watermark
            watermark_bits = ''
            for idx, (i, j) in enumerate(blocks[32:32+watermark_length]):
                block = y_channel[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Cek apakah koefisien positif atau negatif
                if dct_block[4, 3] > 0:
                    watermark_bits += '1'
                else:
                    watermark_bits += '0'
                    
                # Berhenti jika sudah mengekstrak semua bit
                if len(watermark_bits) >= watermark_length:
                    break
            
            # Konversi biner ke teks
            extracted_text = ''
            for i in range(0, len(watermark_bits), 8):
                if i + 8 <= len(watermark_bits):
                    byte = watermark_bits[i:i+8]
                    extracted_text += chr(int(byte, 2))
            
            return extracted_text
        else:
            # Jika signature tidak ada, coba ekstraksi buta
            height, width = y_channel.shape
            h_blocks = height // 8
            w_blocks = width // 8
            
            # Ekstrak 32 blok pertama untuk mendapatkan panjang
            length_bits = ''
            for i in range(min(4, h_blocks)):
                for j in range(min(8, w_blocks)):
                    if len(length_bits) >= 32:
                        break
                    
                    block = y_channel[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    
                    # Cek apakah koefisien frekuensi menengah signifikan dimodifikasi
                    if abs(dct_block[4, 3]) > 0.5 * abs(dct_block[0, 0]):
                        if dct_block[4, 3] > 0:
                            length_bits += '1'
                        else:
                            length_bits += '0'
            
            # Jika tidak bisa mendapatkan 32 bit, kemungkinan tidak ada watermark DCT
            if len(length_bits) < 32:
                return "No valid DCT watermark found"
                
            # Konversi panjang biner ke integer
            try:
                watermark_length = int(length_bits, 2)
            except:
                return "Invalid length data in DCT watermark"
                
            # Sanity check
            if watermark_length <= 0 or watermark_length > 1000:
                return "Invalid DCT watermark length"
                
            # Ekstrak bit watermark
            watermark_bits = ''
            bit_count = 0
            
            for i in range(h_blocks):
                for j in range(w_blocks):
                    # Lewati blok yang sudah diproses untuk panjang
                    if i < 4 and j < 8 and i*8 + j < 32:
                        continue
                        
                    if bit_count >= watermark_length:
                        break
                        
                    block = y_channel[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    
                    # Cek apakah koefisien frekuensi menengah signifikan dimodifikasi
                    if abs(dct_block[4, 3]) > 0.5 * abs(dct_block[0, 0]):
                        if dct_block[4, 3] > 0:
                            watermark_bits += '1'
                        else:
                            watermark_bits += '0'
                        bit_count += 1
            
            # Konversi biner ke teks
            extracted_text = ''
            for i in range(0, len(watermark_bits), 8):
                if i + 8 <= len(watermark_bits):
                    byte = watermark_bits[i:i+8]
                    extracted_text += chr(int(byte, 2))
            
            return extracted_text if extracted_text else "No valid DCT watermark found"
    except Exception as e:
        return f"Error extracting DCT watermark: {str(e)}"
