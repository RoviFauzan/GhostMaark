import os
import uuid
import threading
from flask import request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from utils.file_utils import is_image, is_video, save_metadata
from utils.watermark_image import add_image_watermark, add_image_watermark_to_video
from utils.watermark_lsb import process_image_lsb, process_video_steganography
from utils.watermark_dct import process_image_dct, extract_dct_watermark
from utils.watermark_deep import process_image_deep_learning, process_video_deep_learning, extract_deep_learning_watermark

def setup_watermark_routes(app, config, tasks):
    """Mengatur rute untuk watermarking dan ekstraksi"""
    
    # Fungsi untuk update progres task
    def update_progress(task_id, progress):
        """Update progres untuk task tertentu"""
        if task_id in tasks:
            tasks[task_id]['progress'] = progress
    
    # Helper function untuk memproses file
    def process_file(task_id, file_path, watermark_text, position, watermark_type='visible'):
        try:
            if is_image(file_path):
                if watermark_type == 'deep':
                    output_path = process_image_deep_learning(file_path, watermark_text, config.OUTPUT_FOLDER)
                elif watermark_type == 'dct':
                    output_path = process_image_dct(file_path, watermark_text, config.OUTPUT_FOLDER)
                elif watermark_type == 'steganographic':
                    output_path = process_image_lsb(file_path, watermark_text, config.OUTPUT_FOLDER)
                else:
                    output_path = add_text_watermark(file_path, watermark_text, position, 128, config.OUTPUT_FOLDER)
            elif is_video(file_path):
                if watermark_type == 'deep':
                    output_path = process_video_deep_learning(task_id, file_path, watermark_text, config.OUTPUT_FOLDER, update_progress)
                elif watermark_type == 'steganographic':
                    output_path = process_video_steganography(task_id, file_path, watermark_text, config.OUTPUT_FOLDER, update_progress)
                else:
                    output_path = add_text_watermark_to_video(task_id, file_path, watermark_text, position, 128, config.OUTPUT_FOLDER, update_progress)
            else:
                tasks[task_id] = {'progress': 0, 'status': 'error', 'message': 'Unsupported file type'}
                return
            
            # Update task status
            tasks[task_id] = {
                'progress': 100, 
                'status': 'complete', 
                'output_path': output_path,
                'filename': os.path.basename(output_path)
            }
        except Exception as e:
            tasks[task_id] = {
                'progress': 0, 
                'status': 'error', 
                'message': f'Error: {str(e)}'
            }
    
    def process_file_with_content(task_id, file_path, content_type, watermark_text, watermark_image_path, 
                                 watermark_size, position, watermark_type, watermark_technique='standard'):
        try:
            if is_image(file_path):
                if content_type == 'text':
                    if watermark_type == 'steganographic':
                        if watermark_technique == 'deep':
                            output_path = process_image_deep_learning(file_path, watermark_text, config.OUTPUT_FOLDER)
                        elif watermark_technique == 'dct':
                            output_path = process_image_dct(file_path, watermark_text, config.OUTPUT_FOLDER)
                        else:
                            output_path = process_image_lsb(file_path, watermark_text, config.OUTPUT_FOLDER)
                    else:
                        # Visible text watermark
                        opacity = 128 if watermark_type == 'visible' else 0
                        output_path = add_text_watermark(file_path, watermark_text, position, opacity, config.OUTPUT_FOLDER)
                else:  # Image watermark
                    visible = (watermark_type == 'visible')
                    output_path = add_image_watermark(file_path, watermark_image_path, watermark_size, position, visible, config.OUTPUT_FOLDER)
            elif is_video(file_path):
                if content_type == 'text':
                    if watermark_type == 'steganographic':
                        if watermark_technique == 'deep':
                            output_path = process_video_deep_learning(task_id, file_path, watermark_text, config.OUTPUT_FOLDER, update_progress)
                        else:
                            output_path = process_video_steganography(task_id, file_path, watermark_text, config.OUTPUT_FOLDER, update_progress)
                    else:
                        # Visible text watermark
                        opacity = 128 if watermark_type == 'visible' else 0
                        output_path = add_text_watermark_to_video(task_id, file_path, watermark_text, position, opacity, config.OUTPUT_FOLDER, update_progress)
                else:  # Image watermark
                    visible = (watermark_type == 'visible')
                    output_path = add_image_watermark_to_video(task_id, file_path, watermark_image_path, watermark_size, position, visible, config.OUTPUT_FOLDER, update_progress)
            else:
                tasks[task_id] = {'progress': 0, 'status': 'error', 'message': 'Unsupported file type'}
                return
            
            # Bersihkan watermark image jika diperlukan
            if content_type == 'image' and watermark_image_path and os.path.exists(watermark_image_path):
                try:
                    os.remove(watermark_image_path)
                except:
                    pass  # Abaikan error saat membersihkan
            
            # Update task status
            tasks[task_id] = {
                'progress': 100, 
                'status': 'complete', 
                'output_path': output_path,
                'filename': os.path.basename(output_path)
            }
        except Exception as e:
            tasks[task_id] = {
                'progress': 0, 
                'status': 'error', 
                'message': f'Error: {str(e)}'
            }
    
    @app.route('/process', methods=['POST'])
    def process():
        """Rute untuk memproses file dan menambahkan watermark"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        position = request.form.get('position', 'bottom-right')
        
        # Get watermark type (visible or steganographic)
        watermark_type = request.form.get('watermark_type', 'visible')
        
        # Get content type (text or image)
        content_type = request.form.get('content_type', 'text')
        
        # Get watermark technique
        watermark_technique = request.form.get('watermark_technique', 'standard')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check watermark content based on type
        watermark_text = None
        watermark_image_path = None
        
        # Set a default watermark size regardless of content type
        try:
            watermark_size = int(request.form.get('watermark_size', 20))
            watermark_size = max(5, min(50, watermark_size)) / 100.0  # Convert to ratio (0.05-0.5)
        except:
            watermark_size = 0.2  # Default 20%
        
        if content_type == 'text':
            watermark_text = request.form.get('watermark_text', '')
            if not watermark_text:
                return jsonify({'error': 'No watermark text provided'}), 400
        else:  # image watermark
            if 'watermark_image' not in request.files:
                return jsonify({'error': 'No watermark image provided'}), 400
            
            watermark_image = request.files['watermark_image']
            if watermark_image.filename == '':
                return jsonify({'error': 'No watermark image selected'}), 400
            
            # Save the watermark image
            watermark_filename = secure_filename(watermark_image.filename)
            watermark_image_path = os.path.join(config.UPLOAD_FOLDER, f"watermark_{watermark_filename}")
            watermark_image.save(watermark_image_path)
        
        # Generate unique ID for this task
        task_id = str(uuid.uuid4())
        tasks[task_id] = {'progress': 0, 'status': 'processing'}
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(config.UPLOAD_FOLDER, task_id + '_' + filename)
        file.save(file_path)
        
        # Process the file in a separate thread
        if watermark_type == 'steganographic' and content_type == 'text':
            if watermark_technique == 'dct':
                thread = threading.Thread(
                    target=process_file, 
                    args=(task_id, file_path, watermark_text, position, 'dct')
                )
            elif watermark_technique == 'deep':
                thread = threading.Thread(
                    target=process_file, 
                    args=(task_id, file_path, watermark_text, position, 'deep')
                )
            else:
                thread = threading.Thread(
                    target=process_file, 
                    args=(task_id, file_path, watermark_text, position, 'steganographic')
                )
        else:
            # For all other combinations
            thread = threading.Thread(
                target=process_file_with_content, 
                args=(task_id, file_path, content_type, watermark_text, watermark_image_path, 
                      watermark_size, position, watermark_type, watermark_technique)
            )
        thread.start()
        
        return jsonify({
            'task_id': task_id, 
            'message': 'Processing started'
        })

    # Rute untuk ekstraksi watermark
    @app.route('/extract', methods=['POST'])
    def extract_watermark():
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the uploaded file (this is the watermarked file)
        filename = secure_filename(file.filename)
        file_path = os.path.join(config.UPLOAD_FOLDER, "extract_" + filename)
        file.save(file_path)
        
        try:
            # Open the watermarked image
            watermarked_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if watermarked_img is None:
                raise Exception("Failed to open image file")
            
            # Make a copy for the cleaned version
            cleaned_img = watermarked_img.copy()
            
            # Create a black image for the watermark pattern
            pattern_img = np.zeros_like(watermarked_img)
            
            height, width = watermarked_img.shape[:2]
            
            # Check if there's a signature file for this image in our database
            # Look for similar filenames in the output folder (might be renamed by user)
            ext = os.path.splitext(filename)[1]
            signature_files = [f for f in os.listdir(config.OUTPUT_FOLDER) 
                              if f.startswith('.sig_watermarked_') and f.endswith(ext)]
            
            found_signature = False
            watermark_signature = None
            watermark_text = ""
            
            # Implementasi deteksi watermark dari signature
            # ... (kode deteksi watermark)
            
            # Untuk versi skema terpisah, kode panjang deteksi watermark dapat ditempatkan di file terpisah
            # Dan di sini cukup memanggil fungsi detect_watermark() dari modul terpisah
            
            # Menyimpan tiga versi gambar
            watermarked_filename = "watermarked_" + filename
            cleaned_filename = "cleaned_" + filename
            pattern_filename = "pattern_" + filename
            
            watermarked_path = os.path.join(config.OUTPUT_FOLDER, watermarked_filename)
            cleaned_path = os.path.join(config.OUTPUT_FOLDER, cleaned_filename)
            pattern_path = os.path.join(config.OUTPUT_FOLDER, pattern_filename)
            
            cv2.imwrite(watermarked_path, watermarked_img)
            cv2.imwrite(cleaned_path, cleaned_img)
            cv2.imwrite(pattern_path, pattern_img)
            
            # Dapatkan mime type
            from utils.file_utils import get_mime_type
            mime_type = get_mime_type(file_path)
            
            # Clean up the original uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'status': 'success',
                'watermark': watermark_text,
                'original_image_url': f"/download-temp/{watermarked_filename}",
                'cleaned_image_url': f"/download-temp/{cleaned_filename}",
                'watermark_pattern_url': f"/download-temp/{pattern_filename}",
                'mime_type': mime_type
            })
        except Exception as e:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'status': 'error',
                'message': f'Failed to extract watermark: {str(e)}'
            }), 400
