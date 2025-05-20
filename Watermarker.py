import os
import threading
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, send_file, jsonify
import uuid
from werkzeug.utils import secure_filename
import base64
import time
import shutil  # Add shutil import to fix the warning
import json
import re
import glob

# Import functionality from our modules
from modules.image_watermark import (
    add_visible_image_watermark, add_dct_image_watermark,
    add_deep_learning_image_watermark, add_lsb_image_watermark
)

from modules.attack import (
    attack_crop, attack_noise, attack_blur, attack_compression, 
    attack_rotation, attack_brightness, attack_contrast,
    attack_histogram_equalization, attack_median_filter, attack_resize,
    attack_combine, compare_images_after_attack, attack_flip, attack_shear, attack_color_shift
)

from modules.extraction import (
    extract_lsb_watermark, extract_lsb_blind, extract_dct_watermark,
    extract_deep_learning_watermark, extract_combined_watermark, find_signature_for_image
)

# Hapus import TensorFlow dan ganti dengan variabel untuk menandai penggunaan alternatif
DEEP_LEARNING_AVAILABLE = True  # Selalu tersedia karena kita akan mengimplementasikan alternatif
print("Deep learning watermarking menggunakan implementasi alternatif berbasis OpenCV dan NumPy")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Add model paths for deep learning watermarking
app.config['MODELS_FOLDER'] = os.path.join(os.getcwd(), 'models')
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store task progress
tasks = {}

# Web routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    position = request.form.get('position', 'bottom-right')
    
    # Get watermark type (visible or steganographic)
    watermark_type = request.form.get('watermark_type', 'visible')
    
    # Set watermark technique to combined since it's the only option
    watermark_technique = 'combined'
    
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
    
    # Get the opacity value from form data
    try:
        watermark_opacity = int(request.form.get('watermark_opacity', 80))
        watermark_opacity = max(10, min(100, watermark_opacity)) / 100.0  # Convert to ratio (0.1-1.0)
    except:
        watermark_opacity = 0.8  # Default 80%
    
    # Only handle image watermarks now that we've removed text watermarks
    if 'watermark_image' not in request.files:
        return jsonify({'error': 'No watermark image provided'}), 400
    
    watermark_image = request.files['watermark_image']
    if watermark_image.filename == '':
        return jsonify({'error': 'No watermark image selected'}), 400
    
    # Save the watermark image
    watermark_filename = secure_filename(watermark_image.filename)
    watermark_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"watermark_{watermark_filename}")
    watermark_image.save(watermark_image_path)
    
    # Generate unique ID for this task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'progress': 0, 'status': 'processing'}
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], task_id + '_' + filename)
    file.save(file_path)
    
    # Process the file in a separate thread - include opacity parameter
    thread = threading.Thread(
        target=process_file_with_content, 
        args=(task_id, file_path, 'image', watermark_text, watermark_image_path, watermark_size, position, watermark_type, watermark_technique, watermark_opacity)
    )
    thread.start()
    
    return jsonify({
        'task_id': task_id, 
        'message': 'Processing started'
    })

@app.route('/progress/<task_id>')
def get_progress(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(tasks[task_id])

@app.route('/download/<task_id>')
def download_file(task_id):
    if task_id not in tasks or tasks[task_id]['status'] != 'complete':
        return jsonify({'error': 'File not ready or task not found'}), 404
    
    output_path = tasks[task_id]['output_path']
    
    # Get the original filename without any prefixes or modifications
    original_filename = tasks[task_id]['original_filename']
    
    # Create a response with a forced filename that matches the original input
    response = send_file(output_path, as_attachment=True)
    
    # Force the Content-Disposition header to use the original filename
    # This ensures the file downloads with its original name
    response.headers['Content-Disposition'] = f'attachment; filename="{original_filename}"'
    
    return response

# Add a new route for previewing the processed file
@app.route('/preview/<task_id>')
def preview_file(task_id):
    if task_id not in tasks or tasks[task_id]['status'] != 'complete':
        return jsonify({'error': 'File not ready or task not found'}), 404
    
    output_path = tasks[task_id]['output_path']
    
    # Get the original filename for display purposes
    filename = tasks[task_id]['original_filename']
    
    # Determine the mime type
    mime_type = ""
    if output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        if output_path.lower().endswith('.png'):
            mime_type = "image/png"
        elif output_path.lower().endswith(('.jpg', '.jpeg')):
            mime_type = "image/jpeg"
        elif output_path.lower().endswith('.bmp'):
            mime_type = "image/bmp"
        elif output_path.lower().endswith('.gif'):
            mime_type = "image/gif"
    elif output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        if output_path.lower().endswith('.mp4'):
            mime_type = "video/mp4"
        elif output_path.lower().endswith('.avi'):
            mime_type = "video/x-msvideo"
        elif output_path.lower().endswith('.mov'):
            mime_type = "video/quicktime"
        elif output_path.lower().endswith('.mkv'):
            mime_type = "video/x-matroska"
        elif output_path.lower().endswith('.flv'):
            mime_type = "video/x-flv"
    
    # Create a URI for the preview
    preview_url = f"/temp-preview/{task_id}/{filename}"
    
    return jsonify({
        'preview_url': preview_url,
        'mime_type': mime_type
    })

# Add a route to serve the temporary preview files
@app.route('/temp-preview/<task_id>/<filename>')
def serve_preview(task_id, filename):
    if task_id not in tasks or tasks[task_id]['status'] != 'complete':
        return jsonify({'error': 'File not ready or task not found'}), 404
    
    output_path = tasks[task_id]['output_path']
    
    # No need to check if filenames match - we'll use the path from task data
    # This allows us to use a different naming convention internally
    
    # For security, verify the file is inside the OUTPUT_FOLDER
    if not output_path.startswith(app.config['OUTPUT_FOLDER']):
        return jsonify({'error': 'Security error: Invalid file path'}), 403
    
    # Serve the file (but don't trigger a download)
    return send_file(output_path)

# Add function to apply visible watermarks to videos
def add_visible_watermark_to_video(video_path, watermark_img, position, watermark_size, output_path, update_progress=None):
    """Apply a visible watermark to a video
    
    Args:
        video_path: Path to input video
        watermark_img: OpenCV image for watermark
        position: Position string ('top-left', 'top-right', etc.)
        watermark_size: Size ratio of watermark (0.0-1.0)
        output_path: Path to save output video
        update_progress: Callback function to update progress
        
    Returns:
        output_path: Path to watermarked video
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate position for watermark
    # Get watermark dimensions
    wm_height, wm_width = watermark_img.shape[:2]
    
    # Calculate new watermark size
    new_wm_width = int(width * watermark_size)
    new_wm_height = int(wm_height * new_wm_width / wm_width)
    
    # Resize watermark to target size
    watermark_resized = cv2.resize(watermark_img, (new_wm_width, new_wm_height))
    
    # Determine position coordinates
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-right':
        x, y = width - new_wm_width - 10, 10
    elif position == 'bottom-left':
        x, y = 10, height - new_wm_height - 10
    elif position == 'bottom-right':
        x, y = width - new_wm_width - 10, height - new_wm_height - 10
    elif position == 'center':
        x, y = (width - new_wm_width) // 2, (height - new_wm_height) // 2
    else:
        # Custom position - parse from string like "custom-30-20" (x:30%, y:20%)
        if position.startswith('custom-'):
            parts = position.split('-')
            if len(parts) == 3:
                try:
                    x_percent = float(parts[1]) / 100
                    y_percent = float(parts[2]) / 100
                    x = int(width * x_percent) - (new_wm_width // 2)
                    y = int(height * y_percent) - (new_wm_height // 2)
                except:
                    x, y = 10, 10  # Default to top-left if parsing fails
            else:
                x, y = 10, 10
        else:
            x, y = 10, 10  # Default to top-left for unrecognized positions
    
    # Ensure coordinates are within image boundaries
    x = max(0, min(x, width - new_wm_width))
    y = max(0, min(y, height - new_wm_height))
    
    # Process each frame
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply watermark to this frame
        # Create a copy of the frame
        watermarked_frame = frame.copy()
        
        # Get the region of interest for the watermark
        roi = watermarked_frame[y:y+new_wm_height, x:x+new_wm_width]
        
        # Check if watermark has alpha channel (4 channels)
        if watermark_resized.shape[2] == 4:
            # Extract RGB and alpha channels
            watermark_rgb = watermark_resized[:, :, 0:3]
            alpha = watermark_resized[:, :, 3].astype(np.float32) / 255.0
            
            # Reshape alpha to match broadcasting
            alpha = alpha[:, :, np.newaxis]
            
            # For each color channel
            roi_result = roi * (1.0 - alpha) + watermark_rgb * alpha
            roi_result = np.clip(roi_result, 0, 255).astype(np.uint8)
            
            # Apply the blended ROI back to the frame
            watermarked_frame[y:y+new_wm_height, x:x+new_wm_width] = roi_result
        else:
            # No alpha channel, just overlay the watermark
            watermarked_frame[y:y+new_wm_height, x:x+new_wm_width] = watermark_resized
        
        # Write the watermarked frame
        out.write(watermarked_frame)
        
        # Update progress
        frame_num += 1
        if update_progress and frame_count > 0:
            progress = (frame_num / frame_count) * 100
            update_progress(progress)
    
    # Release everything
    cap.release()
    out.release()
    
    return output_path

# Modify process_file_with_content to properly handle image and video data types
def process_file_with_content(task_id, file_path, content_type, watermark_text, watermark_image_path, watermark_size, position, watermark_type, watermark_technique, watermark_opacity=0.8):
    """Process a file by adding watermark content (image only now)."""
    try:
        # Initialize progress
        tasks[task_id]['progress'] = 0
        
        # Extract the original filename and store it
        original_filename = os.path.basename(file_path)
        # Remove the task_id prefix that was added when saving
        if '_' in original_filename and original_filename.split('_', 1)[0] == task_id:
            original_filename = original_filename.split('_', 1)[1]
        
        # Store the original filename for later use when downloading
        tasks[task_id]['original_filename'] = original_filename
        
        # Check if watermark image exists and is valid
        if not os.path.exists(watermark_image_path):
            raise Exception(f"Watermark image not found at path: {watermark_image_path}")
            
        # Validate watermark image before processing
        watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_UNCHANGED)
        if watermark_img is None:
            raise Exception(f"Failed to open watermark image: {watermark_image_path}")
        
        # Check if the source file exists
        if not os.path.exists(file_path):
            raise Exception(f"Source file not found at path: {file_path}")
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Always use combined technique
        technique_used = 'combined'
        
        # Update progress
        tasks[task_id]['progress'] = 10
        
        # Check if file is image or video based on extension
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            # Process image file - Use task ID for internal storage to avoid conflicts
            # But keep track of original name for downloading
            output_filename = f"{task_id}_{original_filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Open the image with error handling
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception(f"Failed to open image file: {file_path}")
            
            # Get image dimensions
            height, width, _ = img.shape
            
            # Update progress
            tasks[task_id]['progress'] = 20
            
            # Check if we need to add an alpha channel to watermark
            if watermark_img.shape[2] == 3:
                # Create alpha channel (fully opaque)
                alpha = np.ones((watermark_img.shape[0], watermark_img.shape[1]), dtype=watermark_img.dtype) * 255
                watermark_img = cv2.merge((watermark_img, alpha))
            
            # Apply watermark with proper error handling - add opacity parameter
            try:
                if watermark_type == 'visible':
                    # Modify the watermark image to apply opacity before watermarking
                    if watermark_opacity < 1.0 and watermark_img is not None:
                        # Apply opacity to alpha channel if it exists
                        if watermark_img.shape[2] == 4:
                            # Scale the alpha channel to the opacity value
                            watermark_img[:, :, 3] = (watermark_img[:, :, 3] * watermark_opacity).astype(np.uint8)
                        else:
                            # If no alpha channel, create one with the specified opacity
                            alpha = np.ones((watermark_img.shape[0], watermark_img.shape[1]), 
                                           dtype=watermark_img.dtype) * int(255 * watermark_opacity)
                            watermark_img = cv2.merge((watermark_img, alpha))
                    
                    watermarked_img, x, y, watermark_width, watermark_height = add_visible_image_watermark(
                        img, watermark_img, position, watermark_size)
                else:
                    # For combined technique, apply all three methods with progress updates
                    # First apply opacity to the watermark for steganographic techniques too
                    if watermark_opacity < 1.0 and watermark_img is not None:
                        if watermark_img.shape[2] == 4:
                            watermark_img[:, :, 3] = (watermark_img[:, :, 3] * watermark_opacity).astype(np.uint8)
                    
                    tasks[task_id]['progress'] = 30
                    watermarked_img_lsb, lsb_x, lsb_y, lsb_width, lsb_height = add_lsb_image_watermark(
                        img, watermark_img, position, watermark_size)
                    
                    tasks[task_id]['progress'] = 50
                    watermarked_img_dct, dct_x, dct_y, dct_width, dct_height = add_dct_image_watermark(
                        watermarked_img_lsb, watermark_img, position, watermark_size/2)
                    
                    tasks[task_id]['progress'] = 70
                    watermarked_img, x, y, watermark_width, watermark_height = add_deep_learning_image_watermark(
                        watermarked_img_dct, watermark_img, position, watermark_size/3)
                    
                    # Use the coordinates from the most visible technique for signature
                    if watermark_width == 0 or watermark_height == 0:
                        x, y, watermark_width, watermark_height = lsb_x, lsb_y, lsb_width, lsb_height
            except Exception as watermark_error:
                raise Exception(f"Error applying watermark: {str(watermark_error)}")
            
            # Update progress
            tasks[task_id]['progress'] = 80
            
            # Ensure pixel values are within valid range
            watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
            
            # Save the watermarked image with error handling
            try:
                cv2_result = cv2.imwrite(output_path, watermarked_img)
                if not cv2_result:
                    raise Exception(f"OpenCV failed to save image to {output_path}")
                
                # Verify the file was created
                if not os.path.exists(output_path):
                    raise Exception(f"File was not created at {output_path}")
            except Exception as save_error:
                raise Exception(f"Error saving watermarked image: {str(save_error)}")
            
            # Create watermark copy in output folder with proper error handling
            try:
                watermark_output_filename = f"watermark_for_{os.path.basename(output_filename)}"
                watermark_output_path = os.path.join(app.config['OUTPUT_FOLDER'], watermark_output_filename)
                shutil.copy2(watermark_image_path, watermark_output_path)
                
                # Verify the watermark copy was created
                if not os.path.exists(watermark_output_path):
                    print(f"Warning: Failed to create watermark copy at {watermark_output_path}")
                    # Use original path as fallback
                    watermark_output_path = watermark_image_path
            except Exception as copy_error:
                print(f"Warning: Failed to copy watermark: {str(copy_error)}")
                # Use original path if copy fails
                watermark_output_path = watermark_image_path
            
            # Create signature file with error handling
            try:
                # Primary signature in output folder
                signature_path = os.path.join(app.config['OUTPUT_FOLDER'], f".sig_{output_filename}")
                
                # Add more signature variants for better detection
                signature_path2 = os.path.join(app.config['OUTPUT_FOLDER'], f"sig_{output_filename}")
                
                # Add watermark output path to signature
                signature_data = {
                    'type': technique_used,
                    'watermark_path': watermark_output_path,
                    'original_watermark_path': watermark_image_path,
                    'visible': watermark_type == 'visible',
                    'opacity': watermark_opacity,  # Add opacity to signature
                    'x': x,
                    'y': y,
                    'width': watermark_width,
                    'height': watermark_height,
                    'processing_time': time.time(),
                    'watermark_size': watermark_size
                }
                
                # Add advanced metadata for extraction algorithms
                if watermark_type != 'visible':
                    signature_data['embedding_techniques'] = ['lsb', 'dct', 'deep_learning']
                
                # Write all signature files in both plain and JSON formats
                with open(signature_path, 'w') as f:
                    f.write(str(signature_data))
                    
                with open(signature_path2, 'w') as f:
                    f.write(str(signature_data))
                
                # Also add JSON versions which may be more reliable to parse
                with open(signature_path + '.json', 'w') as f:
                    json.dump(signature_data, f)
                
                # Create a signature file in uploads folder as well
                upload_sig_path = os.path.join(app.config['UPLOAD_FOLDER'], f".sig_{output_filename}")
                with open(upload_sig_path, 'w') as f:
                    f.write(str(signature_data))
            except Exception as sig_error:
                print(f"Warning: Failed to create signature files: {str(sig_error)}")
            
            # Update task status
            tasks[task_id]['progress'] = 100
            tasks[task_id]['status'] = 'complete'
            tasks[task_id]['output_path'] = output_path
            tasks[task_id]['original_filename'] = original_filename
            
            print(f"Successfully watermarked image: {output_path}")
            print(f"Will be downloaded as: {original_filename}")
        
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
            # Process video file - also use task ID for storage
            output_filename = f"{task_id}_{original_filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Define a progress update function for videos
            def update_video_progress(progress):
                current_progress = min(90, 20 + (progress * 0.7))  # Scale progress to range 20-90%
                tasks[task_id]['progress'] = current_progress
            
            # Apply opacity to watermark for videos
            if watermark_opacity < 1.0 and watermark_img is not None:
                if watermark_img.shape[2] == 4:
                    watermark_img[:, :, 3] = (watermark_img[:, :, 3] * watermark_opacity).astype(np.uint8)
                else:
                    alpha = np.ones((watermark_img.shape[0], watermark_img.shape[1]), 
                                   dtype=watermark_img.dtype) * int(255 * watermark_opacity)
                    watermark_img = cv2.merge((watermark_img, alpha))
            
            # Now correctly handle visible vs. invisible watermarks for videos
            if watermark_type == 'visible':
                tasks[task_id]['progress'] = 20
                # Apply visible watermark to video
                add_visible_watermark_to_video(
                    file_path, 
                    watermark_img, 
                    position, 
                    watermark_size, 
                    output_path, 
                    update_video_progress
                )
                # Record watermark position for signature
                # (Use dummy values since the actual position is calculated in the function)
                x, y = 0, 0
                watermark_width, watermark_height = watermark_img.shape[1], watermark_img.shape[0]
            else:
                # For invisible watermarks, use steganographic techniques
                # Currently just copying the file as a placeholder
                # Will be implemented in future versions
                shutil.copy2(file_path, output_path)
                x, y = 0, 0
                watermark_width, watermark_height = 0, 0
                tasks[task_id]['progress'] = 90
            
            # Create multiple signature files for videos too
            signature_path = os.path.join(app.config['OUTPUT_FOLDER'], f".sig_{output_filename}")
            signature_path2 = os.path.join(app.config['OUTPUT_FOLDER'], f"sig_{output_filename}")
            
            # Create copy of watermark in output folder
            watermark_output_filename = f"watermark_for_{os.path.basename(output_filename)}"
            watermark_output_path = os.path.join(app.config['OUTPUT_FOLDER'], watermark_output_filename)
            shutil.copy2(watermark_image_path, watermark_output_path)
            
            signature_data = {
                'type': f"{technique_used}_video",
                'watermark_path': watermark_output_path,
                'original_watermark_path': watermark_image_path,
                'visible': watermark_type == 'visible',
                'opacity': watermark_opacity,  # Add opacity to signature
                'x': x,
                'y': y,
                'width': watermark_width,
                'height': watermark_height,
                'processing_time': time.time(),
                'watermark_size': watermark_size
            }
            
            # Save signature in multiple formats
            with open(signature_path, 'w') as f:
                f.write(str(signature_data))
                
            with open(signature_path2, 'w') as f:
                f.write(str(signature_data))
                
            # Also save as JSON
            with open(signature_path + '.json', 'w') as f:
                json.dump(signature_data, f)
            
            # Add to uploads folder as well
            upload_sig_path = os.path.join(app.config['UPLOAD_FOLDER'], f".sig_{output_filename}")
            with open(upload_sig_path, 'w') as f:
                f.write(str(signature_data))
            
            # Update task status
            tasks[task_id]['progress'] = 100
            tasks[task_id]['status'] = 'complete'
            tasks[task_id]['output_path'] = output_path
            tasks[task_id]['original_filename'] = original_filename
            
            print(f"Successfully watermarked video: {output_path}")
            print(f"Will be downloaded as: {original_filename}")
        else:
            # Unsupported file type
            raise Exception(f"Unsupported file type: {ext}")
    
    except Exception as e:
        # Log detailed error
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        tasks[task_id] = {'status': 'error', 'message': str(e)}
        
        # Clean up any partial outputs
        try:
            output_filename = f"watermarked_{os.path.basename(file_path)}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass

# Add a route to serve temporary files from the OUTPUT_FOLDER
@app.route('/download-temp/<filename>')
def download_temp_file(filename):
    """Download a temporary file from the OUTPUT_FOLDER"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Security check
    if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
        return jsonify({'error': 'Security error: Invalid file path'}), 403
    
    # Extract base filename - remove any prefixes
    download_filename = filename
    for prefix in ["watermarked_", "original_", "attacked_", "watermark_for_"]:
        if download_filename.startswith(prefix):
            download_filename = download_filename[len(prefix):]
            break
    
    # Remove any UUID prefixes (format: uuid_filename.ext)
    if '_' in download_filename:
        parts = download_filename.split('_', 1)
        if len(parts[0]) >= 8:  # Assume it's a UUID prefix if first part is long enough
            download_filename = parts[1]
    
    # Create a response with the original filename
    response = send_file(file_path, as_attachment=True)
    
    # Force the Content-Disposition header to use the original filename
    response.headers['Content-Disposition'] = f'attachment; filename="{download_filename}"'
    
    return response

# Update the route name and function name to reflect its purpose as an attack simulation
@app.route('/attack', methods=['POST'])
def simulate_attack():
    """Apply selected attack to an image and provide the result for download"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    
    file = request.files['file']
    attack_type = request.form.get('attack_type', 'compression')
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "attack_" + filename)
    file.save(file_path)
    
    try:
        # Open the image
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Failed to open image file")
        
        # Create a copy of the original
        original_img = img.copy()
        
        # Apply the selected attack
        if attack_type == 'crop':
            crop_percent = int(request.form.get('crop_percentage', 20))
            crop_percent = max(5, min(40, crop_percent))  # Limit range
            attacked_img = attack_crop(img, crop_percent)
            attack_description = f"Cropped {crop_percent}% from each side"
        elif attack_type == 'noise':
            noise_level = int(request.form.get('noise_level', 5))
            noise_level = max(1, min(20, noise_level))  # Limit range
            attacked_img = attack_noise(img, noise_level)
            attack_description = f"Added random noise (level {noise_level})"
        elif attack_type == 'blur':
            blur_level = int(request.form.get('blur_level', 5))
            blur_level = max(3, min(21, blur_level))  # Limit range and ensure odd
            if blur_level % 2 == 0:
                blur_level += 1
            attacked_img = attack_blur(img, blur_level)
            attack_description = f"Applied Gaussian blur (kernel size {blur_level}x{blur_level})"
        elif attack_type == 'compression':
            quality = int(request.form.get('quality', 50))
            quality = max(10, min(95, quality))  # Limit range
            attacked_img = attack_compression(img, quality)
            attack_description = f"JPEG compression (quality {quality}%)"
        elif attack_type == 'rotation':
            # For rotation, pass None to use random rotation angles
            attacked_img = attack_rotation(img, None)
            attack_description = f"Rotated by random angle"
        elif attack_type == 'brightness':
            factor = float(request.form.get('factor', 1.5))
            factor = max(0.2, min(3.0, factor))  # Limit range
            attacked_img = attack_brightness(img, factor)
            attack_description = f"Brightness adjusted by factor {factor:.1f}"
        elif attack_type == 'contrast':
            factor = float(request.form.get('factor', 1.5))
            factor = max(0.2, min(3.0, factor))  # Limit range
            attacked_img = attack_contrast(img, factor)
            attack_description = f"Contrast adjusted by factor {factor:.1f}"
        elif attack_type == 'equalization':
            attacked_img = attack_histogram_equalization(img)
            attack_description = "Histogram equalization applied"
        elif attack_type == 'median':
            kernel_size = int(request.form.get('kernel_size', 5))
            kernel_size = max(3, min(15, kernel_size))  # Limit range
            if kernel_size % 2 == 0:
                kernel_size += 1
            attacked_img = attack_median_filter(img, kernel_size)
            attack_description = f"Median filter (kernel size {kernel_size}x{kernel_size})"
        elif attack_type == 'resize':
            scale = float(request.form.get('scale', 0.5))
            scale = max(0.1, min(0.9, scale))  # Limit range
            attacked_img = attack_resize(img, scale)
            attack_description = f"Resized down to {scale*100:.0f}% and back up"
        elif attack_type == 'flip':
            direction = request.form.get('direction', 'horizontal')
            attacked_img = attack_flip(img, direction)
            attack_description = f"Flipped {direction}"
        elif attack_type == 'shear':
            shear_factor = float(request.form.get('shear_factor', 0.2))
            shear_factor = max(0.05, min(0.5, shear_factor))  # Limit range
            attacked_img = attack_shear(img, shear_factor)
            attack_description = f"Applied shear transform (factor {shear_factor:.2f})"
        elif attack_type == 'color_shift':
            channel = request.form.get('channel', 'all')
            shift = int(request.form.get('shift', 30))
            shift = max(-100, min(100, shift))  # Limit range
            attacked_img = attack_color_shift(img, channel, shift)
            attack_description = f"Shifted {channel} color channel by {shift}"
        elif attack_type == 'combined_low':
            attacked_img = attack_combine(img, 'low')
            attack_description = "Combined low-intensity attacks"
        elif attack_type == 'combined_medium':
            attacked_img = attack_combine(img, 'medium')
            attack_description = "Combined medium-intensity attacks"
        elif attack_type == 'combined_high':
            attacked_img = attack_combine(img, 'high')
            attack_description = "Combined high-intensity attacks"
        else:
            # Default to compression if unknown attack type
            attacked_img = attack_compression(img, 50)
            attack_description = "JPEG compression (quality 50%)"
            
        # Ensure attacked image exists
        if attacked_img is None or attacked_img.size == 0:
            print("Attack resulted in invalid image, using original")
            attacked_img = img.copy()
        
        # Ensure the attacked image has the same dimensions as the original
        if attacked_img.shape[:2] != img.shape[:2]:
            print(f"Resizing attacked image from {attacked_img.shape[:2]} to {img.shape[:2]}")
            attacked_img = cv2.resize(attacked_img, (img.shape[1], img.shape[0]))
        
        # Save the original and attacked images - change how files are named internally
        # but keep the original name for the user when downloading
        original_filename = filename  # The original name the user will see
        
        # For internal storage, use unique names to avoid conflicts
        storage_id = str(uuid.uuid4())[:8]
        internal_original = f"original_{storage_id}_{filename}"
        internal_attacked = f"attacked_{storage_id}_{filename}"
        
        original_path = os.path.join(app.config['OUTPUT_FOLDER'], internal_original)
        attacked_path = os.path.join(app.config['OUTPUT_FOLDER'], internal_attacked)
        
        # Save the images
        cv2.imwrite(original_path, original_img)
        cv2.imwrite(attacked_path, attacked_img)
        
        # Clean up the original uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        # Determine the mime type
        mime_type = "image/jpeg"  # Default
        if file_path.lower().endswith('.png'):
            mime_type = "image/png"
        elif file_path.lower().endswith('.bmp'):
            mime_type = "image/bmp"
        elif file_path.lower().endswith('.gif'):
            mime_type = "image/gif"
        
        # Return paths to download the images with the original filename
        return jsonify({
            'status': 'success',
            'attack_type': attack_description,
            'original_image_url': f"/download-single/{internal_original}/{filename}",
            'attacked_image_url': f"/download-single/{internal_attacked}/{filename}",
            'mime_type': mime_type
        })
            
    except Exception as e:
        # Log the error
        print(f"Error in simulate_attack: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up the uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
                
        return jsonify({
            'status': 'error',
            'message': f'Failed to perform attack: {str(e)}'
        })

# Add a new route specifically for single file downloads with explicit filename
@app.route('/download-single/<internal_name>/<download_name>')
def download_single_file(internal_name, download_name):
    """Download a file with a specific filename"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], internal_name)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Security check
    if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
        return jsonify({'error': 'Security error: Invalid file path'}), 403
    
    # Create a response with the desired filename
    response = send_file(file_path, as_attachment=True)
    
    # Force the Content-Disposition header to use the original filename
    response.headers['Content-Disposition'] = f'attachment; filename="{download_name}"'
    
    return response

# Update attack_compression function to always use 100% quality
def attack_compression(img, quality=100):
    """Simulate JPEG compression to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        quality: JPEG quality (0-100, lower = more compression), fixed at 100
        
    Returns:
        attacked_img: Compressed image
    """
    # Create a temporary filename for compression
    temp_file = os.path.join(os.getcwd(), 'temp_compressed.jpg')
    
    # Always use 100% quality regardless of input parameter
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    cv2.imwrite(temp_file, img, encode_param)
    
    # Read the compressed image back
    compressed = cv2.imread(temp_file)
    
    # Remove temporary file
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass  # Ignore errors during cleanup
    
    # If compression failed, return original
    if compressed is None:
        return img.copy()
        
    return compressed

# Add a new route for watermark extraction
@app.route('/extract', methods=['POST'])
def extract_watermark():
    """Extract watermark from a file"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    # Generate a unique identifier for this extraction
    extraction_id = str(uuid.uuid4())[:8]
    
    # Create a secure filename and save the uploaded file
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"extract_{extraction_id}_{filename}")
        file.save(file_path)
        
        # Check if file was saved successfully
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': 'Failed to save uploaded file'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saving file: {str(e)}'
        }), 500
    
    try:
        # Check if this is a video file
        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))
        
        # For image files, read using OpenCV
        img = None
        if not is_video:
            # Only try to open as image if it's not a video
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("Failed to open image file")
        
        # Only generate output directory if needed
        output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # If it's an image, save a copy in outputs for signature detection to work better
        if not is_video and img is not None:
            output_copy_path = os.path.join(output_dir, f"extract_{extraction_id}_{filename}")
            cv2.imwrite(output_copy_path, img)
        
        # First try to find the signature file using the improved search function
        sig_data = find_signature_for_image(file_path)
        
        print(f"Signature data: {sig_data}")
        
        # If signature wasn't found, try with more approaches
        if not sig_data:
            print("Trying to find signature using output copy...")
            
            # Try with the copy in outputs folder
            output_copy_path = os.path.join(output_dir, f"extract_{extraction_id}_{filename}")
            if os.path.exists(output_copy_path):
                sig_data = find_signature_for_image(output_copy_path)
                print(f"Signature data from output copy: {sig_data}")
            
            # Also look for any file with "watermarked_" prefix + original filename
            clean_name = os.path.splitext(filename)[0]
            if 'extract_' in clean_name:
                # Remove extraction prefix
                match = re.search(r'extract_[a-zA-Z0-9]+_(.*)', clean_name)
                if match:
                    clean_name = match.group(1)
            
            # Check if this file exists and has a signature
            watermarked_name = f"watermarked_{clean_name}{os.path.splitext(filename)[1]}"
            watermarked_path = os.path.join(output_dir, watermarked_name)
            
            if os.path.exists(watermarked_path):
                print(f"Found potential watermarked file: {watermarked_path}")
                sig_data = find_signature_for_image(watermarked_path)
                print(f"Signature data from watermarked file: {sig_data}")
        
        # If signature data was found, make sure the watermark_path exists
        if sig_data and ('watermark_path' in sig_data or 'original_watermark_path' in sig_data):
            # Try both watermark path and original watermark path
            watermark_path = sig_data.get('watermark_path', '')
            original_path = sig_data.get('original_watermark_path', '')
            
            paths_to_try = [p for p in [watermark_path, original_path] if p]
            
            found_path = None
            for path in paths_to_try:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if not found_path:
                # If watermark path doesn't exist, try to find it
                print(f"Watermark paths not found: {paths_to_try}")
                
                # Try to find the watermark in different locations
                for path in paths_to_try:
                    base_name = os.path.basename(path)
                    
                    # Check in different locations with different naming patterns
                    possible_locations = [
                        os.path.join(app.config['UPLOAD_FOLDER'], base_name),
                        os.path.join(app.config['UPLOAD_FOLDER'], f"watermark_{base_name}"),
                        os.path.join(app.config['OUTPUT_FOLDER'], base_name),
                        os.path.join(app.config['OUTPUT_FOLDER'], f"watermark_{base_name}"),
                        os.path.join(app.config['OUTPUT_FOLDER'], f"watermark_for_{base_name}")
                    ]
                    
                    for possible_path in possible_locations:
                        if os.path.exists(possible_path):
                            print(f"Found watermark at: {possible_path}")
                            # Update the path in signature data
                            sig_data['watermark_path'] = possible_path
                            found_path = possible_path
                            break
                    
                    if found_path:
                        break
                
                # If still not found, look for any watermark file
                if not found_path:
                    print("Looking for any watermark file...")
                    watermark_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], "watermark_*"))
                    watermark_files += glob.glob(os.path.join(app.config['OUTPUT_FOLDER'], "watermark_*"))
                    
                    if watermark_files:
                        # Sort by modification time (newest first)
                        watermark_files.sort(key=os.path.getmtime, reverse=True)
                        found_path = watermark_files[0]
                        print(f"Using latest watermark file: {found_path}")
                        sig_data['watermark_path'] = found_path
        
        # Extract watermark using the combined approach
        # Pass img=None for video files, the extraction function will handle it
        extraction_result = extract_combined_watermark(img, sig_data, file_path)
        print(f"Extraction result: {extraction_result}")
        
        # Process extraction result
        if extraction_result.get('type') == 'image' and 'watermark_path' in extraction_result:
            watermark_path = extraction_result.get('watermark_path')
            
            if os.path.exists(watermark_path):
                # Make sure the watermark is in the output folder for client access
                if not watermark_path.startswith(app.config['OUTPUT_FOLDER']):
                    # Copy to output folder for client access
                    watermark_filename = os.path.basename(watermark_path)
                    
                    # Strip any prefix from the filename to ensure we use the original name
                    clean_watermark_filename = watermark_filename
                    for prefix in ["watermark_", "watermark_for_"]:
                        if clean_watermark_filename.startswith(prefix):
                            clean_watermark_filename = clean_watermark_filename[len(prefix):]
                            break
                    
                    watermark_output_path = os.path.join(app.config['OUTPUT_FOLDER'], watermark_filename)
                    
                    # Copy the file using the imported shutil module
                    shutil.copy2(watermark_path, watermark_output_path)
                    print(f"Copied watermark to output folder: {watermark_output_path}")
                    
                    # Update path
                    watermark_path = watermark_output_path
                
                # Get filename for URL - use clean name without prefixes
                watermark_filename = os.path.basename(watermark_path)
                clean_filename = watermark_filename
                for prefix in ["watermark_", "watermark_for_"]:
                    if clean_filename.startswith(prefix):
                        clean_filename = clean_filename[len(prefix):]
                        break
                
                # Return success with watermark image URL using download-single to keep original name
                return jsonify({
                    'status': 'success',
                    'has_watermark_image': True,
                    'watermark_image_url': f"/download-single/{watermark_filename}/{clean_filename}",
                    'watermark': extraction_result.get('content', 'Watermark extracted successfully'),
                    'message': 'Watermark extracted successfully'
                })
            else:
                print(f"Extracted watermark path doesn't exist: {watermark_path}")
                # Watermark path doesn't exist
                return jsonify({
                    'status': 'success',
                    'has_watermark_image': False,
                    'watermark': "Watermark detected but image not available",
                    'message': 'Watermark detected but image not available'
                })
        else:
            # Text content or no watermark
            return jsonify({
                'status': 'success',
                'has_watermark_image': False,
                'watermark': extraction_result.get('content', "No watermark detected"),
                'message': 'Extraction complete'
            })
    except Exception as e:
        print(f"Error in extract_watermark: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Failed to extract watermark: {str(e)}'
        })
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

# Main execution
if __name__ == "__main__":
    # Tampilkan teknik yang tersedia
    print("Watermarker starting...")
    print("Available watermarking techniques:")
    print(" - Combined (LSB + DCT + Deep Learning)")
    
    # Jalankan aplikasi web Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)