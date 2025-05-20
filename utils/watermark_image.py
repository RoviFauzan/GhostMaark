import cv2
import numpy as np
import os
import time
import hashlib

def add_image_watermark(image_path, watermark_image_path, position='bottom-right', 
                        size_ratio=0.2, watermark_type='visible', technique='combined'):
    """
    Add image watermark to another image
    
    Args:
        image_path: Path to the image to watermark
        watermark_image_path: Path to the watermark image
        position: Position of watermark
        size_ratio: Size of watermark relative to image (0.0-1.0)
        watermark_type: 'visible' or 'steganographic'
        technique: Watermarking technique to use
        
    Returns:
        output_path: Path to the watermarked image
    """
    # Read the original image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Failed to open image file")
    
    # Read the watermark image with alpha channel if present
    watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_UNCHANGED)
    if watermark_img is None:
        raise Exception("Failed to open watermark image file")
    
    # Add alpha channel if not present
    if watermark_img.shape[2] == 3:
        alpha = np.ones((watermark_img.shape[0], watermark_img.shape[1]), dtype=watermark_img.dtype) * 255
        watermark_img = cv2.merge((watermark_img, alpha))
    
    # Get dimensions
    height, width = img.shape[:2]
    wm_height, wm_width = watermark_img.shape[:2]
    
    # Calculate watermark position
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-right':
        x, y = width - int(wm_width * size_ratio) - 10, 10
    elif position == 'bottom-left':
        x, y = 10, height - int(wm_height * size_ratio) - 10
    elif position == 'bottom-right':
        x, y = width - int(wm_width * size_ratio) - 10, height - int(wm_height * size_ratio) - 10
    else:  # center
        x, y = (width - int(wm_width * size_ratio)) // 2, (height - int(wm_height * size_ratio)) // 2
    
    # Apply watermark based on type
    if watermark_type == 'visible':
        # Resize watermark
        new_width = int(width * size_ratio)
        new_height = int(wm_height * new_width / wm_width)
        watermark_resized = cv2.resize(watermark_img, (new_width, new_height))
        
        # Create a copy of the image
        watermarked_img = img.copy()
        
        # Apply alpha blending
        alpha_channel = watermark_resized[:, :, 3] / 255.0
        alpha_channel = np.expand_dims(alpha_channel, axis=2)
        
        # Get region of interest
        roi = watermarked_img[y:y+new_height, x:x+new_width].copy()
        
        # Apply blending for each channel
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel[:, :, 0]) + \
                           watermark_resized[:, :, c] * alpha_channel[:, :, 0]
        
        # Update image
        watermarked_img[y:y+new_height, x:x+new_width] = roi
        
        # Save the watermarked image
        output_folder = os.path.dirname(image_path)
        output_filename = f"watermarked_{int(time.time())}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_folder, output_filename)
        
        cv2.imwrite(output_path, watermarked_img)
        
        # Return result info
        return output_path, x, y, new_width, new_height
    else:
        # For invisible watermarks, embed using LSB
        # Create a hash of the watermark for later extraction
        watermark_hash = hashlib.md5(watermark_img.tobytes()).hexdigest()
        
        # Resize watermark to smaller size
        new_width = max(width // 8, 8)
        new_height = max(int(new_width * wm_height / wm_width), 8)
        watermark_resized = cv2.resize(watermark_img, (new_width, new_height))
        
        # Convert to grayscale for embedding
        if watermark_resized.shape[2] >= 3:
            if watermark_resized.shape[2] == 4:  # With alpha
                alpha = watermark_resized[:, :, 3] / 255.0
                gray = cv2.cvtColor(watermark_resized[:, :, :3], cv2.COLOR_BGR2GRAY)
                gray = gray * alpha[:, :]
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            else:
                gray = cv2.cvtColor(watermark_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = watermark_resized
        
        # Binarize the watermark
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary_flat = binary.flatten()
        
        # Create a copy of the image
        watermarked_img = img.copy()
        flat_img = watermarked_img.flatten()
        
        # Embed the watermark hash and dimensions at the beginning
        # First 32 bytes for hash
        for i, char in enumerate(watermark_hash[:32]):
            if i < len(flat_img):
                flat_img[i] = (flat_img[i] & 0xFE) | int(char in '123456789abcdef')
        
        # Then embed width and height (16 bits each)
        width_bits = format(new_width, '016b')
        for i, bit in enumerate(width_bits):
            if 32 + i < len(flat_img):
                flat_img[32 + i] = (flat_img[32 + i] & 0xFE) | int(bit)
        
        height_bits = format(new_height, '016b')
        for i, bit in enumerate(height_bits):
            if 48 + i < len(flat_img):
                flat_img[48 + i] = (flat_img[48 + i] & 0xFE) | int(bit)
        
        # Embed the actual watermark bits
        for i, bit in enumerate(binary_flat):
            idx = 64 + i
            if idx < len(flat_img):
                flat_img[idx] = (flat_img[idx] & 0xFE) | (bit > 0)
        
        # Reshape back to original image
        watermarked_img = flat_img.reshape(img.shape)
        
        # Save the watermarked image
        output_folder = os.path.dirname(image_path)
        output_filename = f"watermarked_{int(time.time())}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_folder, output_filename)
        
        cv2.imwrite(output_path, watermarked_img)
        
        # Return result info - for invisible watermark, x,y are not used but return for consistency
        return output_path, 0, 0, new_width, new_height

def add_image_watermark_to_video(video_path, watermark_image_path, position='bottom-right', 
                                size_ratio=0.2, watermark_type='visible', technique='combined',
                                update_progress=None):
    """Add image watermark to a video"""
    # Create a cap object
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read the watermark image
    watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_UNCHANGED)
    
    # Add alpha channel if not present
    if watermark_img.shape[2] == 3:
        alpha = np.ones((watermark_img.shape[0], watermark_img.shape[1]), dtype=watermark_img.dtype) * 255
        watermark_img = cv2.merge((watermark_img, alpha))
    
    # Determine output path
    output_folder = os.path.dirname(video_path)
    output_filename = f"watermarked_{int(time.time())}_{os.path.basename(video_path)}"
    output_path = os.path.join(output_folder, output_filename)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply watermark to the frame
        if watermark_type == 'visible':
            # Calculate watermark size
            wm_height, wm_width = watermark_img.shape[:2]
            new_width = int(width * size_ratio)
            new_height = int(wm_height * new_width / wm_width)
            watermark_resized = cv2.resize(watermark_img, (new_width, new_height))
            
            # Calculate position
            if position == 'top-left':
                x, y = 10, 10
            elif position == 'top-right':
                x, y = width - new_width - 10, 10
            elif position == 'bottom-left':
                x, y = 10, height - new_height - 10
            elif position == 'bottom-right':
                x, y = width - new_width - 10, height - new_height - 10
            else:  # center
                x, y = (width - new_width) // 2, (height - new_height) // 2
            
            # Apply alpha blending
            alpha_channel = watermark_resized[:, :, 3] / 255.0
            alpha_channel = np.expand_dims(alpha_channel, axis=2)
            
            # Get region of interest
            roi = frame[y:y+new_height, x:x+new_width].copy()
            
            # Apply blending for each channel
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel[:, :, 0]) + \
                               watermark_resized[:, :, c] * alpha_channel[:, :, 0]
            
            # Update frame
            frame[y:y+new_height, x:x+new_width] = roi
        
        # Write the frame
        out.write(frame)
        
        # Update progress
        frame_count += 1
        if update_progress:
            progress = int(frame_count * 100 / total_frames)
            update_progress(progress)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path
