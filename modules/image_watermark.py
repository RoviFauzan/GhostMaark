import cv2
import numpy as np
import hashlib

def add_visible_image_watermark(img, watermark_img, position, size_ratio=0.2):
    """Add visible image watermark to an image
    
    Args:
        img: OpenCV image (target image)
        watermark_img: OpenCV image (watermark to apply)
        position: Position of watermark ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
        size_ratio: Size of watermark relative to target image (0.0-1.0)
        
    Returns:
        watermarked_img: Image with watermark
        x, y: Position of watermark
        width, height: Size of the watermark
    """
    # Create a copy of the image to avoid modifying the original
    watermarked_img = img.copy()
    
    # Get dimensions
    img_h, img_w = img.shape[:2]
    wm_h, wm_w = watermark_img.shape[:2]
    
    # Calculate watermark size based on ratio and target image width
    new_wm_w = max(int(img_w * size_ratio), 1)
    new_wm_h = max(int(wm_h * new_wm_w / wm_w), 1)
    
    # Resize watermark image
    watermark_resized = cv2.resize(watermark_img, (new_wm_w, new_wm_h))
    
    # Calculate position coordinates based on placement option
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-right':
        x, y = img_w - new_wm_w - 10, 10
    elif position == 'bottom-left':
        x, y = 10, img_h - new_wm_h - 10
    elif position == 'bottom-right':
        x, y = img_w - new_wm_w - 10, img_h - new_wm_h - 10
    else:  # center
        x, y = (img_w - new_wm_w) // 2, (img_h - new_wm_h) // 2
    
    # Ensure coordinates are within image boundaries
    x = max(0, min(x, img_w - new_wm_w))
    y = max(0, min(y, img_h - new_wm_h))
    
    # Make sure watermark dimensions don't exceed image boundaries
    effective_width = min(new_wm_w, img_w - x)
    effective_height = min(new_wm_h, img_h - y)
    
    # Check for zero dimensions to avoid errors
    if effective_width <= 0 or effective_height <= 0:
        return watermarked_img, x, y, new_wm_w, new_wm_h
        
    # Check if watermark has alpha channel (4 channels)
    try:
        if watermark_resized.shape[2] == 4:
            # Extract RGB and alpha channels
            watermark_rgb = watermark_resized[:effective_height, :effective_width, 0:3]
            alpha = watermark_resized[:effective_height, :effective_width, 3].astype(np.float32) / 255.0
            
            # Get the region of interest for the watermark
            roi = watermarked_img[y:y+effective_height, x:x+effective_width].copy()
            
            # For each color channel
            for c in range(3):
                # Vectorized alpha blending with proper type casting and range checking
                blended = (roi[:,:,c].astype(np.float32) * (1.0 - alpha) + 
                          watermark_rgb[:,:,c].astype(np.float32) * alpha)
                
                # Ensure the values are within the valid range [0, 255]
                roi[:,:,c] = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Apply the blended ROI back to the image
            watermarked_img[y:y+effective_height, x:x+effective_width] = roi
        else:
            # No alpha channel, just overlay the watermark directly
            # Ensure the watermark is cropped to the effective dimensions
            watermarked_img[y:y+effective_height, x:x+effective_width] = watermark_resized[:effective_height, :effective_width]
    except Exception as e:
        print(f"Error applying visible watermark: {e}")
        # If an error occurs, return the original image
        return img.copy(), x, y, new_wm_w, new_wm_h
    
    return watermarked_img, x, y, new_wm_w, new_wm_h

def add_lsb_image_watermark(img, watermark_img, position, size_ratio=0.2):
    """Add image watermark using LSB steganography"""
    try:
        # Create a copy of the image
        watermarked_img = img.copy()
        height, width, channels = img.shape
        
        # Resize watermark to target size
        wm_height, wm_width = watermark_img.shape[:2]
        new_wm_width = max(int(width * size_ratio), 1)
        new_wm_height = max(int(wm_height * new_wm_width / wm_width), 1)
        
        # Safely resize watermark
        watermark_resized = cv2.resize(watermark_img, (new_wm_width, new_wm_height))
        
        # Convert watermark to binary data
        # First save metadata about dimensions
        wm_data = [new_wm_width, new_wm_height]
        
        # Flatten the resized watermark for embedding
        if len(watermark_resized.shape) > 2 and watermark_resized.shape[2] == 4:  # Has alpha channel
            for i in range(4):  # RGBA
                wm_data.extend(watermark_resized[:, :, i].flatten().tolist())
        elif len(watermark_resized.shape) > 2 and watermark_resized.shape[2] == 3:  # RGB only
            for i in range(3):  # RGB
                wm_data.extend(watermark_resized[:, :, i].flatten().tolist())
        else:  # Grayscale
            wm_data.extend(watermark_resized.flatten().tolist())
        
        # Check if image has enough capacity
        max_capacity = height * width * channels
        data_size = len(wm_data)
        
        if data_size > max_capacity:
            print(f"Warning: Image too small for watermark. Truncating watermark data.")
            wm_data = wm_data[:max_capacity]
            data_size = max_capacity
        
        # Prepare for LSB embedding - use uint16 to avoid overflow issues
        flat_img = watermarked_img.flatten().astype(np.uint16)
        
        # Make the LSB technique more robust by using multiple bits and spreading them
        # Increase redundancy by repeating important bits across the image
        # We'll use the lowest 2 bits instead of just 1 bit for better persistence
        for i in range(min(2, len(wm_data))):
            value = wm_data[i]
            for bit_pos in range(32):
                idx = i*32 + bit_pos
                if idx < flat_img.size:  # Ensure we're within bounds
                    # Store in both lowest and second-lowest bits for redundancy
                    bit = (value >> bit_pos) & 1
                    flat_img[idx] = (flat_img[idx] & 0xFFFC) | (bit << 1) | bit
        
        # Now embed actual watermark data with redundancy
        offset = 64  # Start after dimension data
        for i in range(2, min(data_size, (max_capacity - offset) // 2)):
            if offset + i*2 < flat_img.size:  # Safety check
                # Store each bit twice for redundancy
                bit = wm_data[i] & 1
                flat_img[offset + i*2] = (flat_img[offset + i*2] & 0xFFFE) | bit
                # Store same bit again in another location
                if offset + i*2 + 1 < flat_img.size:
                    flat_img[offset + i*2 + 1] = (flat_img[offset + i*2 + 1] & 0xFFFE) | bit
        
        # Convert back to original shape and ensure uint8 range
        watermarked_img = np.clip(flat_img.reshape(img.shape), 0, 255).astype(np.uint8)
        
        return watermarked_img, 0, 0, width, height
    except Exception as e:
        print(f"Error in LSB watermarking: {e}")
        # If anything fails, return the original image
        return img.copy(), 0, 0, img.shape[1], img.shape[0]

def add_dct_image_watermark(img, watermark_img, position, size_ratio=0.2):
    """Add image watermark using DCT (Discrete Cosine Transform)"""
    try:
        # Create a copy of the image
        watermarked_img = img.copy()
        height, width, _ = img.shape
        
        # Resize watermark to smaller size for embedding
        wm_height, wm_width = watermark_img.shape[:2]
        new_wm_width = max(int(width * size_ratio / 2), 1)
        new_wm_height = max(int(wm_height * new_wm_width / wm_width), 1)
        
        # Safely resize watermark
        watermark_resized = cv2.resize(watermark_img, (new_wm_width, new_wm_height))
        
        # Convert to grayscale for simpler embedding
        if len(watermark_resized.shape) > 2:
            if watermark_resized.shape[2] == 4:  # Has alpha channel
                # Use alpha as weighting factor
                alpha = watermark_resized[:, :, 3].astype(np.float32) / 255.0
                wm_gray = cv2.cvtColor(watermark_resized[:, :, 0:3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                wm_gray = wm_gray * alpha[:, :, np.newaxis] if alpha.ndim < wm_gray.ndim else wm_gray * alpha
            else:
                wm_gray = cv2.cvtColor(watermark_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:  # Already grayscale
            wm_gray = watermark_resized.astype(np.float32) / 255.0
        
        # Normalize watermark to [0, 1]
        wm_gray = np.reshape(wm_gray, (new_wm_height, new_wm_width))  # Ensure 2D shape
        
        # Convert the image to YUV for DCT (works better in Y channel)
        img_yuv = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YUV).astype(np.float32)
        y_channel = img_yuv[:, :, 0].copy()  # Work on a copy to avoid modifying original
        
        # Determine block size and check capacity
        block_size = 8
        
        # Safety check for dimensions
        if wm_gray.size == 0 or height < block_size or width < block_size:
            return watermarked_img, 0, 0, width, height
        
        # Create seed from watermark for reproducibility
        seed = hash(str(wm_gray.shape) + str(np.sum(wm_gray))) % 10000
        np.random.seed(seed)
        
        alpha = 0.1  # Embedding strength - keep small for invisibility
        
        # Use block coordinates for embedding locations
        h_blocks = height // block_size
        w_blocks = width // block_size
        
        # Flatten watermark for sequential embedding
        wm_flat = wm_gray.flatten()
        wm_size = len(wm_flat)
        
        # Limit the number of blocks to modify to avoid errors
        num_blocks = min(wm_size, h_blocks * w_blocks // 4)  # Use at most 1/4 of blocks
        
        # Generate random block positions without repetition
        block_indices = np.random.choice(h_blocks * w_blocks, num_blocks, replace=False)
        
        # Modify each selected block
        for i, block_idx in enumerate(block_indices):
            if i >= wm_size:
                break
                
            # Convert 1D index to 2D block coordinates
            block_y = (block_idx // w_blocks) * block_size
            block_x = (block_idx % w_blocks) * block_size
            
            # Ensure block fits within image
            if block_y + block_size > height or block_x + block_size > width:
                continue
                
            # Extract block
            block = y_channel[block_y:block_y+block_size, block_x:block_x+block_size].copy()
            
            # Apply DCT
            dct_block = cv2.dct(block)
            
            # Modify multiple mid-frequency coefficients for redundancy
            wm_value = wm_flat[i]
            
            # Only modify if we're within a safe range
            if 4 < dct_block.shape[0] and 4 < dct_block.shape[1]:
                # Modify multiple coefficients for better robustness
                # Use both (4,4) and (3,3) coefficients for redundancy
                dct_block[4, 4] = dct_block[4, 4] * (1.0 + alpha * wm_value)
                dct_block[3, 3] = dct_block[3, 3] * (1.0 + alpha * wm_value)
                # Add a third point for triple redundancy
                dct_block[5, 3] = dct_block[5, 3] * (1.0 + alpha * wm_value)
            
            # Apply inverse DCT and ensure no out-of-bounds values
            idct_block = cv2.idct(dct_block)
            idct_block = np.clip(idct_block, 0, 255)
            
            # Replace block in Y channel
            y_channel[block_y:block_y+block_size, block_x:block_x+block_size] = idct_block
        
        # Update Y channel in YUV image
        img_yuv[:, :, 0] = y_channel
        
        # Convert back to BGR
        watermarked_img = cv2.cvtColor(img_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
        
        return watermarked_img, 0, 0, width, height
    except Exception as e:
        print(f"Error in DCT watermarking: {e}")
        # If anything fails, return the original image
        return img.copy(), 0, 0, img.shape[1], img.shape[0]

def add_deep_learning_image_watermark(img, watermark_img, position, size_ratio=0.2):
    """Add image watermark using simulated deep learning approach"""
    try:
        # Create a copy of the image for safe processing
        watermarked_img = img.copy()
        height, width, channels = img.shape
        
        # Resize watermark (smaller = more robust)
        wm_height, wm_width = watermark_img.shape[:2]
        new_wm_width = max(int(width * size_ratio / 2), 1)
        new_wm_height = max(int(wm_height * new_wm_width / wm_width), 1)
        
        # Safely resize watermark
        watermark_resized = cv2.resize(watermark_img, (new_wm_width, new_wm_height))
        
        # Convert to grayscale for pattern generation
        if len(watermark_resized.shape) > 2:
            if watermark_resized.shape[2] == 4:  # Has alpha channel
                # Use alpha as weighting
                alpha = watermark_resized[:, :, 3].astype(np.float32) / 255.0
                wm_gray = cv2.cvtColor(watermark_resized[:, :, 0:3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                # Apply alpha to grayscale
                wm_gray = wm_gray * alpha.reshape(alpha.shape[0], alpha.shape[1])
            else:
                wm_gray = cv2.cvtColor(watermark_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:  # Already grayscale
            wm_gray = watermark_resized.astype(np.float32) / 255.0
        
        # Generate a hash of the watermark for reproducibility
        hash_value = sum([ord(c) for c in str(wm_gray.shape)])
        np.random.seed(hash_value % 10000)
        
        # Create a pattern that can be applied to the entire image
        pattern = np.zeros((height, width), dtype=np.float32)
        
        # Calculate grid size based on watermark size
        grid_size = max(height // new_wm_height, width // new_wm_width)
        grid_size = max(8, min(grid_size, 32))  # Constrain between 8 and 32
        
        # Create sinusoidal pattern for each pixel in watermark
        for y in range(new_wm_height):
            for x in range(new_wm_width):
                if y < wm_gray.shape[0] and x < wm_gray.shape[1]:  # Safety check
                    # Only create pattern for pixels that are not fully transparent
                    pixel_value = wm_gray[y, x]
                    
                    if pixel_value > 0.1:  # Ignore near-transparent pixels
                        # Calculate region to apply this pixel's pattern
                        y_start = max(0, int(y * height / new_wm_height) - grid_size//2)
                        y_end = min(height, int((y+1) * height / new_wm_height) + grid_size//2)
                        x_start = max(0, int(x * width / new_wm_width) - grid_size//2)
                        x_end = min(width, int((x+1) * width / new_wm_width) + grid_size//2)
                        
                        # Create different frequencies based on position
                        freq_x = 0.5 + (x % 5) * 0.1
                        freq_y = 0.5 + (y % 5) * 0.1
                        
                        # Create sub-grid coordinates
                        y_coords = np.arange(y_start, y_end)
                        x_coords = np.arange(x_start, x_end)
                        
                        if len(y_coords) > 0 and len(x_coords) > 0:
                            # Create mesh grid for this region
                            xv, yv = np.meshgrid(x_coords, y_coords)
                            
                            # Generate sinusoidal pattern for this pixel
                            sub_pattern = np.sin(freq_x * xv / 10) * np.sin(freq_y * yv / 10)
                            
                            # Scale by pixel intensity
                            sub_pattern *= pixel_value * 2.0
                            
                            # Add to overall pattern
                            pattern[y_start:y_end, x_start:x_end] += sub_pattern
        
        # Normalize pattern to small values to keep it subtle
        pattern = pattern * 1.5
        pattern = np.clip(pattern, -1.0, 1.0)
        
        # Apply pattern to all channels
        for c in range(3):
            channel = watermarked_img[:, :, c].astype(np.float32)
            channel += pattern
            watermarked_img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return watermarked_img, 0, 0, new_wm_width, new_wm_height
    except Exception as e:
        print(f"Error in deep learning watermarking: {e}")
        import traceback
        traceback.print_exc()
        # If anything fails, return the original image
        return img.copy(), 0, 0, img.shape[1], img.shape[0]
