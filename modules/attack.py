import cv2
import numpy as np
import os
import random  # Add this import for random rotation angles

def attack_crop(img, crop_percentage=20):
    """Simulate a cropping attack on the watermarked image
    
    Args:
        img: OpenCV image to attack
        crop_percentage: Percentage of image to crop from each side
        
    Returns:
        attacked_img: Cropped image
    """
    # Ensure we have a valid image
    if img is None or img.size == 0:
        raise ValueError("Invalid input image for crop attack")
        
    height, width = img.shape[:2]
    
    # Calculate crop amount in pixels
    crop_x = int(width * crop_percentage / 100)
    crop_y = int(height * crop_percentage / 100)
    
    # Ensure we don't crop too much (leave at least 50% of the image)
    crop_x = min(crop_x, width // 4)
    crop_y = min(crop_y, height // 4)
    
    # Create cropped image
    if crop_x <= 0 or crop_y <= 0 or crop_x >= width//2 or crop_y >= height//2:
        # Invalid crop parameters, return original
        return img.copy()
        
    # Apply the crop
    cropped = img[crop_y:height-crop_y, crop_x:width-crop_x].copy()
    
    # Verify we have a valid result
    if cropped.size == 0:
        return img.copy()  # Return original if cropped image is empty
        
    return cropped

def attack_noise(img, noise_level=5):
    """Add random noise to the image to try to destroy watermark
    
    Args:
        img: OpenCV image to attack
        noise_level: Intensity of noise (0-20)
        
    Returns:
        attacked_img: Noisy image
    """
    # Create a copy to avoid modifying the original
    noisy_img = img.copy()
    
    # Convert to float for noise addition
    noisy_img = noisy_img.astype(np.float32)
    
    # Scale noise level to make it more effective
    actual_noise = noise_level * 2.5
    
    # Generate random noise
    noise = np.random.normal(0, actual_noise, img.shape)
    
    # Add noise to image
    noisy_img += noise
    
    # Clip values to valid range
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img

def attack_blur(img, blur_level=5):
    """Apply Gaussian blur to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        blur_level: Blur kernel size (odd number)
        
    Returns:
        attacked_img: Blurred image
    """
    # Ensure blur level is odd and at least 3
    blur_level = max(3, blur_level)
    if blur_level % 2 == 0:
        blur_level += 1
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (blur_level, blur_level), 0)
    
    return blurred

def attack_compression(img, quality=50):
    """Simulate JPEG compression to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        quality: JPEG quality (0-100, lower = more compression)
        
    Returns:
        attacked_img: Compressed image
    """
    # Create a temporary filename for compression
    temp_file = os.path.join(os.getcwd(), 'temp_compressed.jpg')
    
    # Save image with specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
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

def attack_rotation(img, angle=None):
    """Rotate the image to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        angle: Rotation angle in degrees, if None, randomly selects between 45, 90, 180, 270 degrees
        
    Returns:
        attacked_img: Rotated image
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid input image for rotation attack")
    
    # If angle is None, randomly select from common rotation angles
    if angle is None:
        rotation_angles = [45, 90, 180, 270]
        angle = random.choice(rotation_angles)
        print(f"Randomly selected rotation angle: {angle} degrees")
        
    height, width = img.shape[:2]
    
    # Calculate the diagonal length to ensure the whole image fits after rotation
    diagonal = int(np.sqrt(width**2 + height**2)) + 10  # Add a small buffer
    
    # Create a square canvas with white background to hold the rotated image
    canvas = np.ones((diagonal, diagonal, 3), dtype=np.uint8) * 255
    
    # Position the original image in the center of the canvas
    x_offset = (diagonal - width) // 2
    y_offset = (diagonal - height) // 2
    
    # Place the original image on the canvas
    if y_offset >= 0 and x_offset >= 0 and y_offset+height <= diagonal and x_offset+width <= diagonal:
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = img
    else:
        # Fallback if offsets are invalid
        canvas_center = diagonal // 2
        img_center_y, img_center_x = height // 2, width // 2
        y1 = max(0, canvas_center - img_center_y)
        y2 = min(diagonal, canvas_center + height - img_center_y)
        x1 = max(0, canvas_center - img_center_x)
        x2 = min(diagonal, canvas_center + width - img_center_x)
        
        img_y1 = max(0, img_center_y - canvas_center)
        img_y2 = min(height, img_center_y + diagonal - canvas_center)
        img_x1 = max(0, img_center_x - canvas_center)
        img_x2 = min(width, img_center_x + diagonal - canvas_center)
        
        try:
            canvas[y1:y2, x1:x2] = img[img_y1:img_y2, img_x1:img_x2]
        except:
            # Last resort - scale down the image to fit
            scale = min(diagonal / width, diagonal / height) * 0.9
            resized = cv2.resize(img, None, fx=scale, fy=scale)
            h, w = resized.shape[:2]
            y_offset = (diagonal - h) // 2
            x_offset = (diagonal - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = resized
    
    # Define the center of rotation (center of the canvas)
    center = (diagonal // 2, diagonal // 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply the rotation
    rotated_canvas = cv2.warpAffine(canvas, rotation_matrix, (diagonal, diagonal),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
    
    # Crop back to the original size from the center
    crop_x_offset = (diagonal - width) // 2
    crop_y_offset = (diagonal - height) // 2
    
    # Make sure crop offsets are valid
    if crop_y_offset >= 0 and crop_x_offset >= 0 and crop_y_offset+height <= diagonal and crop_x_offset+width <= diagonal:
        rotated = rotated_canvas[crop_y_offset:crop_y_offset+height, crop_x_offset:crop_x_offset+width]
    else:
        # Resize the result if cropping is impossible
        rotated = cv2.resize(rotated_canvas, (width, height))
    
    # Verify we got a valid result
    if rotated.shape[:2] != (height, width):
        rotated = cv2.resize(rotated, (width, height))
    
    return rotated

def attack_brightness(img, factor=1.5):
    """Change brightness to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        factor: Brightness adjustment factor (>1 = brighter, <1 = darker)
        
    Returns:
        attacked_img: Brightness-adjusted image
    """
    # Ensure factor is within reasonable range
    factor = max(0.1, min(3.0, factor))
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Scale the V channel (brightness)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
    
    # Convert back to BGR
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted

def attack_contrast(img, factor=1.5):
    """Change contrast to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        factor: Contrast adjustment factor
        
    Returns:
        attacked_img: Contrast-adjusted image
    """
    # Ensure factor is within reasonable range
    factor = max(0.1, min(3.0, factor))
    
    # Convert to float for contrast adjustment
    adjusted = img.astype(np.float32)
    
    # Apply contrast adjustment (centered at 128)
    adjusted = 128 + factor * (adjusted - 128)
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted

def attack_histogram_equalization(img):
    """Apply histogram equalization to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        
    Returns:
        attacked_img: Equalized image
    """
    # Convert to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Apply histogram equalization to Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    
    # Convert back to BGR
    equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return equalized

def attack_median_filter(img, kernel_size=5):
    """Apply median filter to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        kernel_size: Size of the median filter kernel (odd number)
        
    Returns:
        attacked_img: Filtered image
    """
    # Ensure kernel size is odd and at least 3
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply median filter
    filtered = cv2.medianBlur(img, kernel_size)
    
    return filtered

def attack_resize(img, scale_factor=0.5):
    """Resize down and up to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        scale_factor: Factor to resize by (0-1)
        
    Returns:
        attacked_img: Resized image
    """
    # Ensure scale factor is reasonable
    scale_factor = max(0.1, min(0.9, scale_factor))
    
    height, width = img.shape[:2]
    
    # Resize down - use AREA interpolation for downsampling
    small = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, 
                      interpolation=cv2.INTER_AREA)
    
    # Resize back up - use CUBIC interpolation for upsampling
    resized = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    
    return resized

def attack_flip(img, direction='horizontal'):
    """Flip the image horizontally or vertically
    
    Args:
        img: OpenCV image to attack
        direction: 'horizontal', 'vertical', or 'both'
        
    Returns:
        attacked_img: Flipped image
    """
    if direction == 'horizontal':
        return cv2.flip(img, 1)  # 1 = horizontal flip
    elif direction == 'vertical':
        return cv2.flip(img, 0)  # 0 = vertical flip
    else:  # both
        return cv2.flip(img, -1)  # -1 = both horizontal and vertical

def attack_shear(img, shear_factor=0.2):
    """Apply shear transform to the image
    
    Args:
        img: OpenCV image to attack
        shear_factor: Shear intensity (0.0-1.0)
        
    Returns:
        attacked_img: Sheared image
    """
    height, width = img.shape[:2]
    
    # Create shear matrix
    shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    
    # Calculate new width after shear
    new_width = width + int(height * abs(shear_factor))
    
    # Apply shear transformation
    sheared = cv2.warpAffine(img, shear_matrix, (new_width, height), 
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    # Crop or pad to maintain original size
    if new_width > width:
        # Crop to original width from center
        start_x = (new_width - width) // 2
        sheared = sheared[:, start_x:start_x+width]
    elif new_width < width:
        # Pad to original width
        pad_width = width - new_width
        sheared = cv2.copyMakeBorder(sheared, 0, 0, 0, pad_width, 
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    return sheared

def attack_color_shift(img, channel='red', shift=30):
    """Shift a color channel to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        channel: 'red', 'green', or 'blue'
        shift: Amount to shift the channel (-255 to 255)
        
    Returns:
        attacked_img: Color-shifted image
    """
    # Create a copy to avoid modifying the original
    shifted = img.copy().astype(np.float32)
    
    # Apply shift to the specified channel
    if channel == 'red':
        shifted[:, :, 2] += shift
    elif channel == 'green':
        shifted[:, :, 1] += shift
    elif channel == 'blue':
        shifted[:, :, 0] += shift
    else:
        # Shift all channels slightly
        shifted[:, :, 0] += shift * 0.7
        shifted[:, :, 1] += shift * 0.8
        shifted[:, :, 2] += shift * 0.9
    
    # Clip values to valid range
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    
    return shifted

def attack_combine(img, attack_level='medium'):
    """Combine multiple attacks to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        attack_level: 'low', 'medium', or 'high'
        
    Returns:
        attacked_img: Image with combined attacks
    """
    if attack_level == 'low':
        # Mild combined attack
        attacked = attack_noise(img, 3)
        attacked = attack_blur(attacked, 3)
        attacked = attack_compression(attacked, 75)
    
    elif attack_level == 'high':
        # Strong combined attack
        attacked = attack_crop(img, 10)
        attacked = attack_noise(attacked, 12)
        attacked = attack_blur(attacked, 7)
        attacked = attack_compression(attacked, 30)
        attacked = attack_contrast(attacked, 1.3)
        attacked = attack_brightness(attacked, 1.2)
        # Add color shift for high-intensity attack
        attacked = attack_color_shift(attacked, 'all', 15)
    
    else:  # 'medium' (default)
        # Moderate combined attack
        attacked = attack_noise(img, 7)
        attacked = attack_blur(attacked, 5)
        attacked = attack_compression(attacked, 50)
        attacked = attack_contrast(attacked, 1.2)
    
    return attacked

def compare_images_after_attack(original, attacked, attacked_type):
    """Analyze visual differences between original and attacked images
    
    Args:
        original: Original watermarked image
        attacked: Attacked version of the image
        attacked_type: Type of attack that was applied
        
    Returns:
        result: Dictionary with comparison results
    """
    # Calculate metrics to measure attack impact
    
    # Convert to grayscale for some metrics
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    attacked_gray = cv2.cvtColor(attacked, cv2.COLOR_BGR2GRAY)
    
    # 1. Mean Squared Error
    mse = np.mean((original_gray.astype(float) - attacked_gray.astype(float)) ** 2)
    
    # 2. Custom implementation of Structural Similarity Index (SSIM) to avoid NumPy 2.x / scipy issues
    # Use a simplified version to avoid dependencies
    try:
        # Convert to float
        img1 = original_gray.astype(np.float32)
        img2 = attacked_gray.astype(np.float32)
        
        # Constants to stabilize division
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Calculate means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        # Calculate variances and covariance
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        # Use Gaussian filter for weighted calculations
        sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        # SSIM formula
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = num / (den + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Calculate mean SSIM
        ssim_value = np.mean(ssim_map)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        ssim_value = 0.5  # Default value if calculation fails
    
    # 3. Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = 100  # Perfect match
    else:
        psnr = 10 * np.log10(255 * 255 / max(mse, 1e-10))  # Avoid log(0)
    
    # 4. Create a difference visualization
    diff = cv2.absdiff(original, attacked)
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=5)  # Enhance for better visibility
    
    # Create a heatmap to visualize differences
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
    
    # Return attack results
    return {
        'attack_type': attacked_type,
        'mse': float(mse),
        'ssim': float(ssim_value),
        'psnr': float(psnr),
        'difference': diff_enhanced,
        'heatmap': heatmap,
        'survival_score': float(min(100, psnr))  # Higher PSNR = better watermark survival
    }
