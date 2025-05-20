import cv2
import numpy as np
import random
import os

def create_lsb_pattern(img):
    """Create visualization of LSB modifications
    
    Args:
        img: OpenCV image
        
    Returns:
        pattern: Image highlighting LSB patterns
    """
    # Create a pattern image of the same size
    pattern = np.zeros_like(img)
    height, width, channels = img.shape
    
    # Extract LSB from each channel
    for c in range(channels):
        # Extract the least significant bit
        lsb = img[:, :, c] & 1
        
        # Amplify LSB for visualization (0 or 255)
        lsb_amplified = lsb * 255
        
        # Set the channel in the pattern image
        pattern[:, :, c] = lsb_amplified
    
    return pattern
def attack_crop(img, crop_percentage=20):
    """Simulate a cropping attack on the watermarked image
    
    Args:
        img: OpenCV image to attack
        crop_percentage: Percentage of image to crop from each side
        
    Returns:
        attacked_img: Cropped image
    """
    height, width = img.shape[:2]
    
    # Calculate crop amount in pixels
    crop_x = int(width * crop_percentage / 100)
    crop_y = int(height * crop_percentage / 100)
    
    # Ensure we don't crop too much
    crop_x = min(crop_x, width // 3)
    crop_y = min(crop_y, height // 3)
    
    # Create cropped image
    cropped = img[crop_y:height-crop_y, crop_x:width-crop_x]
    
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
    
    # Generate random noise
    noise = np.random.normal(0, noise_level, img.shape)
    
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
    # Ensure blur level is odd
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
    temp_file = 'temp_compressed.jpg'
    
    # Save image with specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    cv2.imwrite(temp_file, img, encode_param)
    
    # Read the compressed image back
    compressed = cv2.imread(temp_file)
    
    # Remove temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return compressed

def attack_rotation(img, angle=45):
    """Rotate the image to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        angle: Rotation angle in degrees
        
    Returns:
        attacked_img: Rotated image
    """
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height), 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(255, 255, 255))
    
    return rotated

def attack_brightness(img, factor=1.5):
    """Change brightness to try to remove watermark
    
    Args:
        img: OpenCV image to attack
        factor: Brightness adjustment factor (>1 = brighter, <1 = darker)
        
    Returns:
        attacked_img: Brightness-adjusted image
    """
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
    # Ensure kernel size is odd
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
    height, width = img.shape[:2]
    
    # Resize down
    small = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, 
                      interpolation=cv2.INTER_AREA)
    
    # Resize back up
    resized = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return resized

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
        attacked = attack_noise(attacked, 10)
        attacked = attack_blur(attacked, 7)
        attacked = attack_compression(attacked, 30)
        attacked = attack_contrast(attacked, 1.3)
        attacked = attack_brightness(attacked, 1.2)
    
    else:  # 'medium' (default)
        # Moderate combined attack
        attacked = attack_noise(img, 5)
        attacked = attack_blur(attacked, 5)
        attacked = attack_compression(attacked, 50)
        attacked = attack_contrast(attacked, 1.2)
    
    return attacked

def detect_watermark_after_attack(original, attacked, attacked_type):
    """Analyze how well the watermark survived the attack
    
    Args:
        original: Original watermarked image
        attacked: Attacked version of the image
        attacked_type: Type of attack that was applied
        
    Returns:
        result: Dictionary with attack results and similarity metrics
    """
    # Calculate metrics to measure attack impact
    
    # Convert to grayscale for some metrics
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    attacked_gray = cv2.cvtColor(attacked, cv2.COLOR_BGR2GRAY)
    
    # 1. Mean Squared Error
    mse = np.mean((original_gray - attacked_gray) ** 2)
    
    # 2. Structural Similarity Index (SSIM)
    try:
        ssim = cv2.compareSSIM(original_gray, attacked_gray)
    except:
        # Fallback if SSIM is not available
        ssim = 0
        
    # 3. Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = 100  # Perfect match
    else:
        psnr = 10 * np.log10(255 * 255 / mse)
    
    # 4. Create a difference visualization
    diff = cv2.absdiff(original, attacked)
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=5)  # Enhance for better visibility
    
    # For visible watermark attacks, we can check where the most changes happened
    heatmap = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
    
    # Return attack results
    return {
        'attack_type': attacked_type,
        'mse': mse,
        'ssim': ssim,
        'psnr': psnr,
        'difference': diff_enhanced,
        'heatmap': heatmap,
        'survival_score': min(100, psnr)  # Higher PSNR = better watermark survival
    }
