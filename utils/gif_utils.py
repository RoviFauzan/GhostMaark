import os
import cv2
import numpy as np
from PIL import Image, ImageSequence
import tempfile
from modules.image_watermark import add_visible_image_watermark

def is_gif(file_path):
    """Check if a file is a GIF
    
    Args:
        file_path: Path to the file
        
    Returns:
        is_gif: True if the file is a GIF
    """
    try:
        # Check file extension
        if not file_path.lower().endswith('.gif'):
            return False
            
        # Verify it's actually a GIF by trying to open it
        img = Image.open(file_path)
        return img.format == 'GIF'
    except:
        return False

def process_gif_with_watermark(gif_path, watermark_image_path, position='bottom-right', 
                              size_ratio=0.2, watermark_type='visible', output_folder=None):
    """Add watermark to a GIF file
    
    Args:
        gif_path: Path to the GIF file
        watermark_image_path: Path to the watermark image
        position: Position of watermark
        size_ratio: Size of watermark relative to image (0.0-1.0)
        watermark_type: 'visible' or 'steganographic'
        output_folder: Folder to save output
        
    Returns:
        output_path: Path to the watermarked GIF
    """
    try:
        # Open GIF file
        gif = Image.open(gif_path)
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        
        # Get GIF properties
        duration = gif.info.get('duration', 100)  # Default to 100ms if not specified
        loop = gif.info.get('loop', 0)  # Default to loop forever
        
        # Process each frame
        frames = []
        frame_paths = []
        
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            # Save frame to temp file
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            frame.convert("RGBA").save(frame_path)
            frame_paths.append(frame_path)
            
            # Apply watermark to frame
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise Exception(f"Failed to read frame {i}")
            
            # Handle transparency in PNG frame
            if img.shape[2] == 3:  # No alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                img[:, :, 3] = 255  # Add opaque alpha channel
            
            # Read watermark image
            watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_UNCHANGED)
            if watermark_img is None:
                raise Exception(f"Failed to open watermark image: {watermark_image_path}")
            
            # Make sure watermark has alpha channel
            if watermark_img.shape[2] == 3:  # No alpha channel
                watermark_img = cv2.cvtColor(watermark_img, cv2.COLOR_BGR2BGRA)
                watermark_img[:, :, 3] = 255  # Add opaque alpha channel
            
            # Apply watermark
            if watermark_type == 'visible':
                watermarked_img, _, _, _, _ = add_visible_image_watermark(
                    img, watermark_img, position, size_ratio)
            else:
                # For steganographic watermarks in GIFs, just use a very subtle visible watermark
                # since GIFs have limited color palette for effective steganography
                transparent_watermark = watermark_img.copy()
                transparent_watermark[:, :, 3] = transparent_watermark[:, :, 3] // 5  # Make it very transparent
                watermarked_img, _, _, _, _ = add_visible_image_watermark(
                    img, transparent_watermark, position, size_ratio)
            
            # Save watermarked frame
            watermarked_frame_path = os.path.join(temp_dir, f"watermarked_frame_{i:03d}.png")
            cv2.imwrite(watermarked_frame_path, watermarked_img)
            
            # Load frame for GIF assembly
            watermarked_frame = Image.open(watermarked_frame_path)
            frames.append(watermarked_frame)
        
        # Create output path
        base_name = os.path.basename(gif_path)
        output_filename = f"watermarked_{base_name}"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=False
        )
        
        # Clean up temporary files
        for path in frame_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
                    
        # Also clean up watermarked frames
        for i in range(len(frames)):
            wm_path = os.path.join(temp_dir, f"watermarked_frame_{i:03d}.png")
            if os.path.exists(wm_path):
                try:
                    os.remove(wm_path)
                except:
                    pass
                
        # Try to remove temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass  # Ignore if directory not empty or can't be removed
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error processing GIF file: {str(e)}")
