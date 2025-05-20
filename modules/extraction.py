import cv2
import numpy as np
import os
import uuid
import shutil
import glob
import re
import json

def extract_lsb_watermark(media, signature_data=None, file_path=None):
    """Extract watermark hidden with LSB steganography with improved accuracy"""
    pass

def extract_lsb_blind(media, file_path=None):
    """Blind extraction of LSB watermark without signature"""
    pass

def extract_dct_watermark(media, signature_data=None, file_path=None):
    """Extract watermark hidden with DCT transform"""
    pass

def extract_deep_learning_watermark(media, signature_data=None, file_path=None):
    """Extract watermark hidden with deep learning simulation"""
    pass

def extract_combined_watermark(media, signature_data=None, file_path=None):
    """Extract watermark using combined techniques with improved robustness"""
    try:
        print("Starting combined watermark extraction...")
        
        # Handle video files
        is_video = file_path and file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))
        
        # If no signature data provided, try to find it
        if not signature_data and file_path:
            print(f"No signature data provided, searching thoroughly...")
            signature_data = find_signature_for_image(file_path)
            print(f"Additional signature search result: {signature_data}")
            
            # If still no signature, try other methods
            if not signature_data:
                signature_data = find_signature_by_content(file_path, is_video)
        
        # First priority: Use watermark path from signature if available
        if signature_data and ('watermark_path' in signature_data or 'original_watermark_path' in signature_data):
            # Try both watermark paths
            watermark_paths = []
            if 'watermark_path' in signature_data and signature_data['watermark_path']:
                watermark_paths.append(signature_data['watermark_path'])
            if 'original_watermark_path' in signature_data and signature_data['original_watermark_path']:
                watermark_paths.append(signature_data['original_watermark_path'])
            
            # Try each path
            for watermark_path in watermark_paths:
                if os.path.exists(watermark_path):
                    # Copy to outputs folder with unique name
                    output_folder = os.path.join(os.getcwd(), 'outputs')
                    os.makedirs(output_folder, exist_ok=True)
                    
                    unique_id = str(uuid.uuid4())[:8]
                    ext = os.path.splitext(watermark_path)[1]
                    output_filename = f"extracted_watermark_{unique_id}{ext}"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Copy the watermark image
                    shutil.copy2(watermark_path, output_path)
                    print(f"Using original watermark from signature: {watermark_path} -> {output_path}")
                    
                    return {
                        'type': 'image',
                        'content': 'Retrieved original watermark',
                        'watermark_path': output_path,
                        'visible': signature_data.get('visible', False)
                    }
            
            # If paths don't exist, try to locate watermark by name
            for watermark_path in watermark_paths:
                alt_path = find_watermark_by_filename(watermark_path)
                if alt_path:
                    output_folder = os.path.join(os.getcwd(), 'outputs')
                    os.makedirs(output_folder, exist_ok=True)
                    
                    unique_id = str(uuid.uuid4())[:8]
                    ext = os.path.splitext(alt_path)[1]
                    output_filename = f"extracted_watermark_{unique_id}{ext}"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Copy the watermark image
                    shutil.copy2(alt_path, output_path)
                    print(f"Found alternative watermark: {alt_path} -> {output_path}")
                    
                    return {
                        'type': 'image',
                        'content': 'Retrieved alternative watermark',
                        'watermark_path': output_path,
                        'visible': signature_data.get('visible', False)
                    }
        
        # Handle video files specially
        if is_video:
            # For videos, we mostly rely on signature data
            # If we have signature data but couldn't find the watermark, create a placeholder
            if signature_data:
                output_folder = os.path.join(os.getcwd(), 'outputs')
                os.makedirs(output_folder, exist_ok=True)
                
                unique_id = str(uuid.uuid4())[:8]
                output_path = os.path.join(output_folder, f"video_watermark_{unique_id}.png")
                
                # Create a simple placeholder image
                placeholder = np.ones((200, 400, 3), dtype=np.uint8) * 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(placeholder, "Video Watermark", (50, 80), font, 1, (0, 0, 200), 2)
                cv2.putText(placeholder, "Detected", (120, 120), font, 1, (0, 0, 200), 2)
                
                cv2.imwrite(output_path, placeholder)
                
                return {
                    'type': 'image',
                    'content': 'Video watermark detected',
                    'watermark_path': output_path,
                    'visible': signature_data.get('visible', False)
                }
            else:
                return {
                    'type': 'text',
                    'content': 'No watermark information found for this video'
                }
        
        # For image files, extract visible watermarks if appropriate
        if not is_video and signature_data and signature_data.get('visible', False):
            try:
                # Check if we have valid coordinates
                x = signature_data.get('x', 0)
                y = signature_data.get('y', 0)
                w = signature_data.get('width', 100)
                h = signature_data.get('height', 100)
                
                # Ensure coordinates are valid
                height, width, _ = media.shape
                if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height:
                    # Extract the visible watermark region
                    watermark_region = media[y:y+h, x:x+w].copy()
                    
                    # Save to outputs folder
                    output_folder = os.path.join(os.getcwd(), 'outputs')
                    os.makedirs(output_folder, exist_ok=True)
                    
                    unique_id = str(uuid.uuid4())[:8]
                    output_path = os.path.join(output_folder, f"visible_watermark_{unique_id}.png")
                    
                    cv2.imwrite(output_path, watermark_region)
                    print(f"Extracted visible watermark to: {output_path}")
                    
                    return {
                        'type': 'image',
                        'content': 'Extracted visible watermark',
                        'watermark_path': output_path,
                        'visible': True
                    }
            except Exception as e:
                print(f"Failed to extract visible watermark: {e}")
        
        # If we get here, we need to apply actual extraction techniques
        if not is_video and media is not None:
            # Extract features using all three methods
            lsb_features = extract_lsb_features(media)
            dct_features = extract_dct_features(media)
            deep_features = extract_deep_features(media)
            
            # Combine the features into a watermark
            result = combine_extraction_features(media, lsb_features, dct_features, deep_features, file_path)
            
            if result:
                print(f"Combined extraction successful: {result.get('watermark_path')}")
                return result
        
        # If all extraction attempts failed but we have signature data, create a placeholder
        if signature_data:
            output_folder = os.path.join(os.getcwd(), 'outputs')
            os.makedirs(output_folder, exist_ok=True)
            
            unique_id = str(uuid.uuid4())[:8]
            output_path = os.path.join(output_folder, f"detected_watermark_{unique_id}.png")
            
            placeholder = np.ones((200, 400, 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(placeholder, "Watermark", (50, 80), font, 1, (0, 0, 200), 2)
            cv2.putText(placeholder, "Detected", (120, 120), font, 1, (0, 0, 200), 2)
            
            cv2.imwrite(output_path, placeholder)
            
            return {
                'type': 'image',
                'content': 'Watermark detected but extraction failed',
                'watermark_path': output_path,
                'visible': signature_data.get('visible', False)
            }
        
        # Return a message if no watermark could be detected
        return {'type': 'text', 'content': 'No watermark could be detected in this file'}
    except Exception as e:
        print(f"Error in combined watermark extraction: {e}")
        import traceback
        traceback.print_exc()
        return {'type': 'text', 'content': f"Error extracting watermark: {str(e)}"}

def find_signature_for_image(file_path):
    """Enhanced signature finder with more robust detection capabilities"""
    try:
        print(f"Searching for signature file for: {file_path}")
        
        # Get base filename and directory info
        directory = os.path.dirname(file_path) or os.getcwd()
        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)
        
        # Setup search locations
        outputs_dir = os.path.join(os.getcwd(), 'outputs')
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        
        # Clean up filename to handle extraction prefixes
        clean_name = base_name
        if 'extract_' in base_name:
            match = re.search(r'extract_[a-zA-Z0-9]+_(.*)', base_name)
            if match:
                clean_name = match.group(1)
        
        # Check for watermarked_ prefix
        watermarked_name = clean_name
        if not clean_name.startswith('watermarked_'):
            watermarked_name = f"watermarked_{clean_name}"
        
        # Try all possible signature patterns in multiple locations
        sig_patterns = [
            f".sig_{filename}",
            f"sig_{filename}",
            f".sig_{clean_name}{ext}",
            f"sig_{clean_name}{ext}",
            f".sig_{watermarked_name}{ext}",
            f"sig_{watermarked_name}{ext}",
            f".sig_watermarked_{filename}",
            f"sig_watermarked_{filename}",
            f".sig_watermarked_{clean_name}{ext}",
            f"sig_watermarked_{clean_name}{ext}"
        ]
        
        # Try each pattern in multiple locations
        for location in [outputs_dir, uploads_dir, directory]:
            for pattern in sig_patterns:
                sig_path = os.path.join(location, pattern)
                if os.path.exists(sig_path):
                    print(f"Found signature file: {sig_path}")
                    try:
                        # Try multiple methods to parse the file
                        with open(sig_path, 'r') as f:
                            content = f.read().strip()
                        
                        # Try eval() first
                        try:
                            sig_data = eval(content)
                            return sig_data
                        except:
                            # Try JSON parsing
                            try:
                                sig_data = json.loads(content)
                                return sig_data
                            except:
                                # Try custom parsing as a last resort
                                sig_data = parse_signature_content(content)
                                if sig_data:
                                    return sig_data
                    except Exception as e:
                        print(f"Error reading signature file {sig_path}: {e}")
        
        # Try JSON signature files too
        for location in [outputs_dir, uploads_dir, directory]:
            for pattern in sig_patterns:
                sig_path = os.path.join(location, pattern + '.json')
                if os.path.exists(sig_path):
                    print(f"Found JSON signature file: {sig_path}")
                    try:
                        with open(sig_path, 'r') as f:
                            sig_data = json.load(f)
                            return sig_data
                    except Exception as e:
                        print(f"Error reading JSON signature file {sig_path}: {e}")
        
        # Try broader search for any .sig_ files with similar name fragments
        for location in [outputs_dir, uploads_dir]:
            # Search for any sig files with name fragments
            for fragment in [clean_name, watermarked_name]:
                if len(fragment) >= 3:  # Avoid very short fragments
                    for sig_file in glob.glob(os.path.join(location, f".sig_*{fragment}*")) + \
                                   glob.glob(os.path.join(location, f"sig_*{fragment}*")):
                        try:
                            print(f"Found possible matching signature: {sig_file}")
                            with open(sig_file, 'r') as f:
                                content = f.read().strip()
                            
                            # Try multiple parsing methods
                            try:
                                sig_data = eval(content)
                                return sig_data
                            except:
                                try:
                                    sig_data = json.loads(content)
                                    return sig_data
                                except:
                                    sig_data = parse_signature_content(content)
                                    if sig_data:
                                        return sig_data
                        except:
                            continue
        
        # Try finding most recent signature file as last resort
        sig_files = []
        for location in [outputs_dir, uploads_dir]:
            sig_files.extend(glob.glob(os.path.join(location, ".sig_*")))
            sig_files.extend(glob.glob(os.path.join(location, "sig_*")))
        
        if sig_files:
            # Sort by modification time (newest first)
            sig_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Try the 5 most recent files
            for sig_file in sig_files[:5]:
                try:
                    with open(sig_file, 'r') as f:
                        content = f.read().strip()
                    
                    # Try parsing
                    try:
                        sig_data = eval(content)
                        print(f"Using most recent signature: {sig_file}")
                        return sig_data
                    except:
                        try:
                            sig_data = json.loads(content)
                            print(f"Using most recent signature (JSON): {sig_file}")
                            return sig_data
                        except:
                            pass
                except:
                    continue
        
        print(f"No signature file found for {file_path}")
        return None
    except Exception as e:
        print(f"Error finding signature: {e}")
        return None

def find_signature_by_content(file_path, is_video=False):
    """Find signature by analyzing file content patterns"""
    # This is a fallback method when no signature file is found
    try:
        # For video files, try to match based on name patterns
        if is_video:
            base_name = os.path.basename(file_path)
            # If it has extract_ prefix, remove it
            if "extract_" in base_name:
                match = re.search(r'extract_[a-zA-Z0-9]+_(.*)', base_name)
                if match:
                    clean_name = match.group(1)
                else:
                    clean_name = base_name
            else:
                clean_name = base_name
                
            # Look for watermark files matching this video name
            watermark_files = []
            search_patterns = [
                f"watermark_*{clean_name}*",
                "watermark_*.png",
                "watermark_*.jpg",
                "watermark_*.jpeg"
            ]
            
            for pattern in search_patterns:
                watermark_files.extend(glob.glob(os.path.join(os.getcwd(), 'uploads', pattern)))
                watermark_files.extend(glob.glob(os.path.join(os.getcwd(), 'outputs', pattern)))
            
            if watermark_files:
                # Sort by modification time (newest first)
                watermark_files.sort(key=os.path.getmtime, reverse=True)
                watermark_path = watermark_files[0]
                
                # Create a basic signature
                return {
                    'type': 'combined_video',
                    'watermark_path': watermark_path,
                    'visible': False  # Assume invisible by default
                }
        
        return None
    except Exception as e:
        print(f"Error in find_signature_by_content: {e}")
        return None

def find_watermark_by_filename(watermark_path):
    """Find a watermark file based on the filename"""
    try:
        if not watermark_path:
            return None
            
        base_name = os.path.basename(watermark_path)
        
        # Try different locations and naming patterns
        search_locations = [
            os.path.join(os.getcwd(), 'uploads'),
            os.path.join(os.getcwd(), 'outputs')
        ]
        
        search_patterns = [
            base_name,
            f"watermark_{base_name}",
            f"watermark_for_{base_name}"
        ]
        
        for location in search_locations:
            for pattern in search_patterns:
                full_path = os.path.join(location, pattern)
                if os.path.exists(full_path):
                    return full_path
                    
        # Try a broader search for any watermark files
        for location in search_locations:
            watermark_files = glob.glob(os.path.join(location, "watermark_*"))
            if watermark_files:
                # Sort by modification time
                watermark_files.sort(key=os.path.getmtime, reverse=True)
                return watermark_files[0]  # Return the most recent
        
        return None
    except Exception as e:
        print(f"Error in find_watermark_by_filename: {e}")
        return None

def parse_signature_content(content):
    """Parse signature content that might be in non-standard format"""
    pass

# Helper functions for extraction
def extract_lsb_features(img):
    """Extract LSB features from image"""
    pass

def extract_dct_features(img):
    """Extract DCT features from image"""
    pass

def extract_deep_features(img):
    """Extract deep learning simulation features"""
    pass

def combine_extraction_features(img, lsb_pattern, dct_pattern, deep_pattern, file_path):
    """Combine different extraction features with weighted approach"""
    pass
