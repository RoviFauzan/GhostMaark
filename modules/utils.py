import cv2
import numpy as np
import hashlib
import os

def convert_text_to_binary(text):
    """Convert text to binary representation
    
    Args:
        text: String to convert
    
    Returns:
        binary_str: String of binary digits
    """
    return ''.join(format(ord(char), '08b') for char in text)

def convert_binary_to_text(binary_str):
    """Convert binary representation to text
    
    Args:
        binary_str: String of binary digits
    
    Returns:
        text: Converted text string
    """
    text = ""
    for i in range(0, len(binary_str), 8):
        if i + 8 <= len(binary_str):
            byte = binary_str[i:i+8]
            try:
                char = chr(int(byte, 2))
                # Only include printable ASCII
                if 32 <= ord(char) <= 126:
                    text += char
                else:
                    text += '?'
            except:
                text += '?'
    return text

def generate_seed_from_text(text):
    """Generate a reproducible seed from text
    
    Args:
        text: Input text
    
    Returns:
        seed: Integer seed value
    """
    hash_obj = hashlib.md5(text.encode())
    return int(hash_obj.hexdigest(), 16) % 10000

def is_valid_extracted_text(text, threshold=0.3):
    """Check if extracted text appears to be valid
    
    Args:
        text: Text to check
        threshold: Minimum ratio of valid characters (0.0-1.0)
    
    Returns:
        is_valid: True if the text appears valid
    """
    if not text or len(text) == 0:
        return False
    
    # Count valid characters (alphanumeric and space)
    valid_chars = sum(c.isalnum() or c.isspace() for c in text)
    ratio = valid_chars / len(text)
    
    return ratio >= threshold

def save_signature_data(signature_data, output_path):
    """Save signature data as a hidden file
    
    Args:
        signature_data: Dictionary of signature information
        output_path: Path to the output file
    
    Returns:
        signature_path: Path to the signature file
    """
    # Create the signature filename by adding .sig_ prefix
    signature_filename = f".sig_{os.path.basename(output_path)}"
    signature_path = os.path.join(os.path.dirname(output_path), signature_filename)
    
    # Save the signature data
    with open(signature_path, 'w') as f:
        f.write(str(signature_data))
    
    return signature_path

def load_signature_data(file_path):
    """Try to find and load signature data for a file
    
    Args:
        file_path: Path to the watermarked file
    
    Returns:
        signature_data: Dictionary of signature data or None if not found
    """
    # Create the potential signature filename
    signature_filename = f".sig_{os.path.basename(file_path)}"
    signature_path = os.path.join(os.path.dirname(file_path), signature_filename)
    
    # Check if the signature file exists
    if not os.path.exists(signature_path):
        return None
    
    # Load the signature data
    try:
        with open(signature_path, 'r') as f:
            signature_data = eval(f.read())
        return signature_data
    except:
        return None
