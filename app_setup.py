import os
import shutil

def setup_folders():
    """Setup the folder structure for Watermarker"""
    base_dir = os.getcwd()
    
    # Create main directories
    dirs_to_create = [
        os.path.join(base_dir, 'static'),
        os.path.join(base_dir, 'static', 'css'),
        os.path.join(base_dir, 'static', 'js'),
        os.path.join(base_dir, 'templates'),
        os.path.join(base_dir, 'uploads'),
        os.path.join(base_dir, 'outputs'),
        os.path.join(base_dir, 'models')
    ]
    
    # Create directories if they don't exist
    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Move CSS and JS files to their appropriate directories
    styles_path = os.path.join(base_dir, 'static', 'style.css')
    css_dest = os.path.join(base_dir, 'static', 'css', 'style.css')
    if os.path.exists(styles_path) and not os.path.exists(css_dest):
        shutil.copy2(styles_path, css_dest)
        print(f"Copied styles to: {css_dest}")
    
    script_path = os.path.join(base_dir, 'static', 'script.js')
    js_dest = os.path.join(base_dir, 'static', 'js', 'script.js')
    if os.path.exists(script_path) and not os.path.exists(js_dest):
        shutil.copy2(script_path, js_dest)
        print(f"Copied scripts to: {js_dest}")
    
    print("Setup complete! Folder structure is ready.")

if __name__ == "__main__":
    setup_folders()
