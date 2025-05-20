import os
import shutil
from setuptools import setup, find_packages

def setup_folder_structure():
    """Create proper folder structure for the Watermarker application"""
    base_dir = os.getcwd()
    
    # Ensure directories exist
    dirs = [
        os.path.join(base_dir, 'static'),
        os.path.join(base_dir, 'static', 'css'),
        os.path.join(base_dir, 'static', 'js'),
        os.path.join(base_dir, 'templates'),
        os.path.join(base_dir, 'uploads'),
        os.path.join(base_dir, 'outputs'),
        os.path.join(base_dir, 'models')
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Move CSS file to correct location if needed
    css_src = os.path.join(base_dir, 'static', 'style.css')
    css_dst = os.path.join(base_dir, 'static', 'css', 'style.css')
    if os.path.exists(css_src) and not os.path.exists(css_dst):
        shutil.copy2(css_src, css_dst)
        print(f"Copied CSS file to: {css_dst}")
    
    # Move JS file to correct location if needed
    js_src = os.path.join(base_dir, 'static', 'script.js')
    js_dst = os.path.join(base_dir, 'static', 'js', 'script.js')
    if os.path.exists(js_src) and not os.path.exists(js_dst):
        shutil.copy2(js_src, js_dst)
        print(f"Copied JavaScript file to: {js_dst}")
    
    print("Setup complete! Run 'python Watermarker.py' to start the application.")

if __name__ == "__main__":
    setup_folder_structure()

setup(
    name="Watermarker",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask',
        'numpy',
        'opencv-python-headless',
        'pillow',
        'tensorflow'
    ],
    author="Watermarker Team",
    author_email="example@example.com",
    description="Aplikasi untuk menambahkan watermark ke gambar dan video",
    keywords="watermark, image, video, security, steganography",
    url="https://github.com/TrentPierce/watermarker",
)
