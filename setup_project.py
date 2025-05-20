import os
import shutil

def setup_project_structure():
    """Setup struktur folder untuk proyek Watermarker"""
    base_dir = os.getcwd()
    
    # Buat struktur direktori
    directories = [
        os.path.join(base_dir, 'static', 'css'),
        os.path.join(base_dir, 'static', 'js'),
        os.path.join(base_dir, 'templates'),
        os.path.join(base_dir, 'uploads'),
        os.path.join(base_dir, 'outputs'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'utils'),
        os.path.join(base_dir, 'routes')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Memastikan direktori ada: {directory}")
    
    # Pindahkan file CSS yang ada
    css_src = os.path.join(base_dir, 'static', 'style.css')
    css_dst = os.path.join(base_dir, 'static', 'css', 'style.css')
    if os.path.exists(css_src) and not os.path.exists(css_dst):
        shutil.copy2(css_src, css_dst)
        print(f"Memindahkan CSS ke: {css_dst}")
    
    # Pindahkan file JS yang ada
    js_src = os.path.join(base_dir, 'static', 'script.js')
    js_dst = os.path.join(base_dir, 'static', 'js', 'script.js')
    if os.path.exists(js_src) and not os.path.exists(js_dst):
        shutil.copy2(js_src, js_dst)
        print(f"Memindahkan JavaScript ke: {js_dst}")
    
    print("Setup selesai! Struktur proyek siap.")

if __name__ == "__main__":
    setup_project_structure()
