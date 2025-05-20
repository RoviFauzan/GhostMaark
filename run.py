"""
Helper script to run Watermarker with appropriate dependency checks
"""
import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ["flask", "numpy", "opencv-python-headless", "PIL"]
    missing = []
    
    for package in required:
        try:
            if package == "PIL":
                __import__("PIL")
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies(packages):
    """Install missing dependencies"""
    print(f"Installing missing dependencies: {', '.join(packages)}")
    # Properly map package names for pip
    pip_packages = []
    for package in packages:
        if package == "PIL":
            pip_packages.append("Pillow")
        elif package == "opencv-python-headless":
            pip_packages.append("opencv-python-headless")
        else:
            pip_packages.append(package)
    
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pip_packages)
    print("Installation complete!")

if __name__ == "__main__":
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Required dependencies are missing: {', '.join(missing)}")
        install = input("Would you like to install them now? (y/n): ")
        if install.lower() == 'y':
            install_dependencies(missing)
        else:
            print("Cannot continue without required dependencies.")
            sys.exit(1)
            
    print("\nStarting Watermarker application...")
    os.system(f"{sys.executable} Watermarker.py")
