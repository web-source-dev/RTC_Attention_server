#!/usr/bin/env python3
"""
Installation script for RTC Attention Server
Handles optional dependencies gracefully
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("Installing RTC Attention Server dependencies...")
    
    # Required packages
    required_packages = [
        "flask==2.3.3",
        "flask-cors==4.0.0", 
        "Pillow==10.0.1",
        "gunicorn==21.2.0",
        "opencv-python==4.8.1.78",
        "mediapipe==0.10.8",
        "numpy==1.24.3"
    ]
    
    # Optional packages
    optional_packages = [
        "psutil==5.9.6"
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            return False
    
    print("\nInstalling optional packages...")
    for package in optional_packages:
        print(f"Installing {package} (optional)...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"⚠ {package} installation failed (optional - server will work without it)")
    
    print("\nInstallation completed!")
    print("You can now start the server with: python app.py")
    
    return True

if __name__ == "__main__":
    main() 