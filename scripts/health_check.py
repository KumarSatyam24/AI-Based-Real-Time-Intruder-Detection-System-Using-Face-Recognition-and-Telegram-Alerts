#!/usr/bin/env python3
"""
System Health Check Script
Validates that all components of the facial detection system are working properly.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def print_status(message, status):
    """Print status with color coding."""
    colors = {
        'PASS': '\033[92m✅ PASS\033[0m',
        'FAIL': '\033[91m❌ FAIL\033[0m', 
        'WARN': '\033[93m⚠️  WARN\033[0m',
        'INFO': '\033[94mℹ️  INFO\033[0m'
    }
    print(f"{colors.get(status, status)} {message}")

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version >= (3, 9):
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", 'PASS')
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)", 'FAIL')
        return False

def check_dependencies():
    """Check required Python packages."""
    required_packages = [
        'cv2',
        'numpy',
        'face_recognition',
        'telegram',
        'yaml',
        'loguru',
        'pytest'
    ]
    
    all_pass = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_status(f"Package {package}", 'PASS')
        except ImportError:
            print_status(f"Package {package} not found", 'FAIL')
            all_pass = False
    
    return all_pass

def check_directory_structure():
    """Check required directories exist."""
    required_dirs = [
        'src',
        'config',
        'data/authorized_faces',
        'data/captured_intruders',
        'logs',
        'models',
        'tests'
    ]
    
    all_pass = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print_status(f"Directory {directory}", 'PASS')
        else:
            print_status(f"Directory {directory} missing", 'FAIL')
            all_pass = False
    
    return all_pass

def check_configuration_files():
    """Check configuration files."""
    config_files = [
        'config/settings.yaml',
        'requirements.txt',
        'README.md'
    ]
    
    all_pass = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print_status(f"Config file {config_file}", 'PASS')
        else:
            print_status(f"Config file {config_file} missing", 'FAIL')
            all_pass = False
    
    # Check .env file (warning if missing)
    if os.path.exists('config/.env'):
        print_status("Environment file config/.env", 'PASS')
    else:
        print_status("Environment file config/.env missing (use .env.example)", 'WARN')
    
    return all_pass

def check_custom_modules():
    """Check custom modules can be imported."""
    sys.path.append('src')
    
    modules = [
        'video_capture',
        'face_detection', 
        'face_recognition_module',
        'telegram_alerts',
        'database_management',
        'main'
    ]
    
    all_pass = True
    for module in modules:
        try:
            importlib.import_module(module)
            print_status(f"Module {module}", 'PASS')
        except ImportError as e:
            print_status(f"Module {module}: {e}", 'FAIL')
            all_pass = False
    
    return all_pass

def check_camera_access():
    """Check camera accessibility."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print_status("Camera access", 'PASS')
            cap.release()
            return True
        else:
            print_status("Camera not accessible", 'WARN')
            return False
    except Exception as e:
        print_status(f"Camera check failed: {e}", 'FAIL')
        return False

def run_basic_tests():
    """Run basic unit tests."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/', '-q', '--tb=no'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print_status("Unit tests", 'PASS')
            return True
        else:
            print_status(f"Unit tests failed: {result.stdout}", 'FAIL')
            return False
    except subprocess.TimeoutExpired:
        print_status("Unit tests timed out", 'FAIL')
        return False
    except Exception as e:
        print_status(f"Unit test error: {e}", 'FAIL')
        return False

def check_system_resources():
    """Check system resources."""
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.total >= 2 * 1024**3:  # 2GB
            print_status(f"Memory: {memory.total // 1024**3}GB available", 'PASS')
        else:
            print_status(f"Memory: {memory.total // 1024**3}GB (recommended: 4GB+)", 'WARN')
        
        # Check disk space
        disk = psutil.disk_usage('.')
        free_gb = disk.free // 1024**3
        if free_gb >= 2:
            print_status(f"Disk space: {free_gb}GB free", 'PASS')
        else:
            print_status(f"Disk space: {free_gb}GB free (low)", 'WARN')
        
        return True
    except ImportError:
        print_status("System resource check skipped (psutil not available)", 'INFO')
        return True

def main():
    """Run comprehensive system health check."""
    print("🔍 Facial Detection System - Health Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_configuration_files),
        ("Custom Modules", check_custom_modules),
        ("Camera Access", check_camera_access),
        ("System Resources", check_system_resources),
        ("Unit Tests", run_basic_tests),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_status(f"Check failed with exception: {e}", 'FAIL')
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = 'PASS' if result else 'FAIL'
        print_status(f"{check_name:<20}", status)
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print_status("System is ready for deployment! 🚀", 'PASS')
        return 0
    elif passed >= total * 0.8:
        print_status("System mostly ready, address warnings", 'WARN')
        return 0
    else:
        print_status("System needs attention before deployment", 'FAIL')
        return 1

if __name__ == "__main__":
    sys.exit(main())
