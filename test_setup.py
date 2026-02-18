#!/usr/bin/env python3
"""
Setup Test Script for 3D Face Reconstruction
Verifies that all dependencies are installed correctly
"""

import sys
import importlib
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def check_module(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Check if a module can be imported
    
    Args:
        module_name: Name of the module to import
        package_name: Display name (if different from module_name)
    
    Returns:
        Tuple of (success, version_or_error)
    """
    display_name = package_name or module_name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


def print_result(name: str, success: bool, info: str):
    """Print test result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if success else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"  {status} | {name:25} | {info}")


def main():
    """Main test function"""
    print_header("3D Face Reconstruction - Setup Verification")
    
    print(f"{Colors.BOLD}Python Version:{Colors.RESET}")
    print(f"  {sys.version}\n")
    
    # Define required modules
    required_modules = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('open3d', 'Open3D'),
        ('scipy', 'SciPy'),
        ('yaml', 'PyYAML'),
    ]
    
    optional_modules = [
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_passed = True
    
    # Check required modules
    print(f"{Colors.BOLD}Required Dependencies:{Colors.RESET}\n")
    for module_name, display_name in required_modules:
        success, info = check_module(module_name, display_name)
        print_result(display_name, success, f"v{info}" if success else info)
        if not success:
            all_passed = False
    
    # Check optional modules
    print(f"\n{Colors.BOLD}Optional Dependencies:{Colors.RESET}\n")
    for module_name, display_name in optional_modules:
        success, info = check_module(module_name, display_name)
        print_result(display_name, success, f"v{info}" if success else "Not installed")
    
    # Check project modules
    print(f"\n{Colors.BOLD}Project Modules:{Colors.RESET}\n")
    project_modules = [
        'utils',
        'geometry_engine',
        'visualization',
        'export_mesh',
    ]
    
    for module_name in project_modules:
        success, info = check_module(module_name)
        print_result(module_name, success, "Available" if success else info)
        if not success:
            all_passed = False
    
    # Test camera access
    print(f"\n{Colors.BOLD}Hardware Check:{Colors.RESET}\n")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print_result("Camera Access", True, "Webcam detected and accessible")
            cap.release()
        else:
            print_result("Camera Access", False, "Cannot open camera device 0")
            print(f"  {Colors.YELLOW}Note: Camera may be in use by another application{Colors.RESET}")
    except Exception as e:
        print_result("Camera Access", False, str(e))
    
    # Final summary
    print_header("Summary")
    
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All required dependencies are installed!{Colors.RESET}\n")
        print("You can now run the application:")
        print(f"  {Colors.BLUE}python main.py live{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some dependencies are missing!{Colors.RESET}\n")
        print("Please install missing dependencies:")
        print(f"  {Colors.BLUE}pip install -r requirements.txt{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}Unexpected error: {e}{Colors.RESET}")
        sys.exit(1)
