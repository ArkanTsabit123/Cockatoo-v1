# cockatoo_v1/scripts/setup.py

"""
cockatoo_v1 Setup and Installation Script
"""

import sys
import os
import platform
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime

def print_header(text):
    """Display formatted section header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def check_python_version():
    """Verify Python version meets minimum requirements."""
    print("Verifying Python version...")
    major, minor, micro = sys.version_info[:3]
    
    if major < 3 or (major == 3 and minor < 11):
        print(f"ERROR: Python 3.11+ required. Current version: {major}.{minor}.{micro}")
        return False
    print(f"SUCCESS: Python {major}.{minor}.{micro}")
    return True

def install_with_pip(package_spec, upgrade_pip=True):
    """Install package using pip with extended error handling."""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade_pip:
            cmd.append("--break-system-packages")
        cmd.append(package_spec)
        
        print(f"  Installing: {package_spec}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"    SUCCESS: {package_spec}")
            return True
        
        print(f"    FAILED: {package_spec}")
        if "error" in result.stderr.lower():
            error_lines = [line for line in result.stderr.split('\n') 
                          if 'error' in line.lower()]
            for err in error_lines[:3]:
                print(f"      {err[:100]}")
        return False
            
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: {package_spec}")
        return False
    except Exception as e:
        print(f"    ERROR: {package_spec} - {e}")
        return False

def install_requirements_313():
    """Install packages with priority grouping for Python 3.13."""
    print_header("PACKAGE INSTALLATION")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"ERROR: requirements.txt not found at {requirements_file}")
        return False
    
    print(f"Using requirements from: {requirements_file}")
    
    print("\nUpgrading pip...")
    install_with_pip("pip", upgrade_pip=False)
    
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]
    
    print(f"\nFound {len(requirements)} packages to install")
    
    priority_1, priority_2, priority_3, priority_4 = [], [], [], []
    
    for req in requirements:
        req_lower = req.lower()
        
        if any(keyword in req_lower for keyword in [
            'fastapi', 'uvicorn', 'pydantic', 'pyyaml', 
            'requests', 'python-dotenv', 'click', 'rich'
        ]):
            priority_1.append(req)
        elif any(keyword in req_lower for keyword in [
            'pypdf', 'docx', 'beautifulsoup', 'lxml', 
            'pillow', 'markdown', 'customtkinter'
        ]):
            priority_2.append(req)
        elif any(keyword in req_lower for keyword in [
            'torch', 'transformers', 'sentence', 'numpy',
            'scipy', 'scikit', 'chromadb', 'sqlalchemy'
        ]):
            priority_3.append(req)
        else:
            priority_4.append(req)
    
    results = {
        'priority_1': {'total': len(priority_1), 'success': 0},
        'priority_2': {'total': len(priority_2), 'success': 0},
        'priority_3': {'total': len(priority_3), 'success': 0},
        'priority_4': {'total': len(priority_4), 'success': 0},
    }
    
    print("\nInstalling Priority 1 - Core Framework")
    for package in priority_1:
        if install_with_pip(package):
            results['priority_1']['success'] += 1
    
    print("\nInstalling Priority 2 - Document Processing")
    for package in priority_2:
        if install_with_pip(package):
            results['priority_2']['success'] += 1
    
    print("\nInstalling Priority 3 - AI/ML Components")
    for package in priority_3:
        if install_with_pip(package):
            results['priority_3']['success'] += 1
    
    print("\nInstalling Priority 4 - Optional Dependencies")
    for package in priority_4:
        if install_with_pip(package):
            results['priority_4']['success'] += 1
    
    print_header("INSTALLATION SUMMARY")
    
    total_packages = sum(r['total'] for r in results.values())
    total_success = sum(r['success'] for r in results.values())
    
    print(f"Total packages: {total_packages}")
    print(f"Successfully installed: {total_success}")
    print(f"Failed: {total_packages - total_success}")
    
    print("\nBreakdown by priority:")
    for key, data in results.items():
        success_rate = (data['success'] / data['total']) * 100 if data['total'] > 0 else 100
        print(f"  {key}: {data['success']}/{data['total']} ({success_rate:.1f}%)")
    
    core_success_rate = results['priority_1']['success'] / results['priority_1']['total'] if results['priority_1']['total'] > 0 else 0
    
    if core_success_rate >= 0.8:
        print(f"\nSUCCESS: Core packages installed ({core_success_rate*100:.1f}%)")
        return True
    
    print(f"\nFAILED: Insufficient core packages installed ({core_success_rate*100:.1f}%)")
    return False

def create_directories_simple():
    """Create essential project directories."""
    print_header("CREATING DIRECTORIES")
    
    project_root = Path(__file__).parent.parent
    
    directories = [
        project_root / "data",
        project_root / "data" / "config",
        project_root / "data" / "logs",
        project_root / "data" / "database",
        project_root / "data" / "models",
        project_root / "data" / "documents",
        project_root / "data" / "exports",
        project_root / "data" / "cache",
        
        Path.home() / ".cockatoo_v1",
        Path.home() / ".cockatoo_v1" / "config",
        Path.home() / ".cockatoo_v1" / "documents",
        Path.home() / ".cockatoo_v1" / "models",
    ]
    
    created = 0
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                print(f"  {directory}")
                created += 1
            else:
                print(f"  Failed: {directory}")
        except Exception as e:
            print(f"  Error: {directory} - {e}")
    
    print(f"\nCreated {created} directories")
    return created > 0

def setup_crossplatform_data_dirs() -> bool:
    """Setup cross-platform data directories for production deployment."""
    print_header("SETUP CROSS-PLATFORM DATA DIRECTORIES")
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from core.config import setup_crossplatform_dirs
        
        directories = setup_crossplatform_dirs()
        
        print(f"Created {len(directories)} directories:")
        for name, path in directories.items():
            if path.exists():
                print(f"  SUCCESS: {name}: {path}")
            else:
                print(f"  FAILED: {name}: NOT CREATED")
                return False
        
        from utilities.helpers import create_crossplatform_directories
        helper_dirs = create_crossplatform_directories()
        
        print(f"Total directories verified: {len(helper_dirs)}")
        return True
        
    except Exception as e:
        print(f"Error setting up cross-platform directories: {e}")
        
        try:
            system = platform.system()
            
            if system == "Windows":
                base = Path(os.environ.get('APPDATA', 
                         Path.home() / 'AppData' / 'Roaming')) / "cockatoo_v1"
            elif system == "Darwin":
                base = Path.home() / "Library" / "Application Support" / "cockatoo_v1"
            elif system == "Linux":
                base = Path.home() / ".local" / "share" / "cockatoo_v1"
            else:
                base = Path.home() / ".cockatoo_v1"
            
            subdirs = ["models", "documents", "database", "logs", "config", "cache"]
            for subdir in subdirs:
                (base / subdir).mkdir(parents=True, exist_ok=True)
            
            print(f"Fallback directories created at: {base}")
            return True
            
        except Exception as fallback_error:
            print(f"Fallback directory creation failed: {fallback_error}")
            return False

def setup_crossplatform_dirs():
    """Setup cross-platform directories with enhanced compatibility."""
    print_header("SETUP CROSS-PLATFORM DIRECTORIES")
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from utilities.helpers import setup_crossplatform_directories
        
        directories = setup_crossplatform_directories()
        
        print("Created directories:")
        for name, path in directories.items():
            print(f"  {name}: {path}")
        
        test_file = directories['data'] / ".setup_complete"
        test_file.write_text(f"Setup completed: {datetime.now()}")
        
        print(f"Successfully created {len(directories)} directories")
        return True
        
    except Exception as e:
        print(f"Error creating directories: {e}")
        
        try:
            home = Path.home()
            base_dir = home / ".cockatoo_v1"
            subdirs = ["models", "documents", "database", "logs", "config"]
            
            for subdir in subdirs:
                (base_dir / subdir).mkdir(parents=True, exist_ok=True)
                print(f"  Created: {base_dir / subdir}")
            
            print(f"Created fallback directories at: {base_dir}")
            return True
        except Exception as fallback_error:
            print(f"Fallback directory creation failed: {fallback_error}")
            return False

def check_ollama_simple():
    """Verify Ollama installation and service status."""
    print_header("CHECKING OLLAMA")
    
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        
        if result.returncode == 0:
            print(f"Ollama installed: {result.stdout.strip()}")
            
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=3)
                if response.status_code == 200:
                    print("Ollama service is running")
                    models = response.json().get('models', [])
                    if models:
                        print(f"Available models: {len(models)}")
                    else:
                        print("No models installed")
                else:
                    print("Ollama service not responding")
            except:
                print("Could not verify Ollama service status")
            
            return True
        print("Ollama found but command execution failed")
        return False
            
    except FileNotFoundError:
        print("Ollama not installed (optional component)")
        print("\nTo install Ollama for AI functionality:")
        print("  1. Download from https://ollama.ai")
        print("  2. Install and execute: ollama pull llama2:7b")
        print("  3. Start service: ollama serve")
        return False
    except Exception as e:
        print(f"Ollama verification error: {e}")
        return False

def verify_core_installation():
    """Validate core package installation status."""
    print_header("VERIFYING INSTALLATION")
    
    core_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("PyYAML", "yaml"),
        ("requests", "requests"),
        ("customtkinter", "customtkinter"),
        ("PyPDF2", "PyPDF2"),
        ("chromadb", "chromadb"),
    ]
    
    installed, missing = [], []
    
    for pip_name, import_name in core_packages:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                raise ImportError
            installed.append(pip_name)
            print(f"  {pip_name}")
        except ImportError:
            missing.append(pip_name)
            print(f"  MISSING: {pip_name}")
    
    print(f"\nVerification: {len(installed)}/{len(core_packages)} packages installed")
    
    if not missing:
        print("All core packages installed successfully")
        return True
    elif len(missing) <= 2:
        print(f"{len(missing)} packages missing: {', '.join(missing)}")
        print("Limited functionality may be experienced")
        return True
    
    print(f"{len(missing)} packages missing: {', '.join(missing)}")
    return False

def display_next_steps():
    """Display post-installation instructions."""
    print_header("SETUP COMPLETE")
    
    print("\nNEXT STEPS:")
    print("1. Test the installation:")
    print("   python -c \"import fastapi; import chromadb; print('Core packages verified')\"")
    
    print("\n2. Launch cockatoo_v1:")
    print("   GUI interface:   python app.py")
    print("   CLI interface:   python app.py cli")
    print("   Web interface:   python app.py web")
    
    print("\n3. For AI functionality, install Ollama:")
    print("   - Download from https://ollama.ai")
    print("   - Execute: ollama pull llama2:7b")
    print("   - Initialize: ollama serve")
    
    print("\n4. Add documents for processing:")
    print("   - Place files in: ~/.cockatoo_v1/documents/")
    print("   - Or utilize the application upload functionality")
    
    print("\n5. Begin document querying")
    print("=" * 60)

def main():
    """Primary setup execution function."""
    print("=" * 60)
    print(" cockatoo_v1 - PYTHON 3.13 COMPATIBILITY SETUP")
    print("=" * 60)
    
    print(f"\nPython Version: {platform.python_version()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    setup_steps = [
        ("Python Version", check_python_version, True),
        ("Package Installation", install_requirements_313, True),
        ("Directory Setup", create_directories_simple, True),
        ("Cross-Platform Directories", setup_crossplatform_dirs, True),
        ("Cross-Platform Data Directories", setup_crossplatform_data_dirs, True),
        ("Installation Verify", verify_core_installation, True),
        ("Ollama Check", check_ollama_simple, False),
    ]
    
    results = []
    critical_failures = 0
    
    for step_name, step_func, is_critical in setup_steps:
        print_header(step_name.upper())
        
        try:
            success = step_func()
            status = "PASS" if success else ("WARNING" if not is_critical else "FAIL")
            results.append((step_name, success, is_critical, status))
            
            if not success and is_critical:
                critical_failures += 1
                
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((step_name, False, is_critical, "ERROR"))
            if is_critical:
                critical_failures += 1
    
    print_header("SETUP SUMMARY")
    
    for step_name, success, is_critical, status in results:
        critical_marker = " (CRITICAL)" if is_critical else ""
        print(f"{status:8} {step_name}{critical_marker}")
    
    if critical_failures > 0:
        print(f"\nSETUP FAILED with {critical_failures} critical error(s)")
        print("\nTroubleshooting recommendations:")
        print("1. Update pip: python -m pip install --upgrade pip")
        print("2. Manual core installation: pip install fastapi uvicorn pydantic")
        print("3. Verify Python version: 3.11, 3.12, or 3.13 required")
        return False
    
    display_next_steps()
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(130)