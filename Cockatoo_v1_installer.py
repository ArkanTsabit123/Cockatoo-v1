#!/usr/bin/env python3
"""
Cockatoo_v1 Core Dependencies Installer with Progress Tracking
Version: 1.1.0 | Python: 3.10.11
"""

import subprocess
import sys
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class DependencyInstaller:
    def __init__(self):
        self.start_time = datetime.now()
        self.installation_log = []
        self.successful_installs = []
        self.failed_installs = []
        self.packages = self._load_package_list()
        
    def _load_package_list(self) -> List[Dict]:
        """Load all core dependencies with categorization."""
        return [
            # CORE BUILD TOOLS
            {"name": "setuptools", "version": "68.2.0", "category": "core", "priority": 1},
            {"name": "wheel", "version": "0.41.0", "category": "core", "priority": 1},
            {"name": "pip", "version": "23.0.1", "category": "core", "priority": 1},
            
            # AI & MACHINE LEARNING CORE
            {"name": "torch", "version": "2.2.0", "category": "ai", "priority": 1},
            {"name": "torchvision", "version": "0.17.0", "category": "ai", "priority": 1},
            {"name": "torchaudio", "version": "2.2.0", "category": "ai", "priority": 1},
            {"name": "transformers", "version": "4.40.0", "category": "ai", "priority": 1},
            {"name": "sentence-transformers", "version": "2.2.2", "category": "ai", "priority": 2},
            {"name": "accelerate", "version": "0.25.0", "category": "ai", "priority": 2},
            {"name": "tokenizers", "version": "0.19.0", "category": "ai", "priority": 2},
            
            # LangChain Ecosystem
            {"name": "langchain", "version": "0.1.20", "category": "langchain", "priority": 1},
            {"name": "langchain-community", "version": "0.0.38", "category": "langchain", "priority": 2},
            {"name": "langchain-core", "version": "0.1.53", "category": "langchain", "priority": 1},
            {"name": "langchain-text-splitters", "version": "0.0.2", "category": "langchain", "priority": 3},
            {"name": "langchain-chroma", "version": "0.1.4", "category": "langchain", "priority": 2},
            
            # OpenAI
            {"name": "openai", "version": "2.16.0", "category": "api", "priority": 3},
            
            # Numerical Computing
            {"name": "numpy", "version": "1.24.4", "category": "numerical", "priority": 1},
            {"name": "scipy", "version": "1.15.3", "category": "numerical", "priority": 2},
            {"name": "scikit-learn", "version": "1.3.0", "category": "numerical", "priority": 2},
            
            # HuggingFace
            {"name": "huggingface-hub", "version": "0.36.0", "category": "huggingface", "priority": 2},
            {"name": "datasets", "version": "4.5.0", "category": "huggingface", "priority": 3},
            
            # Vector Database
            {"name": "chromadb", "version": "0.4.18", "category": "database", "priority": 1},
            {"name": "hnswlib", "version": "0.7.0", "category": "database", "priority": 2},
            
            # DOCUMENT PROCESSING
            # PDF
            {"name": "PyPDF2", "version": "3.0.1", "category": "documents", "priority": 2},
            {"name": "pdfplumber", "version": "0.10.2", "category": "documents", "priority": 2},
            {"name": "pypdf", "version": "3.17.0", "category": "documents", "priority": 2},
            
            # Office Documents
            {"name": "python-docx", "version": "0.8.11", "category": "documents", "priority": 3},
            {"name": "openpyxl", "version": "3.1.0", "category": "documents", "priority": 3},
            
            # Ebooks
            {"name": "ebooklib", "version": "0.18", "category": "documents", "priority": 3},
            
            # Web Content
            {"name": "beautifulsoup4", "version": "4.12.2", "category": "web", "priority": 2},
            {"name": "lxml", "version": "4.9.0", "category": "web", "priority": 2},
            {"name": "html5lib", "version": "1.1", "category": "web", "priority": 3},
            
            # Text Processing
            {"name": "markdown-it-py", "version": "3.0.0", "category": "text", "priority": 3},
            {"name": "python-frontmatter", "version": "1.0.0", "category": "text", "priority": 3},
            
            # Data Processing
            {"name": "pandas", "version": "2.1.0", "category": "data", "priority": 2},
            
            # Images & Text Extraction
            {"name": "Pillow", "version": "10.0.0", "category": "images", "priority": 2},
            {"name": "markdown", "version": "3.5", "category": "text", "priority": 3},
            {"name": "chardet", "version": "5.2.0", "category": "text", "priority": 2},
            
            # DESKTOP GUI
            {"name": "customtkinter", "version": "5.2.0", "category": "gui", "priority": 3},
            {"name": "tkinterweb", "version": "4.17.1", "category": "gui", "priority": 3},
            
            # DATABASE & STORAGE
            {"name": "sqlalchemy", "version": "2.0.0", "category": "database", "priority": 2},
            {"name": "alembic", "version": "1.13.0", "category": "database", "priority": 3},
            
            # UTILITIES
            # Configuration
            {"name": "pyyaml", "version": "6.0", "category": "utils", "priority": 2},
            {"name": "python-dotenv", "version": "1.0.0", "category": "utils", "priority": 2},
            
            # Networking
            {"name": "requests", "version": "2.32.5", "category": "networking", "priority": 1},
            {"name": "aiohttp", "version": "3.9.0", "category": "networking", "priority": 2},
            
            # System & Monitoring
            {"name": "psutil", "version": "5.9.0", "category": "system", "priority": 2},
            {"name": "watchdog", "version": "3.0.0", "category": "system", "priority": 3},
            
            # CLI & Output
            {"name": "click", "version": "8.1.0", "category": "cli", "priority": 2},
            {"name": "rich", "version": "13.0.0", "category": "cli", "priority": 2},
            {"name": "tqdm", "version": "4.67.1", "category": "cli", "priority": 2},
            
            # Data Serialization
            {"name": "orjson", "version": "3.9.14", "category": "data", "priority": 3},
            {"name": "python-multipart", "version": "0.0.6", "category": "web", "priority": 3},
            
            # Security
            {"name": "cryptography", "version": "41.0.0", "category": "security", "priority": 2},
            
            # Performance
            {"name": "cachetools", "version": "5.3.0", "category": "performance", "priority": 3},
            {"name": "joblib", "version": "1.3.0", "category": "performance", "priority": 3},
            
            # Logging
            {"name": "structlog", "version": "25.5.0", "category": "utils", "priority": 3},
            
            # Typing & Validation
            {"name": "typing-extensions", "version": "4.15.0", "category": "typing", "priority": 1},
            {"name": "jsonschema", "version": "4.19.0", "category": "validation", "priority": 3},
            {"name": "pydantic", "version": "2.5.0", "category": "validation", "priority": 2},
            {"name": "pydantic-settings", "version": "2.1.0", "category": "validation", "priority": 3},
            
            # WEB FRAMEWORK (Optional)
            {"name": "fastapi", "version": "0.104.0", "category": "web", "priority": 3},
            {"name": "uvicorn[standard]", "version": "0.40.0", "category": "web", "priority": 3},
            
            # LOCAL LLM (Optional)
            {"name": "ollama", "version": "0.1.0", "category": "local-llm", "priority": 3},
            
            # WEB UI (Optional)
            {"name": "streamlit", "version": "1.28.0", "category": "web-ui", "priority": 3},
            
            # DEVELOPMENT & TESTING (Optional)
            # Testing
            {"name": "pytest", "version": "7.4.0", "category": "testing", "priority": 3},
            {"name": "pytest-asyncio", "version": "0.23.0", "category": "testing", "priority": 3},
            
            # Code Quality
            {"name": "black", "version": "26.1.0", "category": "dev", "priority": 3},
            {"name": "flake8", "version": "7.0.0", "category": "dev", "priority": 3},
            {"name": "mypy", "version": "1.19.1", "category": "dev", "priority": 3},
            
            # Jupyter & Interactive
            {"name": "ipython", "version": "8.12.0", "category": "dev", "priority": 3},
            {"name": "jupyter", "version": "1.0.0", "category": "dev", "priority": 3},
            {"name": "jupyterlab", "version": "4.5.3", "category": "dev", "priority": 3},
            
            # Packaging
            {"name": "packaging", "version": "23.2", "category": "packaging", "priority": 2},
            {"name": "platformdirs", "version": "4.1.0", "category": "packaging", "priority": 3},
            {"name": "pytest-cov", "version": "4.0.0", "category": "testing", "priority": 3},
            {"name": "pytest-mock", "version": "3.0.0", "category": "testing", "priority": 3},
            {"name": "pytesseract", "version": "0.3.10", "category": "ocr", "priority": 3},
        ]
    
    def check_python_version(self) -> bool:
        """Verify Python version meets requirements."""
        required_version = (3, 10, 0)
        current_version = sys.version_info
        
        if current_version < required_version:
            print(f"ERROR: Python 3.10.0 or higher is required.")
            print(f"Current version: {sys.version}")
            return False
        
        print(f"✓ Python version check passed: {sys.version}")
        return True
    
    def get_pip_command(self) -> List[str]:
        """Get the pip command to use."""
        return [sys.executable, "-m", "pip", "install"]
    
    def install_package(self, package: Dict) -> Tuple[bool, str]:
        """Install a single package with version specification."""
        package_name = package["name"]
        version = package["version"]
        
        # Handle packages with extras differently
        if "[" in package_name:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = f"{package_name}=={version}"
        
        pip_cmd = self.get_pip_command()
        
        try:
            print(f"  Installing {package_name}=={version}...", end=" ", flush=True)
            start_time = time.time()
            
            result = subprocess.run(
                pip_cmd + [package_spec],
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.1f}s)")
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "package": package_name,
                "version": version,
                "status": "success",
                "elapsed_time": elapsed,
                "output": result.stdout[-500:] if result.stdout else ""
            }
            self.installation_log.append(log_entry)
            self.successful_installs.append(package_name)
            
            return True, "Success"
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"✗ ({elapsed:.1f}s)")
            print(f"    Error: {e.stderr[:200] if e.stderr else 'Unknown error'}")
            
            # Try without version constraint
            print(f"  Retrying {package_name} (latest)...", end=" ", flush=True)
            try:
                subprocess.run(
                    pip_cmd + [package_name],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print("✓ (fallback)")
                
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "package": package_name,
                    "version": "latest",
                    "status": "success_fallback",
                    "elapsed_time": elapsed,
                    "output": "Installed latest version"
                }
                self.installation_log.append(log_entry)
                self.successful_installs.append(package_name)
                
                return True, "Success (fallback to latest)"
                
            except subprocess.CalledProcessError as e2:
                elapsed_total = time.time() - start_time
                print(f"✗ ({elapsed_total:.1f}s)")
                
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "package": package_name,
                    "version": version,
                    "status": "failed",
                    "elapsed_time": elapsed_total,
                    "error": str(e2.stderr)[:500] if e2.stderr else str(e2)
                }
                self.installation_log.append(log_entry)
                self.failed_installs.append(package_name)
                
                return False, f"Failed: {str(e2)[:100]}"
    
    def install_by_category(self, category: str) -> Dict[str, int]:
        """Install all packages in a specific category."""
        category_packages = [p for p in self.packages if p["category"] == category]
        category_packages.sort(key=lambda x: x["priority"])
        
        if not category_packages:
            return {"total": 0, "success": 0, "failed": 0}
        
        print(f"\n{'='*60}")
        print(f"Installing {category.upper()} packages")
        print(f"{'='*60}")
        
        success = 0
        failed = 0
        
        for i, package in enumerate(category_packages, 1):
            print(f"[{i}/{len(category_packages)}] ", end="")
            installed, message = self.install_package(package)
            if installed:
                success += 1
            else:
                failed += 1
        
        return {"total": len(category_packages), "success": success, "failed": failed}
    
    def install_by_priority(self, priority: int) -> Dict[str, int]:
        """Install all packages with a specific priority level."""
        priority_packages = [p for p in self.packages if p["priority"] == priority]
        
        if not priority_packages:
            return {"total": 0, "success": 0, "failed": 0}
        
        priority_name = {1: "ESSENTIAL", 2: "IMPORTANT", 3: "OPTIONAL"}[priority]
        print(f"\n{'='*60}")
        print(f"Installing {priority_name} packages (Priority {priority})")
        print(f"{'='*60}")
        
        success = 0
        failed = 0
        
        for i, package in enumerate(priority_packages, 1):
            print(f"[{i}/{len(priority_packages)}] ", end="")
            installed, message = self.install_package(package)
            if installed:
                success += 1
            else:
                failed += 1
        
        return {"total": len(priority_packages), "success": success, "failed": failed}
    
    def install_all(self) -> bool:
        """Install all packages with comprehensive tracking."""
        print("="*60)
        print("COCKATOO_v1 DEPENDENCY INSTALLER")
        print(f"Version: 1.1.0 | Python: 3.10.11")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        if not self.check_python_version():
            return False
        
        print("\nSystem Information:")
        print(f"  Platform: {platform.platform()}")
        print(f"  Machine: {platform.machine()}")
        print(f"  Processor: {platform.processor()}")
        
        # First update pip
        print("\n" + "="*60)
        print("Updating pip...")
        print("="*60)
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                          check=True, capture_output=True)
            print("✓ pip updated successfully")
        except subprocess.CalledProcessError:
            print("⚠ Could not update pip, continuing...")
        
        # Install by priority levels
        results = {}
        for priority in [1, 2, 3]:
            results[f"priority_{priority}"] = self.install_by_priority(priority)
        
        # Alternative: Install by category
        # categories = sorted(set(p["category"] for p in self.packages))
        # for category in categories:
        #     results[f"category_{category}"] = self.install_by_category(category)
        
        # Generate summary
        self._generate_summary(results)
        
        # Save installation log
        self._save_installation_log()
        
        return len(self.failed_installs) == 0
    
    def _generate_summary(self, results: Dict):
        """Generate and display installation summary."""
        total_packages = len(self.packages)
        total_success = len(self.successful_installs)
        total_failed = len(self.failed_installs)
        
        elapsed_time = datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print("INSTALLATION SUMMARY")
        print("="*60)
        print(f"Total packages: {total_packages}")
        print(f"Successfully installed: {total_success}")
        print(f"Failed installations: {total_failed}")
        print(f"Elapsed time: {elapsed_time}")
        
        # Priority breakdown
        print("\nPriority Breakdown:")
        for priority in [1, 2, 3]:
            priority_name = {1: "Essential", 2: "Important", 3: "Optional"}[priority]
            packages_in_priority = [p for p in self.packages if p["priority"] == priority]
            success_in_priority = [p for p in packages_in_priority if p["name"] in self.successful_installs]
            print(f"  {priority_name} (Priority {priority}): {len(success_in_priority)}/{len(packages_in_priority)}")
        
        # Category breakdown
        print("\nCategory Breakdown:")
        categories = {}
        for package in self.packages:
            cat = package["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "success": 0}
            categories[cat]["total"] += 1
            if package["name"] in self.successful_installs:
                categories[cat]["success"] += 1
        
        for cat, stats in sorted(categories.items()):
            success_rate = (stats["success"] / stats["total"]) * 100
            print(f"  {cat}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        if total_failed > 0:
            print(f"\n⚠ Failed packages ({total_failed}):")
            for package in self.failed_installs:
                print(f"  - {package}")
            
            print("\nYou can try installing failed packages manually:")
            for package_name in self.failed_installs:
                package_info = next((p for p in self.packages if p["name"] == package_name), None)
                if package_info:
                    print(f"  pip install {package_name}=={package_info['version']}")
        
        if total_success == total_packages:
            print(f"\n✅ All {total_packages} packages installed successfully!")
        else:
            success_rate = (total_success / total_packages) * 100
            print(f"\n⚠ {success_rate:.1f}% of packages installed successfully")
    
    def _save_installation_log(self):
        """Save installation log to JSON file."""
        log_data = {
            "metadata": {
                "project": "Cockatoo_v1",
                "version": "1.1.0",
                "python_version": sys.version,
                "platform": platform.platform(),
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "elapsed_seconds": (datetime.now() - self.start_time).total_seconds()
            },
            "summary": {
                "total_packages": len(self.packages),
                "successful_installs": len(self.successful_installs),
                "failed_installs": len(self.failed_installs),
                "successful": self.successful_installs,
                "failed": self.failed_installs
            },
            "detailed_log": self.installation_log
        }
        
        log_file = Path("cockatoo_installation_log.json")
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n📄 Installation log saved to: {log_file.absolute()}")
        
        # Also save requirements file
        self._save_requirements_file()
    
    def _save_requirements_file(self):
        """Save requirements to a text file."""
        requirements_file = Path("cockatoo_requirements.txt")
        with open(requirements_file, "w") as f:
            f.write("# Cockatoo_v1 Requirements\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Python: {sys.version}\n")
            f.write(f"# Platform: {platform.platform()}\n\n")
            
            for package in sorted(self.packages, key=lambda x: x["name"]):
                if "[" in package["name"]:
                    f.write(f"{package['name']}=={package['version']}\n")
                else:
                    f.write(f"{package['name']}=={package['version']}\n")
        
        print(f"📄 Requirements file saved to: {requirements_file.absolute()}")

def main():
    """Main entry point."""
    installer = DependencyInstaller()
    
    try:
        success = installer.install_all()
        
        if success:
            print("\n" + "="*60)
            print("✅ INSTALLATION COMPLETED SUCCESSFULLY")
            print("="*60)
            print("\nNext steps:")
            print("1. Check the installation log for details: cockatoo_installation_log.json")
            print("2. Verify installation: python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"")
            print("3. Run your Cockatoo_v1 application")
        else:
            print("\n" + "="*60)
            print("⚠ INSTALLATION COMPLETED WITH ERRORS")
            print("="*60)
            print("\nSome packages failed to install. Check the log for details.")
            print("You may need to install failed packages manually.")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Installation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()