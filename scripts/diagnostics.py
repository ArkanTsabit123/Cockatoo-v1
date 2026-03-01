# data/scripts/diagnostic.py
"""
Cockatoo V1 Diagnostic System

This module provides comprehensive diagnostics for Cockatoo V1 installation.
It performs system checks, validates configuration, tests dependencies,
and generates detailed reports for troubleshooting and maintenance.
"""

import os
import sys
import json
import platform
import sqlite3
import subprocess
import importlib
import socket
import time
import hashlib
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.exceptions import CockatooError
    from src.core.logger import get_logger
    from src.core.config import AppConfig, ConfigManager
    from src.core.app import DocumentProcessor, AIProcessor, StorageManager
    HAS_INTERNAL_IMPORTS = True
except ImportError as import_error:
    HAS_INTERNAL_IMPORTS = False
    print(f"Warning: Internal modules import failed ({import_error}). Running in standalone mode.")


class CockatooDiagnostic:
    """
    Main diagnostic controller for Cockatoo V1.
    
    Orchestrates all diagnostic checks, collects results, manages auto-fix operations,
    and generates comprehensive reports for system analysis and troubleshooting.
    """

    def __init__(self, debug_mode: bool = False, auto_fix: bool = False, mode: str = "project"):
        """
        Initialize diagnostic system with specified operating parameters.

        Args:
            debug_mode: Enable verbose debug output for troubleshooting
            auto_fix: Attempt to automatically resolve detected issues
            mode: Check mode - "user", "project", or "both"
        
        Raises:
            ValueError: If mode is not one of "user", "project", or "both"
        """
        valid_modes = ["user", "project", "both"]
        if mode.lower() not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}")
        
        self.debug_mode = debug_mode
        self.auto_fix = auto_fix
        self.mode = mode.lower()
        self.fixes_applied = []
        self.fixes_failed = []

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'diagnostic_id': self._generate_diagnostic_id(),
            'mode': self.mode,
            'system': {},
            'installation': {},
            'dependencies': {},
            'data': {},
            'configuration': {},
            'performance': {},
            'issues': [],
            'fixes_applied': [],
            'fixes_failed': [],
            'summary': {}
        }

        if HAS_INTERNAL_IMPORTS:
            self.logger = get_logger(__name__)
        else:
            self.logger = None

    def _generate_diagnostic_id(self) -> str:
        """Generate unique identifier for this diagnostic run."""
        unique_string = f"{datetime.now().isoformat()}{platform.node()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:8]

    def _should_check_location(self, key: str) -> bool:
        """
        Determine if a location should be checked based on current mode.

        Args:
            key: Dictionary key that indicates location (prefix 'user_' or 'project_')

        Returns:
            True if location should be checked, False otherwise
        """
        if self.mode == "both":
            return True
        elif self.mode == "user":
            return key.startswith('user_')
        elif self.mode == "project":
            return key.startswith('project_')
        return False

    def run_all_checks(self) -> Dict[str, Any]:
        """
        Execute complete diagnostic suite in defined order.

        Returns:
            Dictionary containing all diagnostic results
        """
        mode_display = {
            "user": "USER HOME DIRECTORY",
            "project": "PROJECT DIRECTORY",
            "both": "USER AND PROJECT DIRECTORIES"
        }
        
        print("Cockatoo V1 Diagnostic System")
        print(f"Mode: {mode_display.get(self.mode, self.mode.upper())}")
        print("=" * 70)

        checks = [
            ("System Requirements", self.check_system),
            ("Python Environment", self.check_python_environment),
            ("Project Structure", self.check_project_structure),
            ("Dependencies", self.check_dependencies),
            ("Data Directories", self.check_data_directories),
            ("Configuration Files", self.check_configuration),
            ("Database Integrity", self.check_databases),
            ("File Permissions", self.check_permissions),
            ("Network Connectivity", self.check_network_connectivity),
            ("Performance Benchmarks", self.check_performance),
            ("Model Availability", self.check_models),
            ("Integration Tests", self.check_integration)
        ]

        total_checks = len(checks)

        for index, (check_name, check_func) in enumerate(checks, 1):
            print(f"[{index}/{total_checks}] {check_name}...")
            try:
                check_func()
                if self.debug_mode:
                    print(f"  Completed: {check_name}")
            except Exception as error:
                self._add_issue(
                    f"Check failed: {check_name}",
                    f"Diagnostic check encountered an error: {str(error)}",
                    "critical",
                    {"traceback": traceback.format_exc()}
                )
                if self.debug_mode:
                    print(f"  Failed: {check_name} - {error}")

        if self.auto_fix:
            self._apply_automatic_fixes()

        self.results['summary'] = self._generate_summary()

        if self.logger:
            self.logger.info(f"Diagnostic completed with {len(self.results['issues'])} issues")

        return self.results

    def check_system(self):
        """Validate system hardware and operating system requirements."""
        if self.debug_mode:
            print("  Checking system requirements...")

        system_info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
        }

        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                system_info.update({
                    'memory_total_gb': round(memory.total / 1024**3, 2),
                    'memory_available_gb': round(memory.available / 1024**3, 2),
                    'memory_percent_used': memory.percent,
                })

                disk = psutil.disk_usage('/')
                system_info.update({
                    'disk_total_gb': round(disk.total / 1024**3, 2),
                    'disk_free_gb': round(disk.free / 1024**3, 2),
                    'disk_percent_used': disk.percent,
                })

                system_info.update({
                    'cpu_count_physical': psutil.cpu_count(logical=False),
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                })

                if system_info.get('memory_total_gb', 0) < 8:
                    self._add_issue(
                        "Insufficient Memory",
                        f"System has {system_info['memory_total_gb']:.1f}GB RAM. 8GB minimum recommended.",
                        "warning",
                        {"actual": system_info['memory_total_gb'], "recommended": 8}
                    )

                if system_info.get('disk_free_gb', 0) < 10:
                    self._add_issue(
                        "Low Disk Space",
                        f"Only {system_info['disk_free_gb']:.1f}GB free space available. 10GB minimum recommended.",
                        "warning",
                        {"actual": system_info['disk_free_gb'], "recommended": 10}
                    )

            except Exception as error:
                self._add_issue(
                    "System Check Failed",
                    f"Error collecting system information: {str(error)}",
                    "warning",
                    {"error": str(error)}
                )
        else:
            self._add_issue(
                "Missing psutil",
                "psutil package not installed. System resource monitoring unavailable.",
                "info",
                {"recommendation": "pip install psutil"}
            )

        python_version = system_info['python_version']
        try:
            version_parts = tuple(map(int, python_version.split('.')[:2]))
            if version_parts < (3, 10):
                self._add_issue(
                    "Outdated Python Version",
                    f"Python {python_version} detected. Version 3.10 or higher recommended.",
                    "warning",
                    {"actual": python_version, "recommended": "3.10+"}
                )
        except (ValueError, AttributeError):
            pass

        self.results['system'] = system_info

    def check_python_environment(self):
        """Verify Python environment and virtual environment status."""
        if self.debug_mode:
            print("  Checking Python environment...")

        env_info = {
            'virtual_env_active': os.getenv('VIRTUAL_ENV') is not None,
            'conda_env_active': os.getenv('CONDA_PREFIX') is not None,
            'python_executable': sys.executable,
            'python_prefix': sys.prefix,
            'working_directory': str(Path.cwd()),
            'project_root': str(PROJECT_ROOT)
        }

        if env_info['virtual_env_active']:
            env_info['virtual_env_path'] = os.getenv('VIRTUAL_ENV')

        if env_info['conda_env_active']:
            env_info['conda_env_path'] = os.getenv('CONDA_PREFIX')

        if not env_info['virtual_env_active'] and not env_info['conda_env_active']:
            self._add_issue(
                "No Virtual Environment",
                "Python is not running in a virtual environment. This may cause dependency conflicts.",
                "info",
                {"recommendation": "Create and activate a virtual environment (venv or conda)"}
            )

        self.results['system']['environment'] = env_info

    def check_project_structure(self):
        """Validate project directory and file structure."""
        if self.debug_mode:
            print("  Checking project structure...")

        required_directories = [
            'src',
            'src/core',
            'src/utilities',
            'data',
            'tests',
            'tests/unit',
            'scripts',
        ]

        required_files = [
            'src/core/__init__.py',
            'src/core/app.py',
            'src/core/config.py',
            'src/core/constants.py',
            'src/core/exceptions.py',
            'src/core/logger.py',
            'src/utilities/__init__.py',
            'src/utilities/cleanup.py',
            'src/utilities/formatter.py',
            'src/utilities/helpers.py',
            'src/utilities/logger.py',
            'src/utilities/monitor.py',
            'src/utilities/retry.py',
            'src/utilities/task_queue.py',
            'src/utilities/validator.py',
            'requirements.txt',
            'README.md',
            '.gitignore'
        ]

        structure = {
            'directories': {},
            'files': {},
            'project_root': str(PROJECT_ROOT),
            'project_root_exists': PROJECT_ROOT.exists()
        }

        for directory_path in required_directories:
            full_path = PROJECT_ROOT / directory_path
            exists = full_path.exists() and full_path.is_dir()
            structure['directories'][directory_path] = exists

            if not exists:
                self._add_issue(
                    "Missing Directory",
                    f"Required directory not found: {directory_path}",
                    "info" if directory_path.startswith('tests/') else "warning",
                    {"path": str(full_path)}
                )

        for file_path in required_files:
            full_path = PROJECT_ROOT / file_path
            exists = full_path.exists() and full_path.is_file()
            structure['files'][file_path] = exists

            if not exists:
                severity = "info" if file_path in ['README.md', '.gitignore'] else "warning"
                self._add_issue(
                    "Missing File",
                    f"Required file not found: {file_path}",
                    severity,
                    {"path": str(full_path)}
                )
            elif full_path.stat().st_size == 0:
                self._add_issue(
                    "Empty File",
                    f"File exists but is empty: {file_path}",
                    "info",
                    {"path": str(full_path)}
                )

        self.results['installation']['structure'] = structure

    def check_dependencies(self):
        """Validate installed Python packages and versions."""
        if self.debug_mode:
            print("  Checking dependencies...")

        dependencies = [
            ('PyPDF2', 'PyPDF2', '3.0.0'),
            ('pdfplumber', 'pdfplumber', '0.10.0'),
            ('python-docx', 'docx', '0.8.11'),
            ('sqlalchemy', 'sqlalchemy', '2.0.0'),
            ('chromadb', 'chromadb', '0.4.0'),
            ('sentence-transformers', 'sentence_transformers', '2.2.0'),
            ('transformers', 'transformers', '4.30.0'),
            ('torch', 'torch', '2.0.0'),
            ('ollama', 'ollama', '0.1.0'),
            ('customtkinter', 'customtkinter', '5.2.0'),
            ('pytest', 'pytest', '7.4.0'),
            ('PyYAML', 'yaml', '6.0'),
            ('psutil', 'psutil', '5.9.0'),
            ('pytesseract', 'pytesseract', '0.3.10'),
            ('Pillow', 'PIL', '10.0.0'),
        ]

        installed = {}
        missing_critical = []
        missing_optional = []

        critical_packages = ['chromadb', 'sentence-transformers', 'ollama']

        for package_name, import_name, min_version in dependencies:
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')

                installed[package_name] = {
                    'installed': True,
                    'version': version,
                    'min_version': min_version,
                    'meets_requirement': self._compare_versions(version, min_version)
                }

                if not installed[package_name]['meets_requirement'] and version != 'unknown':
                    self._add_issue(
                        "Outdated Dependency",
                        f"{package_name} version {version} installed. Version {min_version} or higher recommended.",
                        "info",
                        {"package": package_name, "actual": version, "recommended": min_version}
                    )

            except ImportError:
                installed[package_name] = {
                    'installed': False,
                    'version': None,
                    'min_version': min_version,
                    'meets_requirement': False
                }

                if package_name in critical_packages:
                    missing_critical.append(package_name)
                    self._add_issue(
                        "Missing Critical Dependency",
                        f"Required package not installed: {package_name}",
                        "warning",
                        {"package": package_name, "required_version": min_version}
                    )
                else:
                    missing_optional.append(package_name)
                    self._add_issue(
                        "Missing Optional Dependency",
                        f"Optional package not installed: {package_name}",
                        "info",
                        {"package": package_name, "recommended_version": min_version}
                    )

        self.results['dependencies'] = {
            'packages': installed,
            'missing_critical': missing_critical,
            'missing_optional': missing_optional,
            'total_checked': len(dependencies),
            'total_installed': len([p for p in installed.values() if p['installed']])
        }

    def check_data_directories(self):
        """Validate data directories existence and permissions in both user home and project."""
        if self.debug_mode:
            print("  Checking data directories...")

        user_data_root = Path.home() / ".cockatoo_v1"
        project_data_root = PROJECT_ROOT / "data"

        data_directories = [
            (user_data_root, "user_root"),
            (user_data_root / "models", "user_models"),
            (user_data_root / "documents", "user_documents"),
            (user_data_root / "database", "user_database"),
            (user_data_root / "logs", "user_logs"),
            (user_data_root / "config", "user_config"),
            (user_data_root / "uploads", "user_uploads"),
            (user_data_root / "processed", "user_processed"),
            (user_data_root / "exports", "user_exports"),
            (user_data_root / "cache", "user_cache"),
            (user_data_root / "backups", "user_backups"),
            (user_data_root / "temp", "user_temp"),
            
            (project_data_root, "project_root"),
            (project_data_root / "models", "project_models"),
            (project_data_root / "models" / "sentence-transformers", "project_models_sentence"),
            (project_data_root / "models" / "nltk_data", "project_models_nltk"),
            (project_data_root / "models" / "ocr_tessdata", "project_models_ocr"),
            (project_data_root / "database", "project_database"),
            (project_data_root / "database" / "chroma", "project_database_chroma"),
            (project_data_root / "documents", "project_documents"),
            (project_data_root / "documents" / "uploads", "project_documents_uploads"),
            (project_data_root / "documents" / "processed", "project_documents_processed"),
            (project_data_root / "documents" / "thumbnails", "project_documents_thumbnails"),
            (project_data_root / "documents" / "exports", "project_documents_exports"),
            (project_data_root / "config", "project_config"),
            (project_data_root / "logs", "project_logs"),
        ]

        directory_info = {}
        critical_directories = ['models', 'database', 'config']

        for directory, key in data_directories:
            if not self._should_check_location(key):
                continue

            info = {
                'exists': directory.exists(),
                'is_directory': directory.exists() and directory.is_dir(),
                'writable': False,
                'size_bytes': 0,
                'file_count': 0,
                'location': 'user' if key.startswith('user_') else 'project',
                'path': str(directory)
            }

            if directory.exists():
                try:
                    test_file = directory / ".write_test"
                    test_file.write_text("test")
                    test_file.unlink()
                    info['writable'] = True
                except (PermissionError, OSError):
                    is_critical = any(directory.name == d for d in critical_directories)
                    if info['location'] == 'user' and is_critical:
                        self._add_issue(
                            "Directory Not Writable",
                            f"Cannot write to user directory: {directory}",
                            "warning" if is_critical else "info",
                            {"path": str(directory), "location": "user"}
                        )

                if directory.is_dir():
                    try:
                        total_size = 0
                        file_count = 0
                        for item in directory.rglob("*"):
                            if item.is_file():
                                total_size += item.stat().st_size
                                file_count += 1
                        info['size_bytes'] = total_size
                        info['file_count'] = file_count
                    except (PermissionError, OSError):
                        pass
            else:
                if info['location'] == 'user':
                    is_critical = any(directory.name == d for d in critical_directories)
                    if is_critical:
                        self._add_issue(
                            "Missing User Data Directory",
                            f"Required user data directory not found: {directory}",
                            "warning",
                            {"path": str(directory), "location": "user"}
                        )
                    else:
                        self._add_issue(
                            "Missing User Directory",
                            f"Optional user directory not found: {directory}",
                            "info",
                            {"path": str(directory), "location": "user"}
                        )
                else:
                    if directory.name in critical_directories or directory.parent.name in critical_directories:
                        self._add_issue(
                            "Missing Project Directory",
                            f"Project directory not found: {directory.relative_to(PROJECT_ROOT)}",
                            "warning",
                            {"path": str(directory), "location": "project"}
                        )

            directory_info[key] = info

        self.results['data']['directories'] = directory_info

    def check_databases(self):
        """Verify database files and integrity in both user and project locations."""
        if self.debug_mode:
            print("  Checking databases...")

        database_files = [
            (Path.home() / ".cockatoo_v1" / "database" / "cockatoo.db", "user_cockatoo_db"),
            (Path.home() / ".cockatoo_v1" / "database" / "chroma" / "chroma.sqlite3", "user_chroma_db"),
            (PROJECT_ROOT / "data" / "database" / "sqlite.db", "project_sqlite_db"),
            (PROJECT_ROOT / "data" / "database" / "cache.db", "project_cache_db"),
        ]

        database_info = {}

        for db_file, key in database_files:
            if not self._should_check_location(key):
                continue

            info = {
                'exists': db_file.exists(),
                'size_bytes': db_file.stat().st_size if db_file.exists() else 0,
                'accessible': False,
                'table_count': 0,
                'tables': [],
                'integrity_check': None,
                'location': 'user' if key.startswith('user_') else 'project',
                'path': str(db_file)
            }

            if db_file.exists():
                try:
                    connection = sqlite3.connect(db_file)
                    cursor = connection.cursor()

                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    info['tables'] = tables
                    info['table_count'] = len(tables)

                    cursor.execute("PRAGMA integrity_check")
                    integrity_result = cursor.fetchall()
                    info['integrity_check'] = integrity_result[0][0] if integrity_result else "unknown"

                    connection.close()
                    info['accessible'] = True

                    if info['integrity_check'] != 'ok':
                        self._add_issue(
                            "Database Integrity Issue",
                            f"Database integrity check failed: {db_file.name}",
                            "warning",
                            {"database": str(db_file), "integrity_result": info['integrity_check']}
                        )

                except sqlite3.Error as error:
                    self._add_issue(
                        "Database Access Error",
                        f"Cannot access database {db_file.name}: {error}",
                        "warning",
                        {"database": str(db_file), "error": str(error)}
                    )
            else:
                if info['location'] == 'user':
                    if db_file.name == 'cockatoo.db':
                        self._add_issue(
                            "Missing Database",
                            f"Main database file not found: {db_file.name}",
                            "warning",
                            {"database": str(db_file)}
                        )
                    elif db_file.name == 'chroma.sqlite3':
                        self._add_issue(
                            "Missing Database",
                            f"Chroma database file not found: {db_file.name}",
                            "info",
                            {"database": str(db_file)}
                        )
                elif info['location'] == 'project':
                    if not db_file.exists():
                        self._add_issue(
                            "Missing Project Database",
                            f"Project database file not found: {db_file.name}",
                            "warning",
                            {"database": str(db_file), "location": "project"}
                        )

            database_info[key] = info

        self.results['data']['databases'] = database_info

    def check_configuration(self):
        """Validate configuration files in both user and project locations."""
        if self.debug_mode:
            print("  Checking configuration...")

        config_files = [
            (Path.home() / ".cockatoo_v1" / "config" / "app_config.yaml", "user_app_config"),
            (Path.home() / ".cockatoo_v1" / "config" / "llm_config.yaml", "user_llm_config"),
            (PROJECT_ROOT / "data" / "config" / "app_config.yaml", "project_app_config"),
            (PROJECT_ROOT / "data" / "config" / "llm_config.yaml", "project_llm_config"),
            (PROJECT_ROOT / "data" / "config" / "ui_config.yaml", "project_ui_config"),
            (PROJECT_ROOT / "data" / "config" / "shortcuts.json", "project_shortcuts"),
        ]

        config_info = {}

        for config_file, key in config_files:
            if not self._should_check_location(key):
                continue

            info = {
                'exists': config_file.exists(),
                'size_bytes': config_file.stat().st_size if config_file.exists() else 0,
                'readable': False,
                'valid_yaml': False,
                'has_content': False,
                'location': 'user' if key.startswith('user_') else 'project',
                'path': str(config_file)
            }

            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as file:
                        content = file.read()

                    info['readable'] = True
                    info['has_content'] = len(content.strip()) > 0

                    if YAML_AVAILABLE and config_file.suffix in ['.yaml', '.yml']:
                        try:
                            yaml.safe_load(content)
                            info['valid_yaml'] = True
                        except yaml.YAMLError as error:
                            self._add_issue(
                                "Invalid YAML",
                                f"Configuration file is not valid YAML: {config_file.name}",
                                "warning",
                                {"config_file": str(config_file), "error": str(error)}
                            )
                    elif config_file.suffix == '.json':
                        try:
                            json.loads(content)
                            info['valid_json'] = True
                        except json.JSONDecodeError as error:
                            self._add_issue(
                                "Invalid JSON",
                                f"Configuration file is not valid JSON: {config_file.name}",
                                "warning",
                                {"config_file": str(config_file), "error": str(error)}
                            )

                except Exception as error:
                    self._add_issue(
                        "Unreadable Configuration",
                        f"Cannot read configuration file: {config_file.name}",
                        "warning",
                        {"config_file": str(config_file), "error": str(error)}
                    )
            else:
                if info['location'] == 'user':
                    if config_file.name == 'app_config.yaml':
                        self._add_issue(
                            "Missing Configuration",
                            f"Main configuration file not found: {config_file.name}",
                            "warning",
                            {"config_file": str(config_file)}
                        )
                    elif config_file.name == 'llm_config.yaml':
                        self._add_issue(
                            "Missing Optional Configuration",
                            f"Optional configuration file not found: {config_file.name}",
                            "info",
                            {"config_file": str(config_file)}
                        )
                elif info['location'] == 'project':
                    if not config_file.exists():
                        self._add_issue(
                            "Missing Project Configuration",
                            f"Project configuration file not found: {config_file.name}",
                            "warning",
                            {"config_file": str(config_file), "location": "project"}
                        )

            config_info[key] = info

        self.results['configuration'] = config_info

    def check_permissions(self):
        """Check file and directory permissions for security."""
        if self.debug_mode:
            print("  Checking permissions...")

        critical_paths = [
            (Path.home() / ".cockatoo_v1" / "database" / "cockatoo.db", "user_cockatoo_db"),
            (Path.home() / ".cockatoo_v1" / "config" / "app_config.yaml", "user_app_config"),
            (PROJECT_ROOT / "data" / "database" / "sqlite.db", "project_sqlite_db"),
            (PROJECT_ROOT / "data" / "config" / "app_config.yaml", "project_app_config"),
        ]

        permission_info = {}

        for path, key in critical_paths:
            if not self._should_check_location(key):
                continue

            info = {
                'exists': path.exists(),
                'mode': None,
                'owner_readable': False,
                'owner_writable': False,
                'location': 'user' if key.startswith('user_') else 'project',
                'path': str(path)
            }

            if path.exists():
                try:
                    mode = path.stat().st_mode
                    info['mode'] = oct(mode)[-3:]
                    info['owner_readable'] = bool(mode & 0o400)
                    info['owner_writable'] = bool(mode & 0o200)
                except Exception as error:
                    self._add_issue(
                        "Permission Check Failed",
                        f"Cannot check permissions for: {path.name}",
                        "info",
                        {"file": str(path), "error": str(error)}
                    )

            permission_info[key] = info

        self.results['system']['permissions'] = permission_info

    def check_network_connectivity(self):
        """Test network connectivity to required services."""
        if self.debug_mode:
            print("  Checking network connectivity...")

        network_info = {
            'internet_accessible': False,
            'ollama_accessible': False,
        }

        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            network_info['internet_accessible'] = True
        except (socket.error, socket.timeout):
            self._add_issue(
                "No Internet Connectivity",
                "Cannot reach external network. Some features may be limited.",
                "info"
            )

        if REQUESTS_AVAILABLE:
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                network_info['ollama_accessible'] = response.status_code == 200

                if not network_info['ollama_accessible']:
                    self._add_issue(
                        "Ollama Not Running",
                        "Ollama service is not accessible. Local LLM features will not work.",
                        "info",
                        {"endpoint": "localhost:11434"}
                    )
            except Exception:
                network_info['ollama_accessible'] = False
        else:
            self._add_issue(
                "Missing requests",
                "requests package not installed. Cannot test Ollama connectivity.",
                "info",
                {"recommendation": "pip install requests"}
            )

        self.results['system']['network'] = network_info

    def check_performance(self):
        """Run basic performance benchmarks."""
        if self.debug_mode:
            print("  Running performance benchmarks...")

        performance_info = {
            'disk_speed_mbps': None,
            'memory_operation_time': None,
            'benchmarks': {}
        }

        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
                start_time = time.perf_counter()
                data = b'x' * (10 * 1024 * 1024)
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                end_time = time.perf_counter()

                elapsed = end_time - start_time
                speed_mbps = (10 / elapsed) if elapsed > 0 else 0

                performance_info['disk_speed_mbps'] = round(speed_mbps, 2)
                performance_info['benchmarks']['disk_write_10mb'] = {
                    'time_seconds': round(elapsed, 4),
                    'speed_mbps': round(speed_mbps, 2)
                }

                os.unlink(temp_file.name)

        except Exception as error:
            if self.debug_mode:
                print(f"  Disk benchmark failed: {error}")

        try:
            start_time = time.perf_counter()
            test_list = [i for i in range(1000000)]
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            performance_info['memory_operation_time'] = round(elapsed, 4)
            performance_info['benchmarks']['memory_1m_ints'] = {
                'time_seconds': round(elapsed, 4)
            }

        except Exception as error:
            if self.debug_mode:
                print(f"  Memory benchmark failed: {error}")

        self.results['performance'] = performance_info

    def check_models(self):
        """Check for downloaded AI models in both user and project locations."""
        if self.debug_mode:
            print("  Checking model availability...")

        models_locations = [
            (Path.home() / ".cockatoo_v1" / "models", "user_models"),
            (PROJECT_ROOT / "data" / "models", "project_models"),
        ]

        model_info = {}

        for models_directory, key in models_locations:
            if not self._should_check_location(key):
                continue

            info = {
                'directory_exists': models_directory.exists(),
                'models_found': [],
                'total_size_gb': 0,
                'location': 'user' if key.startswith('user_') else 'project',
                'path': str(models_directory)
            }

            if models_directory.exists():
                try:
                    model_extensions = ['.bin', '.pth', '.pt', '.safetensors', '.gguf', '.onnx']

                    for model_file in models_directory.rglob("*"):
                        if model_file.is_file() and model_file.suffix in model_extensions:
                            info['models_found'].append({
                                'name': model_file.name,
                                'path': str(model_file),
                                'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2)
                            })

                    total_size = sum(m['size_mb'] for m in info['models_found'])
                    info['total_size_gb'] = round(total_size / 1024, 2)

                    if not info['models_found']:
                        if info['location'] == 'user':
                            self._add_issue(
                                "No Models Found",
                                "No AI models found in user models directory. Download models to enable AI features.",
                                "info",
                                {"models_directory": str(models_directory)}
                            )
                        elif info['location'] == 'project':
                            self._add_issue(
                                "No Project Models Found",
                                "No AI models found in project models directory",
                                "info",
                                {"models_directory": str(models_directory), "location": "project"}
                            )

                except Exception as error:
                    self._add_issue(
                        "Model Directory Error",
                        f"Cannot read models directory: {error}",
                        "info",
                        {"models_directory": str(models_directory)}
                    )
            else:
                if info['location'] == 'user':
                    self._add_issue(
                        "Missing Models Directory",
                        "User models directory does not exist. AI features will not work.",
                        "info",
                        {"models_directory": str(models_directory)}
                    )
                elif info['location'] == 'project':
                    self._add_issue(
                        "Missing Project Models Directory",
                        "Project models directory does not exist",
                        "warning",
                        {"models_directory": str(models_directory), "location": "project"}
                    )

            model_info[key] = info

        self.results['data']['models'] = model_info

    def check_integration(self):
        """Run basic integration tests for core modules."""
        if self.debug_mode:
            print("  Running integration tests...")

        integration_info = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'results': []
        }

        test_cases = [
            ("Core Config Import", "from src.core.config import AppConfig"),
            ("Logger Import", "from src.core.logger import get_logger"),
            ("Exceptions Import", "from src.core.exceptions import CockatooError"),
            ("App Core Import", "from src.core.app import DocumentProcessor")
        ]

        for test_name, import_statement in test_cases:
            try:
                exec(import_statement)
                integration_info['tests_run'] += 1
                integration_info['tests_passed'] += 1
                integration_info['results'].append({
                    'test': test_name,
                    'passed': True,
                    'error': None
                })
            except Exception as error:
                integration_info['tests_run'] += 1
                integration_info['tests_failed'] += 1
                integration_info['results'].append({
                    'test': test_name,
                    'passed': False,
                    'error': str(error)
                })

                self._add_issue(
                    "Import Test Failed",
                    f"Failed to import module: {test_name}",
                    "info",
                    {"test": test_name, "import": import_statement, "error": str(error)}
                )

        self.results['installation']['integration_tests'] = integration_info

    def _add_issue(self, title: str, description: str, severity: str = "warning",
                   details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a diagnostic issue with structured information.

        Args:
            title: Brief issue identifier
            description: Detailed explanation of the issue
            severity: Issue severity (critical, warning, info)
            details: Additional contextual information
        """
        issue = {
            'id': f"ISSUE_{len(self.results['issues']) + 1:04d}",
            'title': title,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'details': details or {},
            'auto_fixable': self._is_auto_fixable(title)
        }

        self.results['issues'].append(issue)

        if self.debug_mode:
            severity_prefix = {
                'critical': 'CRITICAL',
                'warning': 'WARNING',
                'info': 'INFO'
            }.get(severity, 'UNKNOWN')
            print(f"  [{severity_prefix}] {title}")

    def _is_auto_fixable(self, title: str) -> bool:
        """
        Determine if an issue can be automatically fixed.

        Args:
            title: Issue title to evaluate

        Returns:
            True if auto-fix is available, False otherwise
        """
        fixable_titles = [
            "Missing Directory",
            "Missing Optional Configuration",
            "Empty File",
            "Missing Optional Dependency"
        ]
        return title in fixable_titles

    def _apply_automatic_fixes(self) -> None:
        """Attempt to automatically resolve fixable issues."""
        print("\nApplying automatic fixes...")

        for issue in self.results['issues']:
            if issue.get('auto_fixable', False):
                success, message = self._fix_issue(issue)

                if success:
                    self.fixes_applied.append({
                        'issue_id': issue['id'],
                        'title': issue['title'],
                        'message': message,
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"  Fixed: {issue['title']} - {message}")
                else:
                    self.fixes_failed.append({
                        'issue_id': issue['id'],
                        'title': issue['title'],
                        'message': message,
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"  Failed: {issue['title']} - {message}")

        self.results['fixes_applied'] = self.fixes_applied
        self.results['fixes_failed'] = self.fixes_failed

    def _fix_issue(self, issue: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Execute fix for a specific issue.

        Args:
            issue: Issue dictionary with details

        Returns:
            Tuple of (success: bool, message: str)
        """
        title = issue['title']
        details = issue.get('details', {})

        try:
            if title == "Missing Directory":
                path = Path(details.get('path', ''))
                if path and not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    return True, f"Created directory: {path}"

            elif title == "Missing Optional Configuration":
                path = Path(details.get('path', ''))
                if path and not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# Auto-generated configuration file\n")
                    return True, f"Created configuration file: {path.name}"

            elif title == "Empty File":
                path = Path(details.get('path', ''))
                if path and path.exists() and path.stat().st_size == 0:
                    extension_map = {
                        '.py': '# Empty Python file\n',
                        '.yaml': '# Empty configuration\n',
                        '.yml': '# Empty configuration\n',
                        '.md': '# Empty documentation\n',
                        '.json': '{}\n'
                    }
                    content = extension_map.get(path.suffix, '')
                    path.write_text(content)
                    return True, f"Added placeholder content to: {path.name}"

            elif title == "Missing Optional Dependency":
                package = details.get('package', '')
                if package:
                    try:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", package],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        return True, f"Installed package: {package}"
                    except subprocess.CalledProcessError:
                        return False, f"Failed to install package: {package}"

        except Exception as error:
            return False, f"Fix failed: {error}"

        return False, "No fix available for this issue type"

    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate diagnostic summary with health assessment.

        Returns:
            Dictionary with summary statistics and recommendations
        """
        issues = self.results['issues']
        
        if self.mode != "both":
            filtered_issues = []
            for issue in issues:
                location = issue.get('details', {}).get('location')
                if location is None or location == self.mode:
                    filtered_issues.append(issue)
            issues = filtered_issues

        critical_count = len([i for i in issues if i.get('severity') == 'critical'])
        warning_count = len([i for i in issues if i.get('severity') == 'warning'])
        info_count = len([i for i in issues if i.get('severity') == 'info'])

        health_score = 100
        if critical_count > 0:
            health_score = max(0, 100 - (critical_count * 25))
        elif warning_count > 0:
            health_score = max(50, 100 - (warning_count * 5))

        if health_score >= 90:
            overall_status = "healthy"
        elif health_score >= 70:
            overall_status = "needs_attention"
        else:
            overall_status = "critical"

        recommendations = []

        if critical_count > 0:
            recommendations.append("Address critical issues immediately before using Cockatoo")

        if warning_count > 0:
            recommendations.append("Resolve warning issues for optimal performance")

        missing_deps = [i for i in issues if "Missing Critical Dependency" in i.get('title', '')]
        if missing_deps:
            dep_names = [d.get('details', {}).get('package', '') for d in missing_deps]
            dep_names = [d for d in dep_names if d]
            if dep_names:
                recommendations.append(f"Install missing dependencies: {', '.join(dep_names)}")

        return {
            'total_issues': len(issues),
            'critical_issues': critical_count,
            'warning_issues': warning_count,
            'info_issues': info_count,
            'fixes_applied': len(self.fixes_applied),
            'fixes_failed': len(self.fixes_failed),
            'overall_status': overall_status,
            'health_score': health_score,
            'recommendations': recommendations[:5]
        }

    def _compare_versions(self, version1: str, version2: str) -> bool:
        """
        Compare two version strings.

        Args:
            version1: Current version
            version2: Required minimum version

        Returns:
            True if version1 meets or exceeds version2
        """
        if version1 == 'unknown':
            return False

        try:
            v1_parts = [int(x) for x in version1.split('.')[:3]]
            v2_parts = [int(x) for x in version2.split('.')[:3]]

            while len(v1_parts) < 3:
                v1_parts.append(0)
            while len(v2_parts) < 3:
                v2_parts.append(0)

            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 > v2:
                    return True
                if v1 < v2:
                    return False
            return True
        except (ValueError, AttributeError):
            return False

    def print_report(self) -> None:
        """Display formatted diagnostic report to console."""
        print("\n" + "=" * 70)
        print("COCKATOO V1 DIAGNOSTIC REPORT")
        print("=" * 70)

        summary = self.results.get('summary', self._generate_summary())

        status_map = {
            'healthy': 'HEALTHY',
            'needs_attention': 'NEEDS ATTENTION',
            'critical': 'CRITICAL'
        }
        status_display = status_map.get(summary.get('overall_status', 'unknown'), 'UNKNOWN')

        mode_display = {
            "user": "User Home Directory Only",
            "project": "Project Directory Only",
            "both": "User and Project Directories"
        }
        
        print(f"Check Mode: {mode_display.get(self.mode, self.mode)}")
        print(f"Overall Status: {status_display}")
        print(f"Health Score: {summary.get('health_score', 0)}/100")
        print(f"Issues Found: {summary.get('total_issues', 0)} "
              f"(Critical: {summary.get('critical_issues', 0)}, "
              f"Warnings: {summary.get('warning_issues', 0)}, "
              f"Info: {summary.get('info_issues', 0)})")

        if self.fixes_applied or self.fixes_failed:
            print(f"Fixes Applied: {len(self.fixes_applied)}, Failed: {len(self.fixes_failed)}")

        if self.results['issues']:
            print("\nDETAILED ISSUES:")
            print("-" * 70)

            for severity in ['critical', 'warning', 'info']:
                severity_issues = [i for i in self.results['issues'] if i.get('severity') == severity]

                if severity_issues:
                    print(f"\n{severity.upper()} ISSUES ({len(severity_issues)}):")

                    for issue in severity_issues:
                        issue_id = issue.get('id', 'UNKNOWN')
                        issue_title = issue.get('title', 'Unknown')
                        location = issue.get('details', {}).get('location', '')
                        location_info = f" [{location}]" if location else ""
                        print(f"  [{issue_id}]{location_info} {issue_title}")

        print("\nSYSTEM INFORMATION:")
        print("-" * 70)
        system = self.results.get('system', {})

        print(f"  Platform: {system.get('platform', 'Unknown')}")
        print(f"  Python: {system.get('python_version', 'Unknown')}")

        memory_total = system.get('memory_total_gb', 0)
        memory_available = system.get('memory_available_gb', 0)
        if memory_total > 0:
            print(f"  Memory: {memory_total:.1f}GB total, {memory_available:.1f}GB available")

        disk_free = system.get('disk_free_gb', 0)
        disk_total = system.get('disk_total_gb', 0)
        if disk_total > 0:
            print(f"  Disk: {disk_free:.1f}GB free of {disk_total:.1f}GB total")

        print("\nRECOMMENDATIONS:")
        print("-" * 70)

        recommendations = summary.get('recommendations', [])
        if recommendations:
            for idx, recommendation in enumerate(recommendations, 1):
                print(f"  {idx}. {recommendation}")
        else:
            print("  No specific recommendations. System appears healthy.")

        print("\n" + "=" * 70)
        print(f"Diagnostic ID: {self.results.get('diagnostic_id', 'UNKNOWN')}")
        print(f"Generated: {self.results.get('timestamp', 'Unknown')}")
        print("=" * 70)

    def save_report(self, output_file: Path) -> None:
        """
        Save diagnostic report to JSON file.

        Args:
            output_file: Path where report will be saved
        """
        if 'summary' not in self.results or not self.results['summary']:
            self.results['summary'] = self._generate_summary()

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(self.results, file, indent=2, default=str)

        print(f"\nFull diagnostic report saved to: {output_file}")

    def validate_installation(self) -> bool:
        """
        Validate complete installation meets minimum requirements.

        Returns:
            True if installation is valid, False otherwise
        """
        summary = self._generate_summary()

        if summary.get('critical_issues', 0) > 0:
            return False

        python_version = self.results.get('system', {}).get('python_version', '0.0')
        try:
            version_parts = tuple(map(int, python_version.split('.')[:2]))
            if version_parts < (3, 10):
                return False
        except (ValueError, AttributeError):
            return False

        return True


def main() -> None:
    """Main entry point for diagnostic tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cockatoo V1 Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable verbose debug output")
    parser.add_argument("--auto-fix", "-a", action="store_true",
                       help="Attempt to automatically fix detected issues")
    parser.add_argument("--output", "-o", type=Path,
                       help="Save detailed report to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimize console output")
    parser.add_argument("--mode", "-m", choices=["user", "project", "both"],
                       default="project",
                       help="Check mode: user, project, or both (default: project)")

    arguments = parser.parse_args()

    try:
        diagnostic = CockatooDiagnostic(
            debug_mode=arguments.debug,
            auto_fix=arguments.auto_fix,
            mode=arguments.mode
        )

        diagnostic.run_all_checks()

        if not arguments.quiet:
            diagnostic.print_report()

        if arguments.output:
            diagnostic.save_report(arguments.output)

        if not diagnostic.validate_installation() and not arguments.quiet:
            print("\nWARNING: Installation does not meet minimum requirements.")
            print("Please address the issues above before using Cockatoo.")
            
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)
    except Exception as error:
        print(f"Unexpected error: {error}")
        if arguments.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()