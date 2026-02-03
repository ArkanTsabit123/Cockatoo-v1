# cockatoo_v1/src/core/config.py

"""
cockatoo_v1 Configuration Management
"""

import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from dataclasses import dataclass, field
from enum import Enum


class PlatformType(Enum):
    """Supported operating system platforms."""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


class LLMModel(Enum):
    """Supported LLM models."""
    LLAMA2_7B = "llama2:7b"
    MISTRAL_7B = "mistral:7b"
    NEURAL_CHAT_7B = "neural-chat:7b"


class EmbeddingModel(Enum):
    """Supported embedding models."""
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"
    MULTILINGUAL_MINILM_L12_V2 = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class AppConfig:
    """Application configuration with platform-specific paths."""
    
    # Application info
    app_name: str = "cockatoo_v1"
    app_version: str = "1.0.0"
    
    # Platform detection
    platform: PlatformType = field(init=False)
    
    # Paths (will be set in __post_init__)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    documents_dir: Path = field(init=False)
    database_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    exports_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    
    # Document processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 100
    supported_formats: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".txt", ".epub", ".md", 
        ".html", ".jpg", ".png", ".csv"
    ])
    ocr_enabled: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["eng", "ind"])
    
    # LLM Configuration
    llm_provider: str = "ollama"
    llm_model: str = "llama2:7b"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_max_tokens: int = 1024
    llm_context_window: int = 4096
    
    # Supported LLM models for model switching
    supported_llm_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "llama2:7b": {
            "description": "Llama 2 7B - Balanced performance",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1024,
            "requirements": {"ram": "8GB", "storage": "4.2GB"}
        },
        "mistral:7b": {
            "description": "Mistral 7B - Fast inference",
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 2048,
            "requirements": {"ram": "8GB", "storage": "4.1GB"}
        },
        "neural-chat:7b": {
            "description": "Neural Chat 7B - Conversation optimized",
            "temperature": 0.05,
            "top_p": 0.9,
            "max_tokens": 1024,
            "requirements": {"ram": "8GB", "storage": "4.3GB"}
        }
    })
    
    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_device: str = "cpu"
    embedding_cache_enabled: bool = True
    embedding_cache_size_mb: int = 500
    
    # Supported embedding models
    supported_embedding_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "context_length": 256,
            "description": "Fast and lightweight",
            "size": "90MB",
            "speed": "Fast",
            "accuracy": "Good"
        },
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "context_length": 384,
            "description": "High quality embeddings",
            "size": "420MB",
            "speed": "Medium",
            "accuracy": "Excellent"
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "dimensions": 384,
            "context_length": 128,
            "description": "Multilingual support",
            "size": "480MB",
            "speed": "Medium",
            "accuracy": "Good",
            "languages": ["en", "id", "es", "fr", "de", "zh"]
        }
    })
    
    # RAG Configuration
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    enable_hybrid_search: bool = True
    
    # UI Configuration
    ui_theme: str = "dark"
    ui_language: str = "en"
    ui_font_size: int = 12
    enable_animations: bool = True
    auto_save_interval: int = 60
    
    # Storage Configuration
    max_documents: int = 10000
    auto_cleanup_days: int = 90
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    encryption_enabled: bool = False
    
    # Performance Configuration
    max_workers: int = 4
    cache_enabled: bool = True
    cache_size_mb: int = 500
    enable_monitoring: bool = True
    
    # Privacy Configuration
    telemetry: bool = False
    auto_update_check: bool = False
    crash_reports: bool = False
    
    def __post_init__(self):
        """Initialize platform detection and directories after dataclass creation."""
        self.platform = self._detect_platform()
        self.data_dir = self.get_data_dir()
        
        self.models_dir = self.data_dir / "models"
        self.documents_dir = self.data_dir / "documents"
        self.database_dir = self.data_dir / "database"
        self.logs_dir = self.data_dir / "logs"
        self.exports_dir = self.data_dir / "exports"
        self.config_dir = self.data_dir / "config"
        self.cache_dir = self.data_dir / "cache"
        
        self.ensure_directories()
    
    def _detect_platform(self) -> PlatformType:
        """Detect the current operating system platform."""
        system = platform.system().lower()
        
        if system.startswith('win'):
            return PlatformType.WINDOWS
        elif system.startswith('darwin'):
            return PlatformType.MACOS
        elif system.startswith('linux'):
            return PlatformType.LINUX
        else:
            return PlatformType.UNKNOWN
    
    def get_data_dir(self) -> Path:
        """Get platform-specific data directory."""
        home = Path.home()
        
        if self.platform == PlatformType.WINDOWS:
            appdata = os.environ.get('APPDATA', '')
            if appdata:
                return Path(appdata) / "cockatoo_v1"
            else:
                return home / "AppData" / "Roaming" / "cockatoo_v1"
        
        elif self.platform == PlatformType.MACOS:
            return home / "Library" / "Application Support" / "cockatoo_v1"
        
        elif self.platform == PlatformType.LINUX:
            xdg_data_home = os.environ.get('XDG_DATA_HOME', '')
            if xdg_data_home:
                return Path(xdg_data_home) / "cockatoo_v1"
            else:
                return home / ".local" / "share" / "cockatoo_v1"
        
        else:
            return home / ".cockatoo_v1"
    
    def setup_crossplatform_paths(self) -> Path:
        """Setup cross-platform paths for checker compatibility."""
        system = platform.system().lower()
        
        if system == "windows":
            base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        elif system == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path.home() / ".local" / "share"
        
        app_dir = base / "cockatoo_v1"
        
        self.data_dir = app_dir
        self.models_dir = app_dir / "models"
        self.documents_dir = app_dir / "documents"
        self.database_dir = app_dir / "database"
        self.logs_dir = app_dir / "logs"
        self.exports_dir = app_dir / "exports"
        self.config_dir = app_dir / "config"
        self.cache_dir = app_dir / "cache"
        
        return app_dir
    
    def ensure_directories(self) -> None:
        """Create all required application directories."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.documents_dir,
            self.database_dir,
            self.logs_dir,
            self.exports_dir,
            self.config_dir,
            self.cache_dir,
            self.models_dir / "sentence-transformers",
            self.models_dir / "nltk_data",
            self.models_dir / "ocr_tessdata",
            self.documents_dir / "uploads",
            self.documents_dir / "processed",
            self.documents_dir / "thumbnails",
            self.database_dir / "chroma",
            self.cache_dir / "embeddings",
            self.cache_dir / "responses"
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for the current model."""
        model_config = self.supported_llm_models.get(self.llm_model, {})
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "description": model_config.get("description", ""),
            "temperature": self.llm_temperature,
            "top_p": self.llm_top_p,
            "max_tokens": self.llm_max_tokens,
            "context_window": self.llm_context_window,
            "requirements": model_config.get("requirements", {})
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration for the current model."""
        model_config = self.supported_embedding_models.get(self.embedding_model, {})
        return {
            "model": self.embedding_model,
            "dimensions": self.embedding_dimensions,
            "description": model_config.get("description", ""),
            "device": self.embedding_device,
            "cache_enabled": self.embedding_cache_enabled,
            "cache_size_mb": self.embedding_cache_size_mb,
            "size": model_config.get("size", ""),
            "speed": model_config.get("speed", ""),
            "accuracy": model_config.get("accuracy", "")
        }
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration values."""
        issues = []
        
        required_dirs = [
            self.data_dir,
            self.models_dir,
            self.documents_dir,
            self.database_dir,
            self.logs_dir
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                issues.append(f"Missing directory: {directory}")
        
        if self.chunk_size < 100 or self.chunk_size > 2000:
            issues.append(f"Invalid chunk_size: {self.chunk_size} (should be 100-2000)")
        
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            issues.append(f"Invalid chunk_overlap: {self.chunk_overlap} (should be 0 to {self.chunk_size-1})")
        
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            issues.append(f"Invalid llm_temperature: {self.llm_temperature} (should be 0.0-2.0)")
        
        if self.llm_model not in self.supported_llm_models:
            issues.append(f"Unsupported LLM model: {self.llm_model}")
        
        if self.embedding_model not in self.supported_embedding_models:
            issues.append(f"Unsupported embedding model: {self.embedding_model}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'platform': self.platform.value,
            'data_dir': str(self.data_dir)
        }


class ConfigManager:
    """Manager for loading and saving configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        if not self.config_path:
            temp_config = AppConfig()
            self.config_path = temp_config.config_dir / "app_config.yaml"
        
        self.config = AppConfig()
        self.config.ensure_directories()
    
    def load(self) -> AppConfig:
        """Load configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                
                app_config = yaml_config.get('app', {})
                
                for key, value in app_config.items():
                    if hasattr(self.config, key):
                        if isinstance(value, dict) and key in ['supported_llm_models', 'supported_embedding_models']:
                            current_value = getattr(self.config, key, {})
                            current_value.update(value)
                            setattr(self.config, key, current_value)
                        else:
                            setattr(self.config, key, value)
            
            return self.config
            
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            return self.config
    
    def save(self) -> bool:
        """Save configuration to YAML file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = {
                'app': {
                    'name': self.config.app_name,
                    'version': self.config.app_version,
                    'platform': self.config.platform.value,
                    'paths': {
                        'data_dir': str(self.config.data_dir),
                        'models_dir': str(self.config.models_dir),
                        'documents_dir': str(self.config.documents_dir),
                        'database_dir': str(self.config.database_dir),
                        'logs_dir': str(self.config.logs_dir),
                        'exports_dir': str(self.config.exports_dir),
                        'config_dir': str(self.config.config_dir),
                        'cache_dir': str(self.config.cache_dir)
                    },
                    'document_processing': {
                        'chunk_size': self.config.chunk_size,
                        'chunk_overlap': self.config.chunk_overlap,
                        'max_file_size_mb': self.config.max_file_size_mb,
                        'supported_formats': self.config.supported_formats,
                        'ocr_enabled': self.config.ocr_enabled,
                        'ocr_languages': self.config.ocr_languages
                    },
                    'ai': {
                        'llm': {
                            'provider': self.config.llm_provider,
                            'model': self.config.llm_model,
                            'temperature': self.config.llm_temperature,
                            'top_p': self.config.llm_top_p,
                            'max_tokens': self.config.llm_max_tokens,
                            'context_window': self.config.llm_context_window
                        },
                        'embeddings': {
                            'model': self.config.embedding_model,
                            'dimensions': self.config.embedding_dimensions,
                            'device': self.config.embedding_device,
                            'cache_enabled': self.config.embedding_cache_enabled,
                            'cache_size_mb': self.config.embedding_cache_size_mb
                        },
                        'rag': {
                            'top_k': self.config.rag_top_k,
                            'similarity_threshold': self.config.rag_similarity_threshold,
                            'enable_hybrid_search': self.config.enable_hybrid_search
                        }
                    },
                    'ui': {
                        'theme': self.config.ui_theme,
                        'language': self.config.ui_language,
                        'font_size': self.config.ui_font_size,
                        'enable_animations': self.config.enable_animations,
                        'auto_save_interval': self.config.auto_save_interval
                    },
                    'storage': {
                        'max_documents': self.config.max_documents,
                        'auto_cleanup_days': self.config.auto_cleanup_days,
                        'backup_enabled': self.config.backup_enabled,
                        'backup_interval_hours': self.config.backup_interval_hours,
                        'encryption_enabled': self.config.encryption_enabled
                    },
                    'performance': {
                        'max_workers': self.config.max_workers,
                        'cache_enabled': self.config.cache_enabled,
                        'cache_size_mb': self.config.cache_size_mb,
                        'enable_monitoring': self.config.enable_monitoring
                    },
                    'privacy': {
                        'telemetry': self.config.telemetry,
                        'auto_update_check': self.config.auto_update_check,
                        'crash_reports': self.config.crash_reports
                    }
                }
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
            
            print(f"Configuration saved to: {self.config_path}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False


default_config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return default_config_manager.load()


def init_config() -> AppConfig:
    """Initialize configuration and ensure directories exist."""
    config = get_config()
    config.ensure_directories()
    default_config_manager.save()
    return config

def setup_crossplatform_environment():
    """Setup complete cross-platform environment for cockatoo_v1."""
    from src.core.config import setup_crossplatform_dirs
    return setup_crossplatform_dirs()

def setup_crossplatform_dirs() -> Dict[str, Path]:
    """
    Setup cross-platform data directories.
    Returns dictionary of all paths created.
    """
    import platform
    from datetime import datetime
    
    system = platform.system().lower()
    
    if system == "windows":
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        app_dir = base / "cockatoo_v1"
    elif system == "darwin":
        app_dir = Path.home() / "Library" / "Application Support" / "cockatoo_v1"
    elif system == "linux":
        xdg_data_home = os.environ.get('XDG_DATA_HOME', '')
        if xdg_data_home:
            app_dir = Path(xdg_data_home) / "cockatoo_v1"
        else:
            app_dir = Path.home() / ".local" / "share" / "cockatoo_v1"
    else:
        app_dir = Path.home() / ".cockatoo_v1"
    
    directories = {
        'data': app_dir,
        'models': app_dir / "models",
        'documents': app_dir / "documents",
        'database': app_dir / "database",
        'logs': app_dir / "logs",
        'exports': app_dir / "exports",
        'config': app_dir / "config",
        'cache': app_dir / "cache",
        'temp': app_dir / "temp",
        'backups': app_dir / "backups"
    }
    
    print(f"Setting up cross-platform directories on {system}...")
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            marker = path / ".dir_created"
            if not marker.exists():
                marker.write_text(f"Created by cockatoo_v1 on {platform.system()} at {datetime.now()}")
            print(f"Created directory: {path}")
        except Exception as e:
            print(f"Failed to create directory {path}: {e}")
    
    return directories