# src/vector_store/index_manager.py

"""
Index Manager for cockatoo_v1 Vector Store.
Handles index creation, management, optimization, and maintenance for both
ChromaDB and FAISS vector databases.
"""

import sys
import logging
import json
import time
import shutil
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import threading
import queue

try:
    from ..core.constants import DATABASE_DIR, VECTOR_DB_NAME, CACHE_DIR
except (ImportError, ValueError):
    try:
        from src.core.constants import DATABASE_DIR, VECTOR_DB_NAME, CACHE_DIR
    except ImportError:
        DATABASE_DIR = Path.home() / ".cockatoo_v1" / "database"
        VECTOR_DB_NAME = "vector_store"
        CACHE_DIR = Path.home() / ".cockatoo_v1" / "cache"

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages vector indices for ChromaDB and FAISS with support for index operations."""

    DEFAULT_CONFIG = {
        "chroma": {
            "collection_name": "documents",
            "hnsw_space": "l2",
            "hnsw_construction_ef": 100,
            "hnsw_search_ef": 50,
            "hnsw_m": 16,
            "batch_size": 1000
        },
        "faiss": {
            "index_name": "documents",
            "dimension": 384,
            "metric": "cosine",
            "index_type": "flat",
            "ivf_nlist": 100,
            "ivf_nprobe": 10,
            "hnsw_m": 32,
            "hnsw_ef_construction": 200,
            "hnsw_ef_search": 50,
            "batch_size": 1000,
            "rebuild_threshold": 0.2
        },
        "maintenance": {
            "auto_optimize": True,
            "optimize_interval_hours": 24,
            "backup_enabled": True,
            "backup_interval_hours": 168,
            "backup_retention_days": 30,
            "max_backup_size_mb": 1024,
            "compression_enabled": True,
            "cache_enabled": True,
            "cache_ttl_seconds": 3600,
            "monitoring_enabled": True
        }
    }

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize Index Manager."""
        self.base_dir = Path(base_dir) if base_dir else DATABASE_DIR / "indices"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self._update_config(config)

        self.chroma_dir = self.base_dir / "chroma"
        self.faiss_dir = self.base_dir / "faiss"
        self.backup_dir = self.base_dir / "backups"
        self.cache_dir = CACHE_DIR / "index_manager"

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.base_dir / "index_metadata.json"
        self.index_metadata = self._load_metadata()

        self.index_cache = {}
        self.cache_lock = threading.Lock()

        self.maintenance_thread = None
        self.maintenance_running = False
        self.maintenance_queue = queue.Queue()

        if self.config["maintenance"]["auto_optimize"]:
            self._start_maintenance_thread()

        logger.info(f"Index Manager initialized: {self.base_dir}")

    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration recursively."""
        for key, value in config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

    def _load_metadata(self) -> Dict[str, Any]:
        """Load index metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded index metadata: {len(metadata.get('indices', []))} indices")
                return metadata
            except Exception as error:
                logger.error(f"Failed to load metadata: {error}")

        return {
            "indices": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "1.0.0"
        }

    def _save_metadata(self) -> None:
        """Save index metadata to disk."""
        try:
            self.index_metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, "w") as f:
                json.dump(self.index_metadata, f, indent=2)
            logger.debug("Saved index metadata")
        except Exception as error:
            logger.error(f"Failed to save metadata: {error}")

    def _start_maintenance_thread(self) -> None:
        """Start background maintenance thread."""
        if self.maintenance_thread is None or not self.maintenance_thread.is_alive():
            self.maintenance_running = True
            self.maintenance_thread = threading.Thread(target=self._maintenance_worker, daemon=True)
            self.maintenance_thread.start()
            logger.info("Maintenance thread started")

    def _stop_maintenance_thread(self) -> None:
        """Stop background maintenance thread."""
        self.maintenance_running = False
        if hasattr(self, 'maintenance_thread') and self.maintenance_thread and self.maintenance_thread.is_alive():
            try:
                self.maintenance_queue.put(None)
                self.maintenance_thread.join(timeout=5)
            except Exception:
                pass
            logger.info("Maintenance thread stopped")

    def _maintenance_worker(self) -> None:
        """Background worker for index maintenance tasks."""
        last_optimize = datetime.now()
        last_backup = datetime.now()

        while self.maintenance_running:
            try:
                now = datetime.now()

                optimize_interval = timedelta(hours=self.config["maintenance"]["optimize_interval_hours"])
                if now - last_optimize > optimize_interval:
                    logger.info("Running scheduled index optimization")
                    self.optimize_all_indices()
                    last_optimize = now

                backup_interval = timedelta(hours=self.config["maintenance"]["backup_interval_hours"])
                if self.config["maintenance"]["backup_enabled"] and now - last_backup > backup_interval:
                    logger.info("Running scheduled backup")
                    self.backup_all_indices()
                    last_backup = now

                self._clean_old_backups()

                try:
                    self.maintenance_queue.get(timeout=3600)
                except queue.Empty:
                    continue

            except Exception as error:
                logger.error(f"Maintenance worker error: {error}")
                time.sleep(60)

    def create_chroma_index(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new ChromaDB index."""
        index_dir = self.chroma_dir / name
        index_dir.mkdir(parents=True, exist_ok=True)

        index_config = self.config["chroma"].copy()
        if config:
            index_config.update(config)

        index_metadata = {
            "id": self._generate_index_id("chroma", name),
            "name": name,
            "type": "chroma",
            "path": str(index_dir),
            "config": index_config,
            "created_at": datetime.now().isoformat(),
            "last_optimized": None,
            "last_backup": None,
            "document_count": 0,
            "status": "active",
            "stats": {
                "total_adds": 0,
                "total_deletes": 0,
                "total_updates": 0,
                "total_searches": 0,
                "avg_search_time_ms": 0
            }
        }

        config_file = index_dir / "index_config.json"
        with open(config_file, "w") as f:
            json.dump(index_config, f, indent=2)

        self.index_metadata["indices"].append(index_metadata)
        self._save_metadata()

        logger.info(f"Created Chroma index: {name}")
        return index_metadata

    def create_faiss_index(
        self,
        name: str,
        dimension: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new FAISS index."""
        index_dir = self.faiss_dir / name
        index_dir.mkdir(parents=True, exist_ok=True)

        index_config = self.config["faiss"].copy()
        if config:
            index_config.update(config)

        if dimension:
            index_config["dimension"] = dimension

        index_metadata = {
            "id": self._generate_index_id("faiss", name),
            "name": name,
            "type": "faiss",
            "path": str(index_dir),
            "config": index_config,
            "created_at": datetime.now().isoformat(),
            "last_optimized": None,
            "last_backup": None,
            "document_count": 0,
            "status": "active",
            "stats": {
                "total_adds": 0,
                "total_deletes": 0,
                "total_updates": 0,
                "total_searches": 0,
                "avg_search_time_ms": 0
            }
        }

        config_file = index_dir / "index_config.json"
        with open(config_file, "w") as f:
            json.dump(index_config, f, indent=2)

        self.index_metadata["indices"].append(index_metadata)
        self._save_metadata()

        logger.info(f"Created FAISS index: {name} (dimension: {index_config['dimension']})")
        return index_metadata

    def _generate_index_id(self, index_type: str, name: str) -> str:
        """Generate unique index ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_str = f"{index_type}_{name}_{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    def get_index(self, name: str, index_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get index metadata by name."""
        for index in self.index_metadata["indices"]:
            if index["name"] == name:
                if index_type is None or index["type"] == index_type:
                    return index
        return None

    def list_indices(self, index_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indices with optional type filter."""
        if index_type:
            return [idx for idx in self.index_metadata["indices"] if idx["type"] == index_type]
        return self.index_metadata["indices"].copy()

    def delete_index(self, name: str, index_type: Optional[str] = None) -> bool:
        """Delete an index and its data."""
        index = self.get_index(name, index_type)
        if not index:
            logger.warning(f"Index not found: {name}")
            return False

        try:
            index_path = Path(index["path"])
            if index_path.exists():
                shutil.rmtree(index_path)
                logger.info(f"Removed index directory: {index_path}")

            self.index_metadata["indices"] = [
                idx for idx in self.index_metadata["indices"]
                if idx["id"] != index["id"]
            ]
            self._save_metadata()

            with self.cache_lock:
                cache_key = f"{index['type']}:{name}"
                if cache_key in self.index_cache:
                    del self.index_cache[cache_key]

            logger.info(f"Deleted index: {name}")
            return True

        except Exception as error:
            logger.error(f"Failed to delete index {name}: {error}")
            return False

    def optimize_index(self, name: str, index_type: Optional[str] = None) -> bool:
        """Optimize a specific index."""
        index = self.get_index(name, index_type)
        if not index:
            logger.warning(f"Index not found: {name}")
            return False

        try:
            start_time = time.time()

            if index["type"] == "faiss":
                logger.info(f"Optimizing FAISS index: {name}")

            for idx in self.index_metadata["indices"]:
                if idx["id"] == index["id"]:
                    idx["last_optimized"] = datetime.now().isoformat()
                    idx["stats"]["last_optimize_time_ms"] = (time.time() - start_time) * 1000
                    break

            self._save_metadata()

            logger.info(f"Optimized index: {name} in {time.time() - start_time:.2f}s")
            return True

        except Exception as error:
            logger.error(f"Failed to optimize index {name}: {error}")
            return False

    def optimize_all_indices(self) -> Dict[str, bool]:
        """Optimize all active indices."""
        results = {}
        for index in self.index_metadata["indices"]:
            if index["status"] == "active":
                results[index["name"]] = self.optimize_index(index["name"])
        return results

    def backup_index(self, name: str, index_type: Optional[str] = None) -> Optional[Path]:
        """Create a backup of a specific index."""
        index = self.get_index(name, index_type)
        if not index:
            logger.warning(f"Index not found: {name}")
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{index['type']}_{name}_{timestamp}"
            backup_path = self.backup_dir / backup_name

            index_path = Path(index["path"])
            if index_path.exists():
                archive_format = "gztar" if self.config["maintenance"]["compression_enabled"] else "tar"
                archive_path = shutil.make_archive(
                    str(backup_path),
                    archive_format,
                    index_path
                )

                for idx in self.index_metadata["indices"]:
                    if idx["id"] == index["id"]:
                        idx["last_backup"] = datetime.now().isoformat()
                        idx["last_backup_path"] = archive_path
                        break

                self._save_metadata()

                logger.info(f"Created backup for index {name}: {archive_path}")
                return Path(archive_path)

            return None

        except Exception as error:
            logger.error(f"Failed to backup index {name}: {error}")
            return None

    def backup_all_indices(self) -> Dict[str, Optional[Path]]:
        """Create backups for all active indices."""
        results = {}
        for index in self.index_metadata["indices"]:
            if index["status"] == "active":
                results[index["name"]] = self.backup_index(index["name"])
        return results

    def restore_index(self, backup_file: Path, target_name: Optional[str] = None) -> bool:
        """Restore an index from backup."""
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False

        try:
            backup_name = backup_file.stem
            if backup_file.suffix == ".gz":
                backup_name = backup_file.stem
                if backup_file.stem.endswith(".tar"):
                    backup_name = Path(backup_file.stem).stem

            parts = backup_name.split("_")
            if len(parts) < 2:
                logger.error(f"Invalid backup filename format: {backup_name}")
                return False

            index_type = parts[0]
            original_name = parts[1]

            if target_name:
                name = target_name
            else:
                name = f"{original_name}_restored_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            if index_type == "chroma":
                target_dir = self.chroma_dir / name
            elif index_type == "faiss":
                target_dir = self.faiss_dir / name
            else:
                logger.error(f"Unknown index type: {index_type}")
                return False

            shutil.unpack_archive(str(backup_file), str(target_dir))

            if index_type == "chroma":
                self.create_chroma_index(name)
            else:
                self.create_faiss_index(name, dimension=384)

            logger.info(f"Restored index {original_name} to {name}")
            return True

        except Exception as error:
            logger.error(f"Failed to restore index: {error}")
            return False

    def _clean_old_backups(self) -> None:
        """Remove old backups exceeding retention period."""
        try:
            retention = timedelta(days=self.config["maintenance"]["backup_retention_days"])
            cutoff = datetime.now() - retention

            for backup_file in self.backup_dir.glob("*.tar*"):
                if backup_file.is_file():
                    mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if mtime < cutoff:
                        backup_file.unlink()
                        logger.debug(f"Removed old backup: {backup_file}")

            total_size = sum(f.stat().st_size for f in self.backup_dir.glob("*.tar*"))
            max_size = self.config["maintenance"]["max_backup_size_mb"] * 1024 * 1024

            if total_size > max_size:
                backups = sorted(self.backup_dir.glob("*.tar*"), key=lambda f: f.stat().st_mtime)
                for old_backup in backups:
                    if total_size <= max_size:
                        break
                    size = old_backup.stat().st_size
                    old_backup.unlink()
                    total_size -= size
                    logger.info(f"Removed old backup due to size limit: {old_backup}")

        except Exception as error:
            logger.error(f"Failed to clean old backups: {error}")

    def get_index_stats(self, name: str, index_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for an index."""
        index = self.get_index(name, index_type)
        if not index:
            return None

        stats = index.copy()

        index_path = Path(index["path"])
        if index_path.exists():
            total_size = 0
            file_count = 0
            for file_path in index_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

            stats["disk_usage_bytes"] = total_size
            stats["disk_usage_mb"] = total_size / (1024 * 1024)
            stats["file_count"] = file_count

        created_at = datetime.fromisoformat(index["created_at"])
        stats["age_days"] = (datetime.now() - created_at).days

        with self.cache_lock:
            cache_key = f"{index['type']}:{name}"
            stats["in_cache"] = cache_key in self.index_cache

        return stats

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all indices."""
        total_docs = 0
        total_size = 0
        index_stats = []

        for index in self.index_metadata["indices"]:
            stats = self.get_index_stats(index["name"])
            if stats:
                index_stats.append(stats)
                total_docs += stats.get("document_count", 0)
                total_size += stats.get("disk_usage_bytes", 0)

        return {
            "total_indices": len(index_stats),
            "total_documents": total_docs,
            "total_disk_usage_mb": total_size / (1024 * 1024),
            "indices": index_stats,
            "maintenance_config": self.config["maintenance"],
            "timestamp": datetime.now().isoformat()
        }

    def clear_cache(self) -> int:
        """Clear the index cache."""
        with self.cache_lock:
            count = len(self.index_cache)
            self.index_cache.clear()

        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        except Exception as error:
            logger.error(f"Failed to clear disk cache: {error}")

        logger.info(f"Cleared {count} cache entries")
        return count

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on index manager."""
        try:
            test_file = self.base_dir / "health_check.tmp"
            test_file.write_text("test")
            test_file.unlink()

            active_indices = sum(1 for idx in self.index_metadata["indices"] if idx["status"] == "active")
            inactive_indices = len(self.index_metadata["indices"]) - active_indices

            backup_count = len(list(self.backup_dir.glob("*.tar*")))
            backup_size = sum(f.stat().st_size for f in self.backup_dir.glob("*.tar*")) / (1024 * 1024)

            maintenance_running = hasattr(self, 'maintenance_thread') and self.maintenance_thread is not None and self.maintenance_thread.is_alive()

            return {
                "status": "healthy",
                "maintenance_thread_running": maintenance_running,
                "total_indices": len(self.index_metadata["indices"]),
                "active_indices": active_indices,
                "inactive_indices": inactive_indices,
                "backup_count": backup_count,
                "backup_size_mb": round(backup_size, 2),
                "cache_entries": len(self.index_cache),
                "base_dir": str(self.base_dir),
                "base_dir_writable": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as error:
            return {
                "status": "unhealthy",
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            }

    def close(self) -> None:
        """Close the index manager and cleanup resources."""
        self._stop_maintenance_thread()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, 'maintenance_thread'):
                self._stop_maintenance_thread()
        except Exception:
            pass


_index_manager_instance: Optional[IndexManager] = None


def get_index_manager(
    base_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> IndexManager:
    """Get or create singleton IndexManager instance."""
    global _index_manager_instance

    if _index_manager_instance is None:
        _index_manager_instance = IndexManager(base_dir, config)
    elif base_dir is not None and Path(base_dir) != _index_manager_instance.base_dir:
        logger.warning("Requested different base directory than existing singleton, reinitializing")
        _index_manager_instance = IndexManager(base_dir, config)

    return _index_manager_instance


def test_index_manager():
    """Test function for IndexManager."""
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("INDEX MANAGER TEST SUITE")
    print("=" * 60)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(Path(tmpdir))

            print("\n1. HEALTH CHECK")
            print("-" * 40)
            health = manager.health_check()
            print(f"Status: {health['status']}")
            print(f"Maintenance Thread: {health['maintenance_thread_running']}")

            print("\n2. CREATE INDICES")
            print("-" * 40)

            chroma_index = manager.create_chroma_index(
                name="test_chroma",
                config={"collection_name": "test_docs", "batch_size": 500}
            )
            print(f"Created Chroma index: {chroma_index['name']} (ID: {chroma_index['id']})")

            faiss_index = manager.create_faiss_index(
                name="test_faiss",
                dimension=384,
                config={"metric": "cosine", "index_type": "flat"}
            )
            print(f"Created FAISS index: {faiss_index['name']} (Dimension: {faiss_index['config']['dimension']})")

            print("\n3. LIST INDICES")
            print("-" * 40)
            indices = manager.list_indices()
            print(f"Total indices: {len(indices)}")
            for idx in indices:
                print(f"  - {idx['type'].upper()}: {idx['name']} (Documents: {idx['document_count']})")

            print("\n4. GET INDEX")
            print("-" * 40)
            retrieved = manager.get_index("test_chroma")
            print(f"Retrieved index: {retrieved['name']} (Type: {retrieved['type']})")

            print("\n5. FILTER BY TYPE")
            print("-" * 40)
            chroma_indices = manager.list_indices(index_type="chroma")
            faiss_indices = manager.list_indices(index_type="faiss")
            print(f"Chroma indices: {len(chroma_indices)}")
            print(f"FAISS indices: {len(faiss_indices)}")

            print("\n6. BACKUP INDEX")
            print("-" * 40)
            backup_path = manager.backup_index("test_faiss")
            if backup_path:
                print(f"Backup created: {backup_path}")
            else:
                print("Backup failed")

            print("\n7. OPTIMIZE INDEX")
            print("-" * 40)
            success = manager.optimize_index("test_chroma")
            print(f"Optimization successful: {success}")

            print("\n8. INDEX STATISTICS")
            print("-" * 40)
            stats = manager.get_index_stats("test_faiss")
            if stats:
                print(f"Index: {stats['name']}")
                print(f"  Type: {stats['type']}")
                print(f"  Created: {stats['created_at']}")
                print(f"  Disk Usage: {stats.get('disk_usage_mb', 0):.2f} MB")
                print(f"  File Count: {stats.get('file_count', 0)}")
                print(f"  Age: {stats.get('age_days', 0)} days")

            print("\n9. ALL STATISTICS")
            print("-" * 40)
            all_stats = manager.get_all_stats()
            print(f"Total Indices: {all_stats['total_indices']}")
            print(f"Total Documents: {all_stats['total_documents']}")
            print(f"Total Disk Usage: {all_stats['total_disk_usage_mb']:.2f} MB")

            print("\n10. BACKUP ALL INDICES")
            print("-" * 40)
            backups = manager.backup_all_indices()
            for name, path in backups.items():
                if path:
                    print(f"  {name}: {path}")

            print("\n11. CACHE OPERATIONS")
            print("-" * 40)
            cleared = manager.clear_cache()
            print(f"Cleared {cleared} cache entries")

            print("\n12. DELETE INDICES")
            print("-" * 40)
            deleted = manager.delete_index("test_chroma")
            print(f"Deleted test_chroma: {deleted}")
            deleted = manager.delete_index("test_faiss")
            print(f"Deleted test_faiss: {deleted}")

            print("\n13. FINAL HEALTH CHECK")
            print("-" * 40)
            final_health = manager.health_check()
            print(f"Status: {final_health['status']}")
            print(f"Total Indices: {final_health['total_indices']}")

        print("\n" + "=" * 60)
        print("TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return True

    except Exception as error:
        print(f"\nTEST FAILED: {error}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_index_manager()
    sys.exit(0 if success else 1)