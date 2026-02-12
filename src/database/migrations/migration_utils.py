# src/database/migrations/migration_utils.py

"""
Migration utility functions.

Provides comprehensive helper functions for migration operations including 
database operations, schema management, data operations, safety utilities,
and logging utilities.
"""

import hashlib
import json
import shutil
import sqlite3
import time
import csv
import io
import importlib
import importlib.util
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, Type
from contextlib import contextmanager
from typing import Dict, List
import logging
import sys

logger = logging.getLogger(__name__)


# ==============================
# DATABASE OPERATIONS
# ==============================

def backup_database(source_path: str, backup_path: str, include_wal: bool = True) -> bool:
    """
    Backup database file to specified location with proper WAL handling.
    
    Args:
        source_path: Path to source database file
        backup_path: Path where backup should be saved
        include_wal: Whether to include WAL files in backup
        
    Returns:
        True if backup successful, False otherwise
    """
    try:
        source = Path(source_path)
        backup = Path(backup_path)
        
        if not source.exists():
            logger.error(f"Source database not found: {source_path}")
            return False
        
        backup.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting database backup: {source_path} -> {backup_path}")
        
        with sqlite3.connect(source_path) as conn:
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            conn.commit()
        
        shutil.copy2(source_path, backup_path)
        
        if include_wal:
            wal_file = source.with_suffix('.db-wal')
            shm_file = source.with_suffix('.db-shm')
            
            if wal_file.exists():
                backup_wal = backup.parent / f"{backup.stem}.db-wal"
                shutil.copy2(wal_file, backup_wal)
                logger.debug(f"Backed up WAL file: {backup_wal}")
            
            if shm_file.exists():
                backup_shm = backup.parent / f"{backup.stem}.db-shm"
                shutil.copy2(shm_file, backup_shm)
                logger.debug(f"Backed up SHM file: {backup_shm}")
        
        if backup.exists() and backup.stat().st_size > 0:
            backup_size = backup.stat().st_size
            logger.info(f"Database backed up successfully: {backup_path} ({backup_size} bytes)")
            return True
        else:
            logger.error(f"Backup verification failed: {backup_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to backup database: {e}", exc_info=True)
        return False


def restore_database(backup_path: str, target_path: str, include_wal: bool = True) -> bool:
    """
    Restore database from backup file with WAL file handling.
    
    Args:
        backup_path: Path to backup file
        target_path: Path where database should be restored
        include_wal: Whether to restore WAL files
        
    Returns:
        True if restore successful, False otherwise
    """
    try:
        backup = Path(backup_path)
        target = Path(target_path)
        
        if not backup.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        target.parent.mkdir(parents=True, exist_ok=True)
        
        if target.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            old_backup = target.parent / f"{target.stem}_pre_restore_{timestamp}.db"
            shutil.move(target_path, str(old_backup))
            logger.info(f"Existing database backed up to: {old_backup}")
        
        shutil.copy2(backup_path, target_path)
        
        if include_wal:
            backup_wal = backup.with_suffix('.db-wal')
            target_wal = target.with_suffix('.db-wal')
            if backup_wal.exists():
                shutil.copy2(backup_wal, target_wal)
                logger.debug(f"Restored WAL file: {target_wal}")
            
            backup_shm = backup.with_suffix('.db-shm')
            target_shm = target.with_suffix('.db-shm')
            if backup_shm.exists():
                shutil.copy2(backup_shm, target_shm)
                logger.debug(f"Restored SHM file: {target_shm}")
        
        if target.exists() and target.stat().st_size > 0:
            restored_size = target.stat().st_size
            logger.info(f"Database restored successfully: {target_path} ({restored_size} bytes)")
            return True
        else:
            logger.error(f"Restore verification failed: {target_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to restore database: {e}", exc_info=True)
        return False


def validate_database_integrity(connection: sqlite3.Connection) -> Tuple[bool, str]:
    """
    Validate SQLite database integrity.
    
    Args:
        connection: SQLite database connection
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        cursor = connection.cursor()
        
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        
        if result == "ok":
            return True, "Database integrity check passed"
        else:
            return False, f"Database integrity check failed: {result}"
            
    except sqlite3.Error as e:
        return False, f"Failed to perform integrity check: {e}"


def get_database_info(connection: sqlite3.Connection) -> Dict[str, Any]:
    """
    Get comprehensive database information.
    
    Args:
        connection: SQLite database connection
        
    Returns:
        Dictionary with database information
    """
    try:
        cursor = connection.cursor()
        info = {
            'timestamp': datetime.utcnow().isoformat(),
            'sqlite_version': sqlite3.sqlite_version,
            'settings': {},
            'statistics': {}
        }
        
        cursor.execute("PRAGMA database_list")
        info['databases'] = cursor.fetchall()
        
        cursor.execute("PRAGMA journal_mode")
        info['journal_mode'] = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA synchronous")
        info['synchronous'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        info['total_tables'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
        info['total_indexes'] = cursor.fetchone()[0]
        
        logger.debug(f"Database info collected: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {'error': str(e)}


# ==============================
# SCHEMA OPERATIONS
# ==============================

def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        connection: Database connection
        table_name: Name of table to check
        
    Returns:
        True if table exists, False otherwise
    """
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        exists = cursor.fetchone() is not None
        logger.debug(f"Table '{table_name}' exists: {exists}")
        return exists
        
    except sqlite3.Error as e:
        logger.error(f"Failed to check if table exists: {e}")
        return False


def column_exists(connection: sqlite3.Connection, table_name: str, 
                 column_name: str) -> bool:
    """
    Check if a column exists in a table.
    
    Args:
        connection: Database connection
        table_name: Name of table
        column_name: Name of column to check
        
    Returns:
        True if column exists, False otherwise
    """
    try:
        cursor = connection.cursor()
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        exists = any(col[1] == column_name for col in columns)
        logger.debug(f"Column '{column_name}' in table '{table_name}' exists: {exists}")
        return exists
        
    except sqlite3.Error as e:
        logger.error(f"Failed to check if column exists: {e}")
        return False


def get_table_schema(connection: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
    """
    Get complete schema information for a table.
    
    Args:
        connection: Database connection
        table_name: Name of table
        
    Returns:
        Dictionary with schema information
    """
    try:
        cursor = connection.cursor()
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        create_statement = cursor.fetchone()
        
        columns = []
        for col in columns_info:
            columns.append({
                'cid': col[0],
                'name': col[1],
                'type': col[2],
                'notnull': bool(col[3]),
                'default_value': col[4],
                'pk': bool(col[5])
            })
        
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = cursor.fetchall()
        
        fk_list = []
        for fk in foreign_keys:
            fk_list.append({
                'id': fk[0],
                'seq': fk[1],
                'table': fk[2],
                'from': fk[3],
                'to': fk[4],
                'on_update': fk[5],
                'on_delete': fk[6],
                'match': fk[7]
            })
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        schema = {
            'table_name': table_name,
            'columns': columns,
            'create_statement': create_statement[0] if create_statement else None,
            'foreign_keys': fk_list,
            'row_count': row_count,
            'exists': len(columns) > 0
        }
        
        logger.debug(f"Schema retrieved for table '{table_name}': {len(columns)} columns, {row_count} rows")
        return schema
        
    except sqlite3.Error as e:
        logger.error(f"Failed to get table schema for '{table_name}': {e}")
        return {'table_name': table_name, 'exists': False, 'error': str(e)}


def get_indexes(connection: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """
    Get all indexes for a table.
    
    Args:
        connection: Database connection
        table_name: Name of table
        
    Returns:
        List of index information dictionaries
    """
    try:
        cursor = connection.cursor()
        
        cursor.execute(
            """
            SELECT 
                m.name as index_name,
                ii.name as column_name,
                ii.seqno as column_order,
                m.sql as create_statement
            FROM sqlite_master m
            JOIN pragma_index_info(m.name) ii
            WHERE m.type='index' 
                AND m.tbl_name=?
            ORDER BY m.name, ii.seqno
            """,
            (table_name,)
        )
        
        indexes_raw = cursor.fetchall()
        
        indexes = {}
        for idx in indexes_raw:
            idx_name = idx[0]
            if idx_name not in indexes:
                create_stmt = idx[3] if len(idx) > 3 else ""
                is_unique = "UNIQUE" in create_stmt.upper() if create_stmt else False
                is_partial = "WHERE" in create_stmt.upper() if create_stmt else False
                
                indexes[idx_name] = {
                    'index_name': idx_name,
                    'columns': [],
                    'is_unique': is_unique,
                    'is_partial': is_partial,
                    'create_statement': create_stmt
                }
            indexes[idx_name]['columns'].append({
                'name': idx[1],
                'order': idx[2]
            })
        
        result = list(indexes.values())
        logger.debug(f"Found {len(result)} indexes for table '{table_name}'")
        return result
        
    except sqlite3.Error as e:
        logger.error(f"Failed to get indexes for table '{table_name}': {e}")
        return []


# ==============================
# DATA OPERATIONS WITH OPTIMIZED CHUNKING
# ==============================

@contextmanager
def transaction(connection: sqlite3.Connection):
    """
    Context manager for database transactions.
    
    Args:
        connection: Database connection
        
    Yields:
        Database cursor
    """
    cursor = connection.cursor()
    try:
        logger.debug("Transaction started")
        yield cursor
        connection.commit()
        logger.debug("Transaction committed")
    except Exception as e:
        connection.rollback()
        logger.error(f"Transaction rolled back: {e}")
        raise e
    finally:
        cursor.close()


@contextmanager
def nested_transaction(connection: sqlite3.Connection, savepoint_name: str = None):
    """
    Context manager for nested transactions using savepoints.
    
    Args:
        connection: Database connection
        savepoint_name: Optional savepoint name
        
    Yields:
        Database cursor
    """
    if savepoint_name is None:
        savepoint_name = f"sp_{int(time.time())}_{hash(str(time.time()))}"
    
    cursor = connection.cursor()
    try:
        cursor.execute(f"SAVEPOINT {savepoint_name}")
        logger.debug(f"Savepoint '{savepoint_name}' created")
        yield cursor
        cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        logger.debug(f"Savepoint '{savepoint_name}' released")
    except Exception as e:
        cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
        logger.error(f"Rolled back to savepoint '{savepoint_name}': {e}")
        raise e
    finally:
        cursor.close()


def batch_insert(connection: sqlite3.Connection, table: str, data: List[Dict[str, Any]], 
                 batch_size: int = None) -> Tuple[int, List[str]]:
    """
    Insert data in batches with optimized chunking.
    
    Args:
        connection: Database connection
        table: Table name
        data: List of dictionaries with column-value pairs
        batch_size: Number of rows per batch (auto-calculated if None)
        
    Returns:
        Tuple of (inserted_count: int, errors: List[str])
    """
    if not data:
        logger.info(f"No data to insert into table '{table}'")
        return 0, []
    
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(len(data), connection)
        logger.debug(f"Auto-calculated batch size: {batch_size}")
    
    inserted = 0
    errors = []
    
    try:
        cursor = connection.cursor()
        
        columns = list(data[0].keys())
        columns_str = ', '.join(columns)
        placeholders = ', '.join(['?'] * len(columns))
        
        insert_sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
        
        total_rows = len(data)
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        logger.info(f"Starting batch insert into '{table}': {total_rows} rows in {total_batches} batches")
        
        start_time = time.time()
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, total_rows)
            batch = data[batch_start:batch_end]
            
            try:
                with nested_transaction(connection, f"batch_{batch_num}"):
                    batch_values = []
                    for row in batch:
                        row_values = []
                        for col in columns:
                            row_values.append(row.get(col))
                        batch_values.append(row_values)
                    
                    cursor.executemany(insert_sql, batch_values)
                    inserted += len(batch)
                
                if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
                    elapsed = time.time() - start_time
                    progress = (batch_num + 1) / total_batches * 100
                    logger.info(f"Batch insert progress: {batch_num + 1}/{total_batches} batches "
                               f"({progress:.1f}%) - {inserted} rows inserted in {format_duration(elapsed)}")
                
            except sqlite3.Error as e:
                error_msg = f"Batch {batch_num + 1} failed: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                
                logger.info(f"Attempting row-by-row insert for failed batch {batch_num + 1}")
                for row_idx, row in enumerate(batch):
                    try:
                        with nested_transaction(connection, f"row_{batch_num}_{row_idx}"):
                            row_values = [row.get(col) for col in columns]
                            cursor.execute(insert_sql, row_values)
                            inserted += 1
                    except sqlite3.Error as row_error:
                        row_error_msg = f"Row {row_idx} in batch {batch_num + 1} failed: {row_error}"
                        errors.append(row_error_msg)
                        logger.error(row_error_msg)
        
        total_time = time.time() - start_time
        rows_per_second = inserted / total_time if total_time > 0 else 0
        
        logger.info(f"Batch insert completed: {inserted}/{total_rows} rows inserted, "
                   f"{len(errors)} errors, {rows_per_second:.1f} rows/sec")
        return inserted, errors
        
    except Exception as e:
        connection.rollback()
        error_msg = f"Batch insert failed: {e}"
        errors.append(error_msg)
        logger.error(error_msg, exc_info=True)
        return inserted, errors


def batch_update(connection: sqlite3.Connection, table: str, 
                 updates: List[Tuple[Dict[str, Any], Dict[str, Any]]],
                 batch_size: int = None) -> Tuple[int, List[str]]:
    """
    Update data in batches with optimized chunking.
    
    Args:
        connection: Database connection
        table: Table name
        updates: List of tuples (set_values, where_conditions)
        batch_size: Number of updates per batch (auto-calculated if None)
        
    Returns:
        Tuple of (updated_count: int, errors: List[str])
    """
    if not updates:
        logger.info(f"No updates for table '{table}'")
        return 0, []
    
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(len(updates), connection)
        logger.debug(f"Auto-calculated batch size: {batch_size}")
    
    updated = 0
    errors = []
    
    try:
        cursor = connection.cursor()
        
        total_updates = len(updates)
        total_batches = (total_updates + batch_size - 1) // batch_size
        
        logger.info(f"Starting batch update on '{table}': {total_updates} updates in {total_batches} batches")
        
        start_time = time.time()
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, total_updates)
            batch = updates[batch_start:batch_end]
            
            with nested_transaction(connection, f"update_batch_{batch_num}"):
                for set_values, where_conditions in batch:
                    try:
                        set_clause = ', '.join([f"{k} = ?" for k in set_values.keys()])
                        set_params = list(set_values.values())
                        
                        where_clause = ' AND '.join([f"{k} = ?" for k in where_conditions.keys()])
                        where_params = list(where_conditions.values())
                        
                        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
                        cursor.execute(sql, set_params + where_params)
                        
                        updated += cursor.rowcount
                        
                    except sqlite3.Error as e:
                        error_msg = f"Update failed: {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
                elapsed = time.time() - start_time
                progress = (batch_num + 1) / total_batches * 100
                logger.info(f"Batch update progress: {batch_num + 1}/{total_batches} batches "
                           f"({progress:.1f}%) - {updated} rows updated in {format_duration(elapsed)}")
        
        total_time = time.time() - start_time
        updates_per_second = updated / total_time if total_time > 0 else 0
        
        logger.info(f"Batch update completed: {updated} rows updated, "
                   f"{len(errors)} errors, {updates_per_second:.1f} updates/sec")
        return updated, errors
        
    except Exception as e:
        connection.rollback()
        error_msg = f"Batch update failed: {e}"
        errors.append(error_msg)
        logger.error(error_msg, exc_info=True)
        return updated, errors


def calculate_optimal_batch_size(total_rows: int, connection: sqlite3.Connection) -> int:
    """
    Calculate optimal batch size based on data volume and connection settings.
    
    Args:
        total_rows: Total number of rows to process
        connection: Database connection
        
    Returns:
        Optimal batch size
    """
    if total_rows <= 100:
        return 10
    elif total_rows <= 1000:
        return 100
    elif total_rows <= 10000:
        return 500
    elif total_rows <= 100000:
        return 1000
    else:
        return 2000


def copy_table_data(source_conn: sqlite3.Connection, target_conn: sqlite3.Connection, 
                    table_name: str, where_clause: Optional[str] = None,
                    batch_size: int = None) -> Tuple[int, List[str]]:
    """
    Copy data from one database connection to another.
    
    Args:
        source_conn: Source database connection
        target_conn: Target database connection
        table_name: Table name to copy
        where_clause: Optional WHERE clause to filter rows
        batch_size: Number of rows per batch
        
    Returns:
        Tuple of (copied_count: int, errors: List[str])
    """
    copied = 0
    errors = []
    
    try:
        source_cursor = source_conn.cursor()
        target_cursor = target_conn.cursor()
        
        select_sql = f"SELECT * FROM {table_name}"
        if where_clause:
            select_sql += f" WHERE {where_clause}"
        
        logger.info(f"Starting table copy: {table_name}")
        
        source_cursor.execute(select_sql)
        rows = source_cursor.fetchall()
        column_names = [desc[0] for desc in source_cursor.description]
        
        if not rows:
            logger.info(f"No data to copy from table {table_name}")
            return 0, []
        
        if batch_size is None:
            batch_size = calculate_optimal_batch_size(len(rows), target_conn)
        
        columns_str = ', '.join(column_names)
        placeholders = ', '.join(['?'] * len(column_names))
        insert_sql = f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        
        total_rows = len(rows)
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        logger.info(f"Copying {total_rows} rows from '{table_name}' in {total_batches} batches")
        
        start_time = time.time()
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, total_rows)
            batch = rows[batch_start:batch_end]
            
            try:
                with nested_transaction(target_conn, f"copy_batch_{batch_num}"):
                    target_cursor.executemany(insert_sql, batch)
                    copied += len(batch)
                
                if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
                    elapsed = time.time() - start_time
                    progress = (batch_num + 1) / total_batches * 100
                    logger.info(f"Copy progress: {batch_num + 1}/{total_batches} batches "
                               f"({progress:.1f}%) - {copied} rows copied in {format_duration(elapsed)}")
                
            except sqlite3.Error as e:
                error_msg = f"Batch copy {batch_num + 1} failed: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                
                logger.info(f"Attempting row-by-row copy for failed batch {batch_num + 1}")
                for row in batch:
                    try:
                        with nested_transaction(target_conn, f"copy_row_{batch_num}"):
                            target_cursor.execute(insert_sql, row)
                            copied += 1
                    except sqlite3.Error as row_error:
                        errors.append(f"Row copy failed: {row_error}")
        
        target_conn.commit()
        total_time = time.time() - start_time
        rows_per_second = copied / total_time if total_time > 0 else 0
        
        logger.info(f"Table data copy completed: {copied}/{total_rows} rows copied, "
                   f"{len(errors)} errors, {rows_per_second:.1f} rows/sec")
        return copied, errors
        
    except Exception as e:
        target_conn.rollback()
        error_msg = f"Table data copy failed: {e}"
        errors.append(error_msg)
        logger.error(error_msg, exc_info=True)
        return copied, errors


# ==============================
# SAFETY UTILITIES
# ==============================

def create_temp_table(connection: sqlite3.Connection, original_table: str, 
                      suffix: str = "_temp") -> Optional[str]:
    """
    Create a temporary copy of a table as backup.
    
    Args:
        connection: Database connection
        original_table: Original table name
        suffix: Suffix for temporary table name
        
    Returns:
        Name of temporary table or None if failed
    """
    temp_table = f"{original_table}{suffix}"
    
    try:
        cursor = connection.cursor()
        
        if table_exists(connection, temp_table):
            logger.warning(f"Temporary table '{temp_table}' already exists, dropping it")
            cursor.execute(f"DROP TABLE {temp_table}")
        
        logger.info(f"Creating temporary table '{temp_table}' from '{original_table}'")
        cursor.execute(f"CREATE TABLE {temp_table} AS SELECT * FROM {original_table}")
        
        cursor.execute(f"SELECT COUNT(*) FROM {temp_table}")
        row_count = cursor.fetchone()[0]
        
        connection.commit()
        logger.info(f"Created temporary table '{temp_table}' with {row_count} rows")
        return temp_table
        
    except sqlite3.Error as e:
        connection.rollback()
        logger.error(f"Failed to create temporary table '{temp_table}': {e}")
        return None


def swap_tables(connection: sqlite3.Connection, table1: str, table2: str) -> bool:
    """
    Swap table names atomically.
    
    Args:
        connection: Database connection
        table1: First table name
        table2: Second table name
        
    Returns:
        True if swap successful, False otherwise
    """
    try:
        cursor = connection.cursor()
        
        temp_table = f"temp_swap_{int(time.time())}"
        
        logger.info(f"Swapping tables '{table1}' and '{table2}' using temporary table '{temp_table}'")
        
        cursor.execute(f"ALTER TABLE {table1} RENAME TO {temp_table}")
        cursor.execute(f"ALTER TABLE {table2} RENAME TO {table1}")
        cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table2}")
        
        connection.commit()
        logger.info(f"Successfully swapped tables: '{table1}' <-> '{table2}'")
        return True
        
    except sqlite3.Error as e:
        connection.rollback()
        logger.error(f"Failed to swap tables '{table1}' and '{table2}': {e}")
        return False


def verify_row_counts(connection: sqlite3.Connection, table: str, 
                      expected_count: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify row counts in a table.
    
    Args:
        connection: Database connection
        table: Table name
        expected_count: Optional expected row count
        
    Returns:
        Tuple of (is_valid: bool, info: Dict[str, Any])
    """
    try:
        cursor = connection.cursor()
        
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        actual_count = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT MIN(rowid), MAX(rowid) FROM {table}")
        min_rowid, max_rowid = cursor.fetchone()
        
        info = {
            'table': table,
            'actual_count': actual_count,
            'min_rowid': min_rowid,
            'max_rowid': max_rowid,
            'is_valid': True,
            'message': f"Table {table} has {actual_count} rows"
        }
        
        if expected_count is not None:
            if actual_count != expected_count:
                info.update({
                    'is_valid': False,
                    'expected_count': expected_count,
                    'difference': actual_count - expected_count,
                    'message': f"Row count mismatch: expected {expected_count}, got {actual_count}"
                })
                logger.warning(f"Row count mismatch for table '{table}': expected {expected_count}, got {actual_count}")
        
        logger.debug(f"Row count verification for table '{table}': {actual_count} rows")
        return info['is_valid'], info
        
    except sqlite3.Error as e:
        error_info = {
            'table': table,
            'is_valid': False,
            'message': f"Failed to verify row counts: {e}"
        }
        logger.error(f"Failed to verify row counts for table '{table}': {e}")
        return False, error_info


# ==============================
# LOGGING UTILITIES
# ==============================

def log_migration_step(message: str, level: str = "INFO", 
                       extra_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a migration step with structured data.
    
    Args:
        message: Log message
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        extra_data: Optional extra data to include in log
    """
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'message': message,
        'level': level.upper()
    }
    
    if extra_data:
        log_entry['data'] = extra_data
    
    if extra_data:
        data_str = json.dumps(extra_data, default=str, indent=2)
        log_message = f"{message}\n{data_str}"
    else:
        log_message = message
    
    log_level = level.upper()
    if log_level == "DEBUG":
        logger.debug(log_message)
    elif log_level == "INFO":
        logger.info(log_message)
    elif log_level == "WARNING":
        logger.warning(log_message)
    elif log_level == "ERROR":
        logger.error(log_message)
    else:
        logger.info(log_message)
    
    try:
        log_file = Path("logs/migration.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.debug(f"Failed to write to migration log file: {e}")


def log_data_stats(connection: sqlite3.Connection, table_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Log comprehensive data statistics.
    
    Args:
        connection: Database connection
        table_filter: Optional list of table names to include
        
    Returns:
        Dictionary with statistics
    """
    try:
        cursor = connection.cursor()
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'database_size': 0,
            'tables': {}
        }
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        all_tables = [row[0] for row in cursor.fetchall()]
        
        tables = all_tables
        if table_filter:
            tables = [t for t in all_tables if t in table_filter]
        
        total_rows = 0
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                cursor.execute(f"PRAGMA table_info({table})")
                column_count = len(cursor.fetchall())
                
                cursor.execute(f"SELECT SUM(LENGTH(hex(rowid))/2) FROM {table}")
                data_size = cursor.fetchone()[0] or 0
                
                stats['tables'][table] = {
                    'row_count': row_count,
                    'column_count': column_count,
                    'data_size_bytes': data_size
                }
                
                total_rows += row_count
                
            except sqlite3.Error as e:
                stats['tables'][table] = {
                    'error': str(e),
                    'row_count': 0
                }
        
        cursor.execute("PRAGMA database_list")
        for db_info in cursor.fetchall():
            if db_info[1] == 'main':
                db_file = Path(db_info[2])
                if db_file.exists():
                    stats['database_size'] = db_file.stat().st_size
                break
        
        stats['summary'] = {
            'total_tables': len(tables),
            'total_rows': total_rows,
            'total_size_bytes': stats['database_size']
        }
        
        log_migration_step(
            "Data Statistics Report",
            "INFO",
            stats
        )
        
        return stats
        
    except Exception as e:
        error_stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }
        log_migration_step(
            "Failed to collect data statistics",
            "ERROR",
            error_stats
        )
        return error_stats


# ==============================
# MIGRATION CLASS UTILITIES - FIXED VERSION
# ==============================

def load_migration_class(file_path: str, class_name: str = "Migration") -> Optional[Type]:
    """
    Dynamically load a migration class from a file path.
    
    Args:
        file_path: Path to the migration file
        class_name: Name of the class to load (default: "Migration")
        
    Returns:
        Migration class if successful, None otherwise
    """
    import importlib.util
    import sys
    from pathlib import Path
    import inspect
    
    try:
        logger.debug(f"Attempting to load migration class from: {file_path}")
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error(f"Migration file not found: {file_path}")
            return None
        
        module_name = file_path_obj.stem
        
        parent_dir = str(file_path_obj.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            logger.debug(f"Added {parent_dir} to sys.path")
        
        try:
            module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported module: {module_name}")
        except ImportError:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                logger.error(f"Failed to create module spec for {file_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.debug(f"Successfully loaded module from file: {module_name}")
        
        migration_class = getattr(module, class_name, None)
        
        if migration_class is None:
            try:
                from .base_migration import Migration
                
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Migration) and obj != Migration:
                        migration_class = obj
                        logger.debug(f"Found migration class: {name}")
                        break
            except ImportError:
                try:
                    from src.database.migrations.base_migration import Migration
                    
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Migration) and obj != Migration:
                            migration_class = obj
                            logger.debug(f"Found migration class: {name}")
                            break
                except ImportError as e:
                    logger.debug(f"Could not import base Migration class: {e}")
        
        if migration_class is None:
            logger.error(f"No migration class found in {file_path}")
            return None
        
        required_methods = ['upgrade', 'downgrade']
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(migration_class, method) or not callable(getattr(migration_class, method)):
                missing_methods.append(method)
        
        if missing_methods:
            logger.error(f"Migration class missing required methods: {missing_methods}")
            return None
        
        logger.info(f"Successfully loaded migration class from {file_path_obj.name}")
        return migration_class
        
    except Exception as e:
        logger.error(f"Failed to load migration class from {file_path}: {e}")
        return None


def discover_migration_classes(migrations_dir: str, base_module: str) -> List[Tuple[str, str, Type]]:
    """
    Discover all migration classes in a directory.
    
    Args:
        migrations_dir: Path to migrations directory
        base_module: Base module path (e.g., 'src.database.migrations')
        
    Returns:
        List of tuples (module_name, class_name, migration_class)
    """
    migrations = []
    migrations_path = Path(migrations_dir)
    
    if not migrations_path.exists():
        logger.error(f"Migrations directory not found: {migrations_dir}")
        return migrations
    
    logger.info(f"Scanning migration directory: {migrations_dir}")
    
    migration_files = list(migrations_path.glob("*.py"))
    logger.debug(f"Found {len(migration_files)} Python files in migrations directory")
    
    for py_file in migration_files:
        if py_file.name.startswith("__"):
            continue
        
        try:
            migration_class = load_migration_class(str(py_file), "Migration")
            if migration_class:
                module_name = py_file.stem
                full_module_path = f"{base_module}.{module_name}"
                class_name = migration_class.__name__
                migrations.append((full_module_path, class_name, migration_class))
                logger.info(f"Discovered migration: {full_module_path}.{class_name}")
        except Exception as e:
            logger.warning(f"Error processing migration file {py_file}: {e}")
    
    migrations.sort(key=lambda x: x[0])
    logger.info(f"Total migration classes discovered: {len(migrations)}")
    
    return migrations


def get_migration_class_by_version(version: str, migrations_dir: str, 
                                   base_module: str) -> Optional[Tuple[str, Type]]:
    """
    Get migration class by version identifier.
    
    Args:
        version: Migration version identifier (e.g., 'v001_initial')
        migrations_dir: Path to migrations directory
        base_module: Base module path
        
    Returns:
        Tuple of (class_name, migration_class) or None if not found
    """
    logger.info(f"Looking for migration class with version: {version}")
    
    migration_classes = discover_migration_classes(migrations_dir, base_module)
    
    for module_path, class_name, migration_class in migration_classes:
        if module_path.endswith(f".{version}"):
            logger.info(f"Found migration class for version '{version}': {class_name}")
            return class_name, migration_class
    
    logger.error(f"Migration class for version '{version}' not found")
    return None


def validate_migration_class(migration_class: Type) -> Tuple[bool, List[str]]:
    """
    Validate that a migration class has all required methods and attributes.
    
    Args:
        migration_class: Migration class to validate
        
    Returns:
        Tuple of (is_valid: bool, error_messages: List[str])
    """
    errors = []
    
    if not hasattr(migration_class, 'upgrade'):
        errors.append("Missing 'upgrade' method")
    
    if not hasattr(migration_class, 'downgrade'):
        errors.append("Missing 'downgrade' method")
    
    if not callable(getattr(migration_class, 'upgrade', None)):
        errors.append("'upgrade' method is not callable")
    
    if not callable(getattr(migration_class, 'downgrade', None)):
        errors.append("'downgrade' method is not callable")
    
    if not hasattr(migration_class, '__name__'):
        errors.append("Migration class has no name")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.debug(f"Migration class '{migration_class.__name__}' validation passed")
    else:
        logger.error(f"Migration class '{migration_class.__name__}' validation failed: {errors}")
    
    return is_valid, errors


# ==============================
# ADDITIONAL UTILITIES
# ==============================

def calculate_checksum(file_path: str) -> str:
    """
    Calculate SHA-256 checksum of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 checksum as hex string
    """
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        checksum = sha256_hash.hexdigest()
        logger.debug(f"Checksum calculated for {file_path}: {checksum[:16]}...")
        return checksum
        
    except Exception as e:
        logger.error(f"Failed to calculate checksum for {file_path}: {e}")
        return ""


def execute_sql_script(connection: sqlite3.Connection, script: str) -> bool:
    """
    Execute SQL script with proper transaction handling.
    
    Args:
        connection: SQLite database connection
        script: SQL script to execute
        
    Returns:
        True if execution successful, False otherwise
    """
    try:
        cursor = connection.cursor()
        
        statements = [stmt.strip() for stmt in script.split(';') if stmt.strip()]
        
        logger.info(f"Executing SQL script: {len(statements)} statements")
        
        start_time = time.time()
        
        for i, statement in enumerate(statements):
            try:
                cursor.execute(statement)
                logger.debug(f"Executed statement {i + 1}/{len(statements)}")
            except sqlite3.Error as e:
                logger.error(f"Failed to execute statement {i + 1}: {e}")
                logger.error(f"Problematic statement: {statement[:100]}...")
                raise
        
        connection.commit()
        
        execution_time = time.time() - start_time
        logger.info(f"SQL script execution completed: {len(statements)} statements in {format_duration(execution_time)}")
        return True
        
    except sqlite3.Error as e:
        connection.rollback()
        logger.error(f"Failed to execute SQL script: {e}")
        return False


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining = seconds % 60
        return f"{minutes}m {remaining:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# ==============================
# MIGRATION SPECIFIC UTILITIES
# ==============================

def generate_migration_id() -> str:
    """
    Generate a unique migration ID.
    
    Returns:
        Migration ID string
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    import random
    random_suffix = random.randint(1000, 9999)
    migration_id = f"mig_{timestamp}_{random_suffix}"
    logger.debug(f"Generated migration ID: {migration_id}")
    return migration_id


def estimate_migration_complexity(sql_statements: List[str]) -> Dict[str, Any]:
    """
    Estimate complexity of migration SQL statements.
    
    Args:
        sql_statements: List of SQL statements
        
    Returns:
        Complexity analysis dictionary
    """
    analysis = {
        'total_statements': len(sql_statements),
        'statement_types': {},
        'estimated_risk': 'low',
        'recommendations': []
    }
    
    for stmt in sql_statements:
        stmt_lower = stmt.lower().strip()
        
        if stmt_lower.startswith('create table'):
            analysis['statement_types']['create_table'] = analysis['statement_types'].get('create_table', 0) + 1
        elif stmt_lower.startswith('alter table'):
            analysis['statement_types']['alter_table'] = analysis['statement_types'].get('alter_table', 0) + 1
            analysis['estimated_risk'] = 'medium'
        elif stmt_lower.startswith('drop table'):
            analysis['statement_types']['drop_table'] = analysis['statement_types'].get('drop_table', 0) + 1
            analysis['estimated_risk'] = 'high'
            analysis['recommendations'].append('DROP TABLE operation detected - ensure backup')
        elif stmt_lower.startswith('delete from'):
            analysis['statement_types']['delete'] = analysis['statement_types'].get('delete', 0) + 1
            analysis['estimated_risk'] = 'high'
            analysis['recommendations'].append('DELETE operation detected - consider using soft delete')
        elif stmt_lower.startswith('update'):
            analysis['statement_types']['update'] = analysis['statement_types'].get('update', 0) + 1
            if 'where' not in stmt_lower:
                analysis['estimated_risk'] = 'critical'
                analysis['recommendations'].append('UPDATE without WHERE clause - will affect all rows!')
    
    logger.info(f"Migration complexity analysis: {analysis['total_statements']} statements, "
               f"risk level: {analysis['estimated_risk']}")
    return analysis


def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """
    Perform topological sort on dependency graph.
    
    Args:
        graph: Dictionary mapping node to list of dependencies
        
    Returns:
        List of nodes in topological order
        
    Raises:
        ValueError: If graph contains cycles
    """
    from collections import deque
    
    in_degree = {node: 0 for node in graph}
    for node, deps in graph.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[dep] = in_degree.get(dep, 0) + 1
    
    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for dep in graph.get(node, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
    
    if len(result) != len(graph):
        raise ValueError("Circular dependency detected in migration graph")
    
    return result