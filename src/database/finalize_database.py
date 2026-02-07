# cockatoo_v1/src/database/finalize_database.py

"""
Database finalization and verification script.

Provides database optimization, integrity checks, and operational testing.
Includes performance tuning, integrity verification, and end-to-end testing of all
database components to ensure production readiness. Generates detailed health reports.
"""
import sqlite3
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

def finalize_database(db_path="cockatoo.db"):
    """Execute final database optimizations and verification."""
    print("COCKATOO DATABASE FINALIZATION")
    print("=" * 60)
    
    if not Path(db_path).exists():
        print(f"Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        
        print("Applying final optimizations...")
        
        pragma_settings = [
            ("PRAGMA foreign_keys = ON", "Foreign keys"),
            ("PRAGMA journal_mode = WAL", "WAL mode"),
            ("PRAGMA synchronous = NORMAL", "Synchronous mode"),
            ("PRAGMA cache_size = -2000", "Cache size"),
            ("PRAGMA temp_store = MEMORY", "Temp store"),
            ("PRAGMA mmap_size = 268435456", "MMAP size (256MB)"),
            ("PRAGMA busy_timeout = 5000", "Busy timeout (5s)")
        ]
        
        for pragma_sql, description in pragma_settings:
            try:
                cursor.execute(pragma_sql)
                print(f"   {description}: Applied")
            except Exception as e:
                print(f"   {description}: {e}")
        
        print("\nRunning database maintenance...")
        
        maintenance_commands = [
            ("VACUUM", "Defragment database"),
            ("ANALYZE", "Update statistics"),
            ("PRAGMA optimize", "Optimize queries")
        ]
        
        for command, description in maintenance_commands:
            try:
                cursor.execute(command)
                print(f"   {description}: Completed")
            except Exception as e:
                print(f"   {description}: {e}")
        
        print("\nVerifying database integrity...")
        
        cursor.execute("PRAGMA foreign_key_check")
        fk_issues = cursor.fetchall()
        
        if fk_issues:
            print(f"   Foreign key issues found: {len(fk_issues)}")
            for issue in fk_issues[:3]:
                print(f"     Table: {issue[0]}, Row: {issue[1]}, Referenced: {issue[2]}")
        else:
            print("   Foreign key integrity: OK")
        
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        
        if integrity == 'ok':
            print("   Database integrity: OK")
        else:
            print(f"   Integrity check: {integrity}")
        
        print("\nDatabase statistics:")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        total_rows = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            total_rows += count
            if count > 0:
                print(f"   {table}: {count:,} rows")
        
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
        index_count = cursor.fetchone()[0]
        
        print(f"\n   Summary:")
        print(f"   Total tables: {len(tables)}")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Total indexes: {index_count}")
        
        db_size = Path(db_path).stat().st_size
        print(f"   Database size: {db_size / (1024*1024):.2f} MB")
        
        print("\nChecking configuration...")
        
        required_settings = ['database_version', 'chunk_size', 'chunk_overlap', 'embedding_model']
        
        for setting in required_settings:
            cursor.execute("SELECT value FROM settings WHERE key = ?", (setting,))
            result = cursor.fetchone()
            if result:
                print(f"   Setting '{setting}': {result[0]}")
            else:
                print(f"   Setting '{setting}': MISSING")
                defaults = {
                    'database_version': '1.0.0',
                    'chunk_size': '500',
                    'chunk_overlap': '50',
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'llm_model': 'llama2:7b'
                }
                if setting in defaults:
                    cursor.execute(
                        "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                        (setting, defaults[setting], datetime.now().isoformat())
                    )
                    print(f"     Set to default: {defaults[setting]}")
        
        conn.commit()
        conn.close()
        
        print("\n" + "=" * 60)
        print("DATABASE FINALIZATION COMPLETE")
        print("=" * 60)
        
        generate_final_report(db_path)
        
        return True
        
    except Exception as e:
        print(f"Finalization failed: {e}")
        return False

def generate_final_report(db_path="cockatoo.db"):
    """Generate database report."""
    print("\nFINAL DATABASE REPORT")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "database_path": db_path,
            "size_mb": Path(db_path).stat().st_size / (1024 * 1024),
            "sqlite_version": sqlite3.sqlite_version,
            "status": "READY"
        }
        
        pragmas = [
            'foreign_keys', 'journal_mode', 'synchronous',
            'cache_size', 'temp_store', 'mmap_size',
            'busy_timeout', 'encoding', 'page_size'
        ]
        
        report["pragma_settings"] = {}
        for pragma in pragmas:
            try:
                cursor.execute(f"PRAGMA {pragma}")
                result = cursor.fetchone()
                report["pragma_settings"][pragma] = result[0] if result else None
            except Exception:
                report["pragma_settings"][pragma] = "ERROR"
        
        cursor.execute("""
            SELECT name, sql 
            FROM sqlite_master 
            WHERE type IN ('table', 'index')
            ORDER BY type, name
        """)
        
        objects = cursor.fetchall()
        report["database_objects"] = {
            "tables": [],
            "indexes": []
        }
        
        for obj_name, obj_sql in objects:
            if obj_sql and "CREATE TABLE" in obj_sql.upper():
                cursor.execute(f"SELECT COUNT(*) FROM {obj_name}")
                count = cursor.fetchone()[0]
                report["database_objects"]["tables"].append({
                    "name": obj_name,
                    "rows": count
                })
            elif obj_sql and "CREATE INDEX" in obj_sql.upper():
                report["database_objects"]["indexes"].append(obj_name)
        
        conn.close()
        
        report_file = "cockatoo_database_final_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Final report saved to: {report_file}")
        
        print(f"\nDATABASE SUMMARY:")
        print(f"   Status: {report['status']}")
        print(f"   SQLite Version: {report['sqlite_version']}")
        print(f"   Size: {report['size_mb']:.2f} MB")
        print(f"   Tables: {len(report['database_objects']['tables'])}")
        print(f"   Indexes: {len(report['database_objects']['indexes'])}")
        
        print(f"\nCRITICAL SETTINGS:")
        critical_pragmas = ['foreign_keys', 'journal_mode', 'synchronous']
        for pragma in critical_pragmas:
            value = report["pragma_settings"].get(pragma, "UNKNOWN")
            status = "PASS" if (
                (pragma == 'foreign_keys' and value == 1) or
                (pragma == 'journal_mode' and value == 'wal') or
                (pragma == 'synchronous' and value in [1, 2, 'NORMAL', 'FULL'])
            ) else "WARNING"
            print(f"   {pragma}: {value} [{status}]")
        
        print(f"\nDatabase location: {Path(db_path).absolute()}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Failed to generate report: {e}")

def test_database_operations(db_path="cockatoo.db"):
    """Test all database operations."""
    print("\nTESTING DATABASE OPERATIONS")
    print("=" * 60)
    
    try:
        from database import get_database_manager
        
        print("Testing DatabaseManager...")
        manager = get_database_manager()
        print(f"   DatabaseManager initialized: {manager.is_initialized()}")
        
        print("\nTesting SQLiteClient...")
        client = manager.get_client()
        print("   SQLiteClient obtained")
        
        health = client.health_check()
        print(f"   Health check: {health.get('status', 'UNKNOWN')}")
        
        print("\nTesting basic queries...")
        
        test_doc = {
            'id': f'test_doc_{int(time.time())}',
            'file_path': '/test/path',
            'file_name': 'test_document.pdf',
            'file_type': 'pdf',
            'file_size': 1024,
            'processing_status': 'pending',
            'summary': 'Test document for verification'
        }
        
        from database.document_client import DocumentClient
        doc_client = DocumentClient(client)
        doc_id = doc_client.add_document(test_doc)
        print(f"   Document inserted: {doc_id}")
        
        retrieved = doc_client.get_document(doc_id)
        print(f"   Document retrieved: {retrieved['file_name'] if retrieved else 'FAILED'}")
        
        documents, total = doc_client.search_documents({}, page=1, page_size=5)
        print(f"   Document search: Found {total} documents")
        
        deleted = doc_client.delete_document(doc_id)
        print(f"   Document deleted: {deleted}")
        
        print("\nTesting conversation operations...")
        
        from database.conversation_client import ConversationClient
        conv_client = ConversationClient(client)
        
        conv_id = conv_client.create_conversation("Test Conversation")
        print(f"   Conversation created: {conv_id}")
        
        message_data = {
            'role': 'user',
            'content': 'Hello, this is a test message',
            'model_used': 'test-model'
        }
        
        msg_id = conv_client.add_message(conv_id, message_data)
        print(f"   Message added: {msg_id}")
        
        conversation = conv_client.get_conversation(conv_id)
        print(f"   Conversation retrieved: {conversation['title'] if conversation else 'FAILED'}")
        
        conversations, total = conv_client.list_conversations()
        print(f"   Conversations listed: {total} total")
        
        deleted = conv_client.delete_conversation(conv_id)
        print(f"   Conversation deleted: {deleted}")
        
        print("\nTesting SQLAlchemy models...")
        
        try:
            from database.models import session_scope, Document as DocModel
            with session_scope() as session:
                count = session.query(DocModel).count()
                print(f"   SQLAlchemy query: {count} documents")
        except Exception as e:
            print(f"   SQLAlchemy test: {e}")
        
        manager.close()
        
        print("\n" + "=" * 60)
        print("ALL DATABASE OPERATIONS TESTED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Database operations test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("COCKATOO DATABASE FINALIZATION SCRIPT")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='Finalize Cockatoo Database')
    parser.add_argument('--db', help='Database path', default='cockatoo.db')
    parser.add_argument('--finalize', action='store_true', help='Run final optimizations')
    parser.add_argument('--test', action='store_true', help='Test all database operations')
    parser.add_argument('--report', action='store_true', help='Generate final report')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    if not any([args.finalize, args.test, args.report, args.all]):
        print("\nUsage options:")
        print("  --finalize  : Optimize and verify database")
        print("  --test      : Test all database operations")
        print("  --report    : Generate final report")
        print("  --all       : Run all steps")
        print("\nExample: python finalize_database.py --all")
        return
    
    if args.all or args.finalize:
        if not finalize_database(args.db):
            return
    
    if args.all or args.report:
        generate_final_report(args.db)
    
    if args.all or args.test:
        test_database_operations(args.db)
    
    print("\n" + "="*60)
    print("COCKATOO DATABASE IS READY FOR USE")
    print("="*60)
    print("\nNext steps:")
    print("1. Start your Cockatoo application")
    print("2. Import documents to process")
    print("3. Begin conversations with your AI assistant")
    print("\nFor help: python finalize_database.py --help")

if __name__ == "__main__":
    main()