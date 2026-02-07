# cockatoo_v1/src/database/debugger.py
"""
Database debugger and diagnostic tool for Cockatoo.

Provides database health checks, diagnostics, and automatic fixes.
Includes validation for structure, constraints, performance settings, and data integrity.
Generates detailed reports with actionable recommendations for production readiness.
"""
import sqlite3
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class CockatooDebugger:
    """Simple database debugger untuk Cockatoo."""
    
    def __init__(self, database_path: str = "cockatoo.db"):
        self.database_path = database_path
        self.results = []
    
    def log(self, test: str, status: str, details: Any = None) -> None:
        """Log hasil test."""
        result = {
            'test': test,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.results.append(result)
        
        if status == 'PASS':
            print(f"✅ {test}: PASS")
        elif status == 'FAIL':
            print(f"❌ {test}: FAIL - {details}")
        elif status == 'WARNING':
            print(f"⚠️  {test}: WARNING - {details}")
    
    def run_basic_checks(self) -> None:
        """Jalankan pengecekan dasar."""
        print("=" * 60)
        print("COCKATOO DATABASE BASIC DEBUG")
        print("=" * 60)
        
        checks = [
            self.check_file_existence,
            self.check_database_connection,
            self.check_foreign_keys,
            self.check_table_structure,
            self.check_basic_operations,
            self.check_settings
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.log(check.__name__, 'FAIL', f"Check error: {e}")
        
        self.generate_report()
    
    def check_file_existence(self) -> None:
        """Cek keberadaan file database."""
        db_path = Path(self.database_path)
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            details = {
                'path': str(db_path.absolute()),
                'size_mb': round(size_mb, 3),
                'exists': True
            }
            self.log("File Existence", 'PASS', details)
        else:
            self.log("File Existence", 'FAIL', 
                    f'File tidak ditemukan: {db_path.absolute()}')
    
    def check_database_connection(self) -> None:
        """Cek koneksi database."""
        try:
            start = time.time()
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Test version
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            
            # Test query
            cursor.execute("SELECT 1")
            
            # Get pragma info
            cursor.execute("PRAGMA compile_options")
            compile_options = [row[0] for row in cursor.fetchall()[:3]]
            
            conn.close()
            elapsed = time.time() - start
            
            details = {
                'version': version,
                'connection_time_ms': round(elapsed * 1000, 2),
                'compile_options': compile_options
            }
            self.log("Database Connection", 'PASS', details)
            
        except Exception as e:
            self.log("Database Connection", 'FAIL', str(e))
    
    def check_foreign_keys(self) -> None:
        """Cek foreign keys."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Enable and check
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA foreign_keys")
            fk_status = cursor.fetchone()[0]
            
            # Test foreign key constraint
            test_id = f"fk_test_{int(time.time())}"
            try:
                cursor.execute(
                    "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
                    (test_id, 'non_existent_conv', 'user', 'foreign key test')
                )
                # Should fail - if it doesn't, foreign keys not working
                conn.rollback()
                conn.close()
                self.log("Foreign Keys", 'FAIL', 
                        'Foreign key constraint not enforced')
                return
            except sqlite3.IntegrityError as e:
                if "FOREIGN KEY" in str(e):
                    # This is expected - foreign keys working
                    pass
                conn.rollback()
            
            conn.close()
            
            details = {
                'enabled': fk_status == 1,
                'working': True,
                'status': 'ENABLED' if fk_status == 1 else 'DISABLED'
            }
            
            if fk_status == 1:
                self.log("Foreign Keys", 'PASS', details)
            else:
                self.log("Foreign Keys", 'WARNING', 
                        'Foreign keys disabled but might work on connection')
                
        except Exception as e:
            self.log("Foreign Keys", 'FAIL', str(e))
    
    def check_table_structure(self) -> None:
        """Cek struktur tabel."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT name, sql 
                FROM sqlite_master 
                WHERE type='table' 
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = cursor.fetchall()
            
            table_info = {}
            for table_name, table_sql in tables:
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                table_info[table_name] = {
                    'columns': len(columns),
                    'rows': row_count,
                    'has_primary_key': any(col[5] == 1 for col in columns)
                }
            
            conn.close()
            
            # Check for required tables
            required_tables = ['documents', 'chunks', 'conversations', 'messages', 'tags', 'settings']
            found_tables = [t[0] for t in tables]
            missing_tables = [t for t in required_tables if t not in found_tables]
            
            # Check for typos
            has_typo = 'documments' in found_tables
            
            details = {
                'total_tables': len(tables),
                'found_tables': found_tables,
                'missing_required': missing_tables,
                'has_typo': has_typo,
                'table_details': table_info
            }
            
            if not missing_tables and not has_typo:
                self.log("Table Structure", 'PASS', details)
            elif has_typo:
                self.log("Table Structure", 'FAIL', 
                        "Typo found: 'documments' should be 'documents'")
            else:
                self.log("Table Structure", 'WARNING', 
                        f"Missing tables: {missing_tables}")
                
        except Exception as e:
            self.log("Table Structure", 'FAIL', str(e))
    
    def check_basic_operations(self) -> None:
        """Test operasi dasar CRUD."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            operations = {}
            
            # Test ID
            test_id = f"debug_test_{int(time.time())}"
            
            # 1. INSERT
            try:
                cursor.execute(
                    "INSERT INTO documents (id, file_path, file_name, file_type, processing_status) VALUES (?, ?, ?, ?, ?)",
                    (test_id, '/debug/test', 'debug_test.txt', 'text', 'pending')
                )
                operations['insert'] = True
            except Exception as e:
                operations['insert'] = False
                self.log("Basic Operations", 'FAIL', f"Insert failed: {e}")
                conn.rollback()
                conn.close()
                return
            
            # 2. SELECT
            cursor.execute("SELECT id FROM documents WHERE id = ?", (test_id,))
            result = cursor.fetchone()
            operations['select'] = result is not None
            
            # 3. UPDATE
            if result:
                cursor.execute(
                    "UPDATE documents SET file_name = ? WHERE id = ?",
                    ('debug_updated.txt', test_id)
                )
                operations['update'] = True
            else:
                operations['update'] = False
            
            # 4. DELETE
            cursor.execute("DELETE FROM documents WHERE id = ?", (test_id,))
            operations['delete'] = True
            
            conn.commit()
            conn.close()
            
            all_passed = all(operations.values())
            
            details = {
                'operations': operations,
                'all_passed': all_passed
            }
            
            if all_passed:
                self.log("Basic Operations", 'PASS', details)
            else:
                failed_ops = [op for op, success in operations.items() if not success]
                self.log("Basic Operations", 'FAIL', f"Failed: {failed_ops}")
                
        except Exception as e:
            self.log("Basic Operations", 'FAIL', str(e))
    
    def check_settings(self) -> None:
        """Cek pengaturan database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            pragmas = {
                'journal_mode': 'PRAGMA journal_mode',
                'synchronous': 'PRAGMA synchronous',
                'temp_store': 'PRAGMA temp_store',
                'cache_size': 'PRAGMA cache_size',
                'page_size': 'PRAGMA page_size'
            }
            
            results = {}
            for name, pragma in pragmas.items():
                try:
                    cursor.execute(pragma)
                    result = cursor.fetchone()
                    results[name] = result[0] if result else 'N/A'
                except:
                    results[name] = 'ERROR'
            
            # Get settings from settings table
            cursor.execute("SELECT key, value FROM settings")
            settings = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            # Analyze
            recommendations = []
            
            if results.get('journal_mode') != 'wal':
                recommendations.append("Enable WAL mode for better concurrency")
            
            if results.get('temp_store') != 2:  # 2 = MEMORY
                recommendations.append("Set temp_store to MEMORY for performance")
            
            if results.get('synchronous') == 0:  # 0 = OFF
                recommendations.append("Set synchronous to NORMAL for durability")
            
            details = {
                'pragma_settings': results,
                'app_settings': settings,
                'recommendations': recommendations if recommendations else ['Settings optimal']
            }
            
            if not recommendations:
                self.log("Database Settings", 'PASS', details)
            else:
                self.log("Database Settings", 'WARNING', details)
                
        except Exception as e:
            self.log("Database Settings", 'FAIL', str(e))
    
    def generate_report(self) -> None:
        """Hasilkan laporan debug."""
        print("\n" + "=" * 60)
        print("DEBUG REPORT")
        print("=" * 60)
        
        total = len(self.results)
        passed = len([r for r in self.results if r['status'] == 'PASS'])
        warnings = len([r for r in self.results if r['status'] == 'WARNING'])
        failed = len([r for r in self.results if r['status'] == 'FAIL'])
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Checks: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ⚠️  Warnings: {warnings}")
        print(f"   ❌ Failed: {failed}")
        
        if failed > 0:
            print(f"\n🔴 FAILED CHECKS:")
            for result in self.results:
                if result['status'] == 'FAIL':
                    print(f"   • {result['test']}")
        
        if warnings > 0:
            print(f"\n⚠️  WARNINGS:")
            for result in self.results:
                if result['status'] == 'WARNING':
                    details = result.get('details', {})
                    if isinstance(details, dict) and 'recommendations' in details:
                        for rec in details['recommendations']:
                            print(f"   • {rec}")
        
        # Overall status
        print("\n" + "=" * 60)
        if failed > 0:
            print("🔴 DATABASE HAS ISSUES")
            print("   Fix failed checks first")
        elif warnings > 0:
            print("🟡 DATABASE OK - WITH WARNINGS")
            print("   Consider addressing warnings")
        else:
            print("✅ DATABASE HEALTHY")
            print("   Ready for production use")
        print("=" * 60)
        
        # Save report
        self.save_json_report()
    
    def save_json_report(self) -> None:
        """Simpan laporan ke JSON."""
        report_file = f"cockatoo_debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'database': self.database_path,
                    'results': self.results,
                    'summary': {
                        'total': len(self.results),
                        'passed': len([r for r in self.results if r['status'] == 'PASS']),
                        'warnings': len([r for r in self.results if r['status'] == 'WARNING']),
                        'failed': len([r for r in self.results if r['status'] == 'FAIL'])
                    }
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 Report saved: {report_file}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save report: {e}")
    
    def quick_fix(self) -> None:
        """Perbaikan cepat untuk masalah umum."""
        print("=" * 60)
        print("QUICK FIX FOR COMMON ISSUES")
        print("=" * 60)
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            fixes = [
                ("Enable foreign keys", "PRAGMA foreign_keys = ON"),
                ("Enable WAL mode", "PRAGMA journal_mode = WAL"),
                ("Set temp store to memory", "PRAGMA temp_store = MEMORY"),
                ("Set synchronous mode", "PRAGMA synchronous = NORMAL"),
                ("Set cache size", "PRAGMA cache_size = -2000"),
                ("Set busy timeout", "PRAGMA busy_timeout = 5000"),
            ]
            
            for desc, sql in fixes:
                try:
                    cursor.execute(sql)
                    print(f"✅ {desc}")
                except Exception as e:
                    print(f"⚠️  {desc}: {e}")
            
            # Run VACUUM if needed
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            if table_count > 0:
                try:
                    cursor.execute("VACUUM")
                    print("✅ Database optimized (VACUUM)")
                except:
                    print("⚠️  Could not run VACUUM")
            
            conn.commit()
            conn.close()
            
            print("\n✅ Quick fixes applied")
            print("Run debug again to verify")
            
        except Exception as e:
            print(f"❌ Quick fix failed: {e}")


def main():
    """Fungsi utama debugger."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cockatoo Database Debugger',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debugger.py                    # Run basic checks
  python debugger.py --db custom.db     # Check custom database
  python debugger.py --fix              # Apply quick fixes
  python debugger.py --verify           # Verify after fixes
        """
    )
    
    parser.add_argument('--db', default='cockatoo.db', 
                       help='Database file path')
    parser.add_argument('--fix', action='store_true',
                       help='Apply quick fixes')
    parser.add_argument('--verify', action='store_true',
                       help='Verify database health')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report only')
    
    args = parser.parse_args()
    
    debugger = CockatooDebugger(args.db)
    
    if args.fix:
        # Apply quick fixes
        debugger.quick_fix()
    elif args.verify:
        # Run verification
        print("\n" + "="*60)
        print("VERIFICATION MODE")
        print("="*60)
        debugger.run_basic_checks()
    elif args.report:
        # Generate report from existing results
        if debugger.results:
            debugger.generate_report()
        else:
            print("No results to report. Run checks first.")
    else:
        # Default: run basic checks
        debugger.run_basic_checks()


if __name__ == "__main__":
    main()