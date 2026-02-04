# cockatoo_v1/src/document_processing/extractors/csv_extractor.py

"""
CSV file extractor with support for various delimiters and encodings.
"""

import os
import csv
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import pandas as pd

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class CSVExtractor(BaseExtractor):
    """
    CSV file extractor with automatic delimiter detection.
    """
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.csv', '.tsv', '.txt']
    
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract data from CSV file.
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            Dictionary containing:
                - text: Tabular data as formatted text
                - metadata: CSV metadata
                - headers: List of column headers
                - rows: List of rows (list of values)
                - dataframe: Pandas DataFrame (if pandas available)
                - summary: Statistical summary
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Extracting CSV: {file_path}")
        
        result = {
            "text": "",
            "metadata": self.get_basic_metadata(file_path),
            "headers": [],
            "rows": [],
            "row_count": 0,
            "column_count": 0,
            "delimiter": ",",
            "encoding": "utf-8",
            "has_header": True,
            "dataframe": None,
            "summary": {},
            "warnings": [],
        }
        
        try:
            # Detect delimiter and encoding
            delimiter = self._detect_delimiter(file_path)
            encoding = self._detect_encoding(file_path)
            
            result["delimiter"] = delimiter
            result["encoding"] = encoding
            
            # Read CSV
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                # Sniff dialect
                sample = f.read(1024)
                f.seek(0)
                
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    has_header = sniffer.has_header(sample)
                except:
                    dialect = csv.excel()
                    has_header = True
                
                result["has_header"] = has_header
                
                # Read all rows
                reader = csv.reader(f, dialect)
                rows = list(reader)
                
                if not rows:
                    result["warnings"].append("CSV file is empty")
                    return result
                
                # Separate headers and data
                if has_header and rows:
                    result["headers"] = rows[0]
                    data_rows = rows[1:]
                else:
                    result["headers"] = [f"Column_{i+1}" for i in range(len(rows[0]))]
                    data_rows = rows
                
                result["rows"] = data_rows
                result["row_count"] = len(data_rows)
                result["column_count"] = len(result["headers"])
                
                # Generate formatted text
                result["text"] = self._format_as_text(result["headers"], data_rows, delimiter)
                
                # Try to create pandas DataFrame
                try:
                    df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                    result["dataframe"] = df
                    
                    # Generate summary statistics
                    result["summary"] = {
                        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                        "text_columns": df.select_dtypes(include=['object']).columns.tolist(),
                        "date_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
                        "null_counts": df.isnull().sum().to_dict(),
                        "unique_counts": df.nunique().to_dict(),
                    }
                except Exception as e:
                    logger.warning(f"Could not create DataFrame: {e}")
                    result["warnings"].append(f"Pandas processing failed: {e}")
                
                # Add column information
                result["columns"] = []
                for i, header in enumerate(result["headers"]):
                    col_data = [row[i] if i < len(row) else "" for row in data_rows]
                    
                    # Try to infer data type
                    data_type = self._infer_column_type(col_data)
                    
                    column_info = {
                        "name": header,
                        "index": i,
                        "data_type": data_type,
                        "non_empty_count": sum(1 for val in col_data if val and str(val).strip()),
                        "sample_values": col_data[:5] if col_data else [],
                    }
                    
                    # Add type-specific info
                    if data_type == "numeric":
                        try:
                            numeric_vals = [float(v) for v in col_data if v and self._is_numeric(v)]
                            if numeric_vals:
                                column_info.update({
                                    "min": min(numeric_vals),
                                    "max": max(numeric_vals),
                                    "avg": sum(numeric_vals) / len(numeric_vals),
                                })
                        except:
                            pass
                    elif data_type == "date":
                        # Could add date parsing here
                        pass
                    
                    result["columns"].append(column_info)
        
        except Exception as e:
            logger.error(f"Failed to extract CSV: {e}")
            result["warnings"].append(f"Extraction error: {e}")
        
        return result
    
    def _detect_delimiter(self, file_path: Path) -> str:
        """
        Detect CSV delimiter.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Detected delimiter
        """
        delimiters = [',', ';', '\t', '|', ':', '~']
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(4096)
            
            # Count occurrences of each delimiter
            delimiter_counts = {}
            for delim in delimiters:
                count = sample.count(delim)
                delimiter_counts[delim] = count
            
            # Return most common delimiter
            if delimiter_counts:
                return max(delimiter_counts.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logger.warning(f"Delimiter detection failed: {e}")
        
        return ","  # Default comma
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected encoding
        """
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
            
            if raw_data:
                detection = chardet.detect(raw_data)
                if detection['confidence'] > 0.5:
                    return detection['encoding'].lower()
        except:
            pass
        
        return 'utf-8'  # Default
    
    def _format_as_text(self, headers: List[str], rows: List[List[str]], delimiter: str) -> str:
        """
        Format CSV data as readable text.
        
        Args:
            headers: List of column headers
            rows: List of data rows
            delimiter: CSV delimiter
            
        Returns:
            Formatted text
        """
        if not headers and not rows:
            return ""
        
        # Create formatted output
        lines = []
        
        # Add headers
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * (len(" | ".join(headers))))
        
        # Add data rows (limit to first 50 rows for text representation)
        max_rows_display = 50
        for i, row in enumerate(rows[:max_rows_display]):
            # Ensure row has same number of columns as headers
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append(" | ".join(str(cell) for cell in padded_row))
        
        if len(rows) > max_rows_display:
            lines.append(f"\n... and {len(rows) - max_rows_display} more rows")
        
        return "\n".join(lines)
    
    def _infer_column_type(self, values: List[str]) -> str:
        """
        Infer data type from column values.
        
        Args:
            values: List of string values
            
        Returns:
            Inferred type: "text", "numeric", "date", or "boolean"
        """
        if not values:
            return "text"
        
        # Check if all values are numeric
        numeric_count = sum(1 for v in values if v and self._is_numeric(v))
        if numeric_count > len(values) * 0.8:  # 80% are numeric
            return "numeric"
        
        # Check for boolean
        bool_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
        bool_count = sum(1 for v in values if str(v).lower() in bool_values)
        if bool_count > len(values) * 0.8:
            return "boolean"
        
        # Check for dates (simplified)
        date_patterns = ['-', '/', ':' ]
        date_count = sum(1 for v in values if v and any(p in v for p in date_patterns))
        if date_count > len(values) * 0.5:
            return "date"
        
        return "text"
    
    def _is_numeric(self, value: str) -> bool:
        """
        Check if string can be converted to number.
        
        Args:
            value: String value
            
        Returns:
            True if value is numeric
        """
        try:
            float(value.replace(',', '').replace(' ', ''))
            return True
        except:
            return False
    
    def extract_summary_statistics(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract summary statistics from CSV.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Summary statistics
        """
        try:
            result = self.extract(file_path)
            
            summary = {
                "file_info": {
                    "rows": result["row_count"],
                    "columns": result["column_count"],
                    "has_header": result["has_header"],
                    "delimiter": result["delimiter"],
                },
                "columns": result.get("columns", []),
                "warnings": result.get("warnings", []),
            }
            
            # Add DataFrame-based statistics if available
            if result.get("dataframe") is not None:
                df = result["dataframe"]
                
                # Basic statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    summary["numeric_statistics"] = df[numeric_cols].describe().to_dict()
                
                # Value counts for categorical columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    summary["categorical_info"] = {}
                    for col in categorical_cols[:5]:  # Limit to first 5
                        top_values = df[col].value_counts().head(5).to_dict()
                        summary["categorical_info"][col] = {
                            "unique_values": df[col].nunique(),
                            "top_values": top_values,
                        }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to extract summary statistics: {e}")
            return {"error": str(e)}