# cockatoo_v1/src/document_processing/extractors/json_extractor.py

"""
JSON document extractor with support for various JSON structures.
"""

import os
import json
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import yaml

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class JSONExtractor(BaseExtractor):
    """
    JSON document extractor with smart parsing and formatting.
    """
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.json', '.jsonld', '.geojson', '.topojson', '.jsonl']
    
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract data from JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Dictionary containing:
                - text: Formatted JSON content as text
                - metadata: File metadata
                - data: Parsed JSON data
                - structure: JSON structure information
                - stats: JSON statistics
                - validation: Validation results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        logger.info(f"Extracting JSON: {file_path}")
        
        result = {
            "text": "",
            "metadata": self.get_basic_metadata(file_path),
            "data": None,
            "structure": {},
            "stats": {},
            "validation": {
                "is_valid": False,
                "errors": [],
                "format": "unknown",
            },
            "lines": [],
        }
        
        # Determine JSON format
        file_format = self._detect_json_format(file_path)
        result["validation"]["format"] = file_format
        
        # Read and parse JSON
        try:
            if file_format == "jsonl":
                result = self._parse_jsonl(file_path, result)
            else:
                result = self._parse_json(file_path, result)
            
            result["validation"]["is_valid"] = True
            
        except json.JSONDecodeError as e:
            result["validation"]["errors"].append(f"JSON parsing error: {e}")
            logger.error(f"Failed to parse JSON: {e}")
            
            # Try to extract partial content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    result["text"] = content[:10000]  # Limit text output
            except:
                pass
        except Exception as e:
            result["validation"]["errors"].append(f"Unexpected error: {e}")
            logger.error(f"Failed to extract JSON: {e}")
        
        # Add language detection (JSON is typically English for keys)
        if result["text"]:
            result["metadata"]["language"] = "en"  # JSON keys are typically English
            result["metadata"]["summary"] = self._generate_json_summary(result)
        
        return result
    
    def _detect_json_format(self, file_path: Path) -> str:
        """
        Detect JSON file format.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Format string: 'json', 'jsonl', or 'unknown'
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
                
                # Check for JSONL (each line is a JSON object)
                if first_line and second_line:
                    try:
                        json.loads(first_line)
                        json.loads(second_line)
                        return "jsonl"
                    except:
                        pass
                
                # Check for regular JSON
                f.seek(0)
                content = f.read()
                try:
                    json.loads(content)
                    return "json"
                except:
                    pass
                
        except Exception as e:
            logger.warning(f"Format detection failed: {e}")
        
        return "unknown"
    
    def _parse_json(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse regular JSON file.
        
        Args:
            file_path: Path to JSON file
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse JSON
        data = json.loads(content)
        result["data"] = data
        
        # Generate formatted text
        formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
        result["text"] = formatted_json
        
        # Analyze structure
        structure = self._analyze_json_structure(data)
        result["structure"] = structure
        
        # Calculate statistics
        stats = self._calculate_json_stats(data)
        result["stats"] = stats
        
        # Extract lines for text representation
        lines = formatted_json.split('\n')
        result["lines"] = lines
        
        return result
    
    def _parse_jsonl(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JSON Lines (JSONL) file.
        
        Args:
            file_path: Path to JSONL file
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        lines = []
        objects = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    lines.append(line)
                    try:
                        obj = json.loads(line)
                        objects.append(obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        result["validation"]["errors"].append(f"Line {line_num}: {e}")
        
        result["data"] = objects
        result["lines"] = lines
        
        # Generate formatted text
        formatted_objects = []
        for i, obj in enumerate(objects[:50]):  # Limit to first 50 objects
            formatted_obj = json.dumps(obj, indent=2, ensure_ascii=False)
            formatted_objects.append(f"Object {i+1}:\n{formatted_obj}")
        
        if len(objects) > 50:
            formatted_objects.append(f"\n... and {len(objects) - 50} more objects")
        
        result["text"] = "\n\n".join(formatted_objects)
        
        # Analyze structure (using first object as sample)
        if objects:
            structure = self._analyze_json_structure(objects[0])
            result["structure"] = structure
            
            # Check if all objects have similar structure
            if len(objects) > 1:
                second_structure = self._analyze_json_structure(objects[1])
                if structure != second_structure:
                    result["validation"]["warnings"] = ["JSONL objects have different structures"]
        
        # Calculate statistics
        stats = {
            "total_objects": len(objects),
            "valid_lines": len(objects),
            "invalid_lines": len(lines) - len(objects),
            "avg_object_size": sum(len(json.dumps(obj)) for obj in objects) / len(objects) if objects else 0,
        }
        result["stats"] = stats
        
        return result
    
    def _analyze_json_structure(self, data: Any, path: str = "") -> Dict[str, Any]:
        """
        Recursively analyze JSON structure.
        
        Args:
            data: JSON data
            path: Current path in JSON
            
        Returns:
            Structure information
        """
        structure = {
            "type": type(data).__name__,
            "path": path or "root",
        }
        
        if isinstance(data, dict):
            structure["type"] = "object"
            structure["keys"] = list(data.keys())
            structure["key_count"] = len(data.keys())
            structure["children"] = []
            
            for key, value in data.items():
                child_path = f"{path}.{key}" if path else key
                child_structure = self._analyze_json_structure(value, child_path)
                structure["children"].append(child_structure)
            
        elif isinstance(data, list):
            structure["type"] = "array"
            structure["length"] = len(data)
            
            if data:
                # Analyze first element as sample
                structure["item_type"] = self._analyze_json_structure(data[0], f"{path}[0]")
                
                # Check if all items have same type
                if len(data) > 1:
                    second_type = type(data[1]).__name__
                    if type(data[0]).__name__ != second_type:
                        structure["uniform"] = False
                    else:
                        structure["uniform"] = True
                else:
                    structure["uniform"] = True
            else:
                structure["item_type"] = "unknown"
                structure["uniform"] = True
        
        elif isinstance(data, (str, int, float, bool, type(None))):
            structure["value_sample"] = str(data)[:100]  # Limit sample length
        
        return structure
    
    def _calculate_json_stats(self, data: Any) -> Dict[str, Any]:
        """
        Calculate JSON statistics.
        
        Args:
            data: JSON data
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_nodes": 0,
            "max_depth": 0,
            "string_count": 0,
            "number_count": 0,
            "boolean_count": 0,
            "null_count": 0,
            "array_count": 0,
            "object_count": 0,
            "total_string_length": 0,
        }
        
        def traverse(node, depth=0):
            stats["total_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)
            
            if isinstance(node, dict):
                stats["object_count"] += 1
                for key, value in node.items():
                    traverse(value, depth + 1)
            elif isinstance(node, list):
                stats["array_count"] += 1
                for item in node:
                    traverse(item, depth + 1)
            elif isinstance(node, str):
                stats["string_count"] += 1
                stats["total_string_length"] += len(node)
            elif isinstance(node, (int, float)):
                stats["number_count"] += 1
            elif isinstance(node, bool):
                stats["boolean_count"] += 1
            elif node is None:
                stats["null_count"] += 1
        
        traverse(data)
        
        # Calculate averages
        if stats["string_count"] > 0:
            stats["avg_string_length"] = stats["total_string_length"] / stats["string_count"]
        
        # Size estimates
        json_str = json.dumps(data)
        stats["raw_size_bytes"] = len(json_str.encode('utf-8'))
        stats["formatted_size_bytes"] = len(json.dumps(data, indent=2).encode('utf-8'))
        
        return stats
    
    def _generate_json_summary(self, result: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of JSON.
        
        Args:
            result: Extraction result
            
        Returns:
            Summary string
        """
        stats = result.get("stats", {})
        structure = result.get("structure", {})
        
        summary_parts = []
        
        if "format" in result["validation"]:
            summary_parts.append(f"Format: {result['validation']['format']}")
        
        if "total_nodes" in stats:
            summary_parts.append(f"Total nodes: {stats['total_nodes']}")
        
        if structure.get("type") == "object" and "key_count" in structure:
            summary_parts.append(f"Top-level keys: {structure['key_count']}")
            if structure.get("keys"):
                key_sample = ", ".join(structure["keys"][:5])
                if len(structure["keys"]) > 5:
                    key_sample += f" (+{len(structure['keys']) - 5} more)"
                summary_parts.append(f"Keys: {key_sample}")
        
        elif structure.get("type") == "array" and "length" in structure:
            summary_parts.append(f"Array length: {structure['length']}")
            if "item_type" in structure:
                summary_parts.append(f"Item type: {structure['item_type'].get('type', 'unknown')}")
        
        # Add data type counts
        type_counts = []
        for key in ["object_count", "array_count", "string_count", "number_count", "boolean_count", "null_count"]:
            if stats.get(key, 0) > 0:
                type_name = key.replace("_count", "").replace("_", " ")
                type_counts.append(f"{type_name}s: {stats[key]}")
        
        if type_counts:
            summary_parts.append("Type breakdown: " + ", ".join(type_counts))
        
        return "; ".join(summary_parts)
    
    def extract_schema(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract JSON schema (simplified).
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Simplified schema
        """
        result = self.extract(file_path)
        
        if not result["validation"]["is_valid"]:
            return {"error": "Invalid JSON", "details": result["validation"]["errors"]}
        
        def generate_schema(node, path=""):
            if isinstance(node, dict):
                schema = {
                    "type": "object",
                    "properties": {},
                    "required": list(node.keys()),
                }
                
                for key, value in node.items():
                    child_path = f"{path}.{key}" if path else key
                    schema["properties"][key] = generate_schema(value, child_path)
                
                return schema
            
            elif isinstance(node, list):
                if node:
                    # Use first item as sample
                    return {
                        "type": "array",
                        "items": generate_schema(node[0], f"{path}[0]"),
                    }
                else:
                    return {
                        "type": "array",
                        "items": {},
                    }
            
            else:
                return {
                    "type": type(node).__name__,
                    "example": str(node)[:100] if node is not None else None,
                }
        
        if result["data"]:
            schema = generate_schema(result["data"])
            schema["stats"] = result["stats"]
            return schema
        else:
            return {"error": "No data to generate schema"}
    
    def flatten_json(self, file_path: Union[str, Path], separator: str = '.') -> Dict[str, Any]:
        """
        Flatten JSON structure into key-value pairs.
        
        Args:
            file_path: Path to JSON file
            separator: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        result = self.extract(file_path)
        
        if not result["validation"]["is_valid"]:
            return {}
        
        def flatten(obj, parent_key='', sep=separator):
            items = []
            
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    items.extend(flatten(v, new_key, sep).items())
            
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                    items.extend(flatten(v, new_key, sep).items())
            
            else:
                items.append((parent_key, obj))
            
            return dict(items)
        
        if result["data"]:
            return flatten(result["data"])
        else:
            return {}