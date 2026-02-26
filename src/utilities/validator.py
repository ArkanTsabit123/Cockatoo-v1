# src/utilities/validator.py

"""Data validation utilities with schema support and common validators."""

import re
import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool = True, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings: List[str] = []
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.is_valid = False
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def __str__(self) -> str:
        if self.is_valid:
            return "Valid"
        return f"Invalid: {', '.join(self.errors)}"


@dataclass
class ValidationRule:
    """A single validation rule."""
    name: str
    validator: Callable[[Any], bool]
    message: str
    level: str = "error"


class Validator:
    """Base validator that can combine multiple rules."""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules.append(rule)
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate data against all rules."""
        result = ValidationResult()
        
        for rule in self.rules:
            try:
                if not rule.validator(data):
                    if rule.level == "error":
                        result.add_error(rule.message)
                    else:
                        result.add_warning(rule.message)
            except Exception as e:
                result.add_error(f"Rule '{rule.name}' failed: {str(e)}")
        
        return result


class SchemaValidator(Validator):
    """Validator based on a JSON-like schema."""
    
    def __init__(self, schema: Dict[str, Any]):
        super().__init__()
        self.schema = schema
        self._build_rules()
    
    def _build_rules(self):
        for field_name, field_schema in self.schema.items():
            if field_schema.get('required', False):
                self.add_rule(ValidationRule(
                    name=f"{field_name}_required",
                    validator=lambda d, f=field_name: f in d,
                    message=f"Field '{field_name}' is required"
                ))
            
            expected_type = field_schema.get('type')
            if expected_type:
                self.add_rule(ValidationRule(
                    name=f"{field_name}_type",
                    validator=lambda d, f=field_name, t=expected_type: 
                        isinstance(d.get(f), t) if f in d else True,
                    message=f"Field '{field_name}' must be of type {expected_type.__name__}"
                ))
            
            if expected_type in (int, float):
                min_val = field_schema.get('min')
                if min_val is not None:
                    self.add_rule(ValidationRule(
                        name=f"{field_name}_min",
                        validator=lambda d, f=field_name, m=min_val: 
                            d.get(f, m) >= m,
                        message=f"Field '{field_name}' must be >= {min_val}"
                    ))
                
                max_val = field_schema.get('max')
                if max_val is not None:
                    self.add_rule(ValidationRule(
                        name=f"{field_name}_max",
                        validator=lambda d, f=field_name, m=max_val: 
                            d.get(f, m) <= m,
                        message=f"Field '{field_name}' must be <= {max_val}"
                    ))
            
            elif expected_type == str:
                min_len = field_schema.get('min_length')
                if min_len is not None:
                    self.add_rule(ValidationRule(
                        name=f"{field_name}_min_length",
                        validator=lambda d, f=field_name, m=min_len: 
                            len(d.get(f, '')) >= m,
                        message=f"Field '{field_name}' must have at least {min_len} characters"
                    ))
                
                max_len = field_schema.get('max_length')
                if max_len is not None:
                    self.add_rule(ValidationRule(
                        name=f"{field_name}_max_length",
                        validator=lambda d, f=field_name, m=max_len: 
                            len(d.get(f, '')) <= m,
                        message=f"Field '{field_name}' must have at most {max_len} characters"
                    ))
                
                pattern = field_schema.get('pattern')
                if pattern:
                    regex = re.compile(pattern)
                    self.add_rule(ValidationRule(
                        name=f"{field_name}_pattern",
                        validator=lambda d, f=field_name, r=regex: 
                            r.match(d.get(f, '')) is not None,
                        message=f"Field '{field_name}' must match pattern {pattern}"
                    ))
            
            enum_values = field_schema.get('enum')
            if enum_values:
                self.add_rule(ValidationRule(
                    name=f"{field_name}_enum",
                    validator=lambda d, f=field_name, e=enum_values: 
                        d.get(f) in e,
                    message=f"Field '{field_name}' must be one of: {enum_values}"
                ))


class DataValidator:
    """Collection of static validation methods for common data types."""
    
    @staticmethod
    def email(email: str) -> bool:
        """Validate email address format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def url(url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?::\d+)?(?:/[-\w$.+!*\'(),;:@&=?/~#%]*)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def ip_address(ip: str) -> bool:
        """Validate IPv4 address format."""
        pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        return bool(re.match(pattern, ip))
    
    @staticmethod
    def phone_number(phone: str, country: str = "US") -> bool:
        """Validate phone number format for specified country."""
        patterns = {
            "US": r'^\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})$',
            "ID": r'^(\+62|62|0)8[1-9][0-9]{6,9}$',
        }
        pattern = patterns.get(country, patterns["US"])
        return bool(re.match(pattern, phone))
    
    @staticmethod
    def date_string(date_str: str, format: str = "%Y-%m-%d") -> bool:
        """Validate date string against specified format."""
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def json_string(json_str: str) -> bool:
        """Validate JSON string."""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    @staticmethod
    def xml_string(xml_str: str) -> bool:
        """Validate XML string."""
        try:
            ET.fromstring(xml_str)
            return True
        except ET.ParseError:
            return False
    
    @staticmethod
    def in_range(value: Union[int, float], min_val: Union[int, float], 
                max_val: Union[int, float]) -> bool:
        """Check if value is within range."""
        return min_val <= value <= max_val
    
    @staticmethod
    def not_empty(value: Any) -> bool:
        """Check if value is not empty."""
        if value is None:
            return False
        if isinstance(value, (str, list, dict, tuple)):
            return len(value) > 0
        return True


def validate_email(email: str) -> bool:
    """Convenience function to validate email."""
    return DataValidator.email(email)


def validate_url(url: str) -> bool:
    """Convenience function to validate URL."""
    return DataValidator.url(url)


def validate_ip_address(ip: str) -> bool:
    """Convenience function to validate IP address."""
    return DataValidator.ip_address(ip)


def validate_phone_number(phone: str, country: str = "US") -> bool:
    """Convenience function to validate phone number."""
    return DataValidator.phone_number(phone, country)


def validate_date_string(date_str: str, format: str = "%Y-%m-%d") -> bool:
    """Convenience function to validate date string."""
    return DataValidator.date_string(date_str, format)


def validate_json(json_str: str) -> bool:
    """Convenience function to validate JSON string."""
    return DataValidator.json_string(json_str)


def validate_xml(xml_str: str) -> bool:
    """Convenience function to validate XML string."""
    return DataValidator.xml_string(xml_str)