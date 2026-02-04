# cockatoo_v1/src/document_processing/extractors/html_extractor.py

"""
HTML document extractor with BeautifulSoup.
"""

import os
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import re
import urllib.parse
from html import unescape

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    logger.warning("BeautifulSoup4 not installed. HTML parsing will be limited.")


class HTMLExtractor(BaseExtractor):
    """
    HTML document extractor.
    """
    
    def __init__(self):
        """Initialize HTML extractor."""
        super().__init__()
        if not HAS_BEAUTIFULSOUP:
            self.logger.warning(
                "BeautifulSoup4 is not installed. Install with: pip install beautifulsoup4"
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.html', '.htm', '.xhtml', '.shtml', '.php', '.asp', '.jsp']
    
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text and metadata from HTML file.
        
        Args:
            file_path: Path to HTML file
        
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: HTML metadata
                - title: Page title
                - headings: List of headings
                - links: List of hyperlinks
                - images: List of images
                - scripts: List of scripts
                - styles: List of styles
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {file_path}")
        
        logger.info(f"Extracting HTML: {file_path}")
        
        result = {
            "text": "",
            "metadata": self.get_basic_metadata(file_path),
            "title": "",
            "headings": [],
            "links": [],
            "images": [],
            "scripts": [],
            "styles": [],
            "tables": [],
            "forms": [],
            "structure": {},
            "extraction_method": "beautifulsoup" if HAS_BEAUTIFULSOUP else "regex",
        }
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            html_content = ""
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        html_content = f.read()
                    break
                except:
                    continue
        
        if not html_content:
            result["warnings"] = ["Failed to read HTML file"]
            return result
        
        if HAS_BEAUTIFULSOUP:
            try:
                result = self._extract_with_bs4(html_content, result)
            except Exception as e:
                logger.error(f"BeautifulSoup extraction failed: {e}")
                result = self._extract_with_regex(html_content, result)
        else:
            result = self._extract_with_regex(html_content, result)
        
        # Clean and post-process
        result["text"] = self.clean_text(result["text"])
        
        # Add language detection and summary
        if result["text"]:
            result["metadata"]["language"] = self.detect_language(result["text"])
            result["metadata"]["summary"] = self.extract_summary(result["text"])
        
        return result
    
    def _extract_with_bs4(self, html_content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract using BeautifulSoup.
        
        Args:
            html_content: HTML content as string
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        result["title"] = title_tag.get_text(strip=True) if title_tag else ""
        
        # Extract metadata
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                metadata[name.lower()] = content
        
        # Additional metadata from common tags
        metadata_tags = {
            'description': soup.find('meta', {'name': 'description'}),
            'keywords': soup.find('meta', {'name': 'keywords'}),
            'author': soup.find('meta', {'name': 'author'}),
            'viewport': soup.find('meta', {'name': 'viewport'}),
            'charset': soup.find('meta', {'charset': True}),
        }
        
        for key, tag in metadata_tags.items():
            if tag:
                if key == 'charset':
                    metadata[key] = tag.get('charset')
                else:
                    metadata[key] = tag.get('content', '')
        
        result["metadata"].update(metadata)
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        result["text"] = unescape(text)  # Convert HTML entities
        
        # Extract headings
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    "level": i,
                    "text": heading.get_text(strip=True),
                    "id": heading.get('id', ''),
                })
        result["headings"] = headings
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            if link_text or link.get('href'):
                links.append({
                    "text": link_text[:100],  # Limit text length
                    "href": link['href'],
                    "title": link.get('title', ''),
                    "target": link.get('target', ''),
                })
        result["links"] = links
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            images.append({
                "src": img['src'],
                "alt": img.get('alt', ''),
                "title": img.get('title', ''),
                "width": img.get('width', ''),
                "height": img.get('height', ''),
            })
        result["images"] = images
        
        # Extract scripts
        scripts = []
        for script in soup.find_all('script', src=True):
            scripts.append({
                "src": script['src'],
                "type": script.get('type', ''),
                "async": script.get('async', False),
                "defer": script.get('defer', False),
            })
        result["scripts"] = scripts
        
        # Extract styles
        styles = []
        for style in soup.find_all('link', rel='stylesheet'):
            styles.append({
                "href": style.get('href', ''),
                "media": style.get('media', 'all'),
            })
        
        # Inline styles
        for style in soup.find_all('style'):
            styles.append({
                "inline": True,
                "content": style.get_text()[:500],  # Limit content length
            })
        
        result["styles"] = styles
        
        # Extract tables
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    cell_text = cell.get_text(strip=True)
                    row_data.append(cell_text)
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append({
                    "rows": len(table_data),
                    "columns": len(table_data[0]) if table_data else 0,
                    "data": table_data,
                    "caption": table.find('caption').get_text(strip=True) if table.find('caption') else "",
                })
        
        result["tables"] = tables
        
        # Extract forms
        forms = []
        for form in soup.find_all('form'):
            form_info = {
                "action": form.get('action', ''),
                "method": form.get('method', 'get'),
                "id": form.get('id', ''),
                "name": form.get('name', ''),
                "inputs": [],
            }
            
            for input_elem in form.find_all(['input', 'textarea', 'select']):
                input_info = {
                    "type": input_elem.name,
                    "name": input_elem.get('name', ''),
                    "id": input_elem.get('id', ''),
                    "placeholder": input_elem.get('placeholder', ''),
                }
                
                if input_elem.name == 'input':
                    input_info["type"] = input_elem.get('type', 'text')
                    input_info["value"] = input_elem.get('value', '')
                elif input_elem.name == 'textarea':
                    input_info["value"] = input_elem.get_text(strip=True)
                
                form_info["inputs"].append(input_info)
            
            forms.append(form_info)
        
        result["forms"] = forms
        
        # Document structure
        result["structure"] = {
            "doctype": soup.find('!doctype'),
            "html_tag": {
                "lang": soup.html.get('lang') if soup.html else '',
            },
            "body_classes": soup.body.get('class') if soup.body and soup.body.get('class') else [],
            "element_counts": {
                "div": len(soup.find_all('div')),
                "p": len(soup.find_all('p')),
                "span": len(soup.find_all('span')),
                "img": len(soup.find_all('img')),
                "a": len(soup.find_all('a')),
                "table": len(soup.find_all('table')),
                "form": len(soup.find_all('form')),
            }
        }
        
        return result
    
    def _extract_with_regex(self, html_content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback extraction using regex patterns.
        
        Args:
            html_content: HTML content as string
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            result["title"] = self._clean_html_text(title_match.group(1))
        
        # Extract meta tags
        meta_matches = re.finditer(r'<meta[^>]+>', html_content, re.IGNORECASE)
        metadata = {}
        
        for match in meta_matches:
            meta_tag = match.group()
            # Extract name/content pairs
            name_match = re.search(r'name=["\']([^"\']+)["\']', meta_tag, re.IGNORECASE)
            property_match = re.search(r'property=["\']([^"\']+)["\']', meta_tag, re.IGNORECASE)
            content_match = re.search(r'content=["\']([^"\']+)["\']', meta_tag, re.IGNORECASE)
            
            name = None
            if name_match:
                name = name_match.group(1).lower()
            elif property_match:
                name = property_match.group(1).lower()
            
            if name and content_match:
                metadata[name] = content_match.group(1)
        
        result["metadata"].update(metadata)
        
        # Remove script and style tags
        cleaned_html = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.IGNORECASE | re.DOTALL)
        cleaned_html = re.sub(r'<style[^>]*>.*?</style>', ' ', cleaned_html, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', ' ', cleaned_html)
        
        # Convert HTML entities and clean up
        text = unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        result["text"] = text
        
        # Extract headings (simplified)
        headings = []
        for i in range(1, 7):
            pattern = f'<h{i}[^>]*>(.*?)</h{i}>'
            for match in re.finditer(pattern, html_content, re.IGNORECASE | re.DOTALL):
                heading_text = self._clean_html_text(match.group(1))
                if heading_text:
                    headings.append({
                        "level": i,
                        "text": heading_text,
                    })
        
        result["headings"] = headings
        
        # Extract links (simplified)
        links = []
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
        for match in re.finditer(link_pattern, html_content, re.IGNORECASE | re.DOTALL):
            link_text = self._clean_html_text(match.group(2))
            if link_text or match.group(1):
                links.append({
                    "text": link_text[:100],
                    "href": match.group(1),
                })
        
        result["links"] = links
        
        return result
    
    def _clean_html_text(self, text: str) -> str:
        """
        Clean HTML text by removing tags and entities.
        
        Args:
            text: HTML text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Convert HTML entities
        text = unescape(text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_seo_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract SEO information from HTML.
        
        Args:
            file_path: Path to HTML file
            
        Returns:
            SEO information
        """
        result = self.extract(file_path)
        
        seo_info = {
            "basic": {
                "title": result.get("title", ""),
                "title_length": len(result.get("title", "")),
                "description": result.get("metadata", {}).get("description", ""),
                "description_length": len(result.get("metadata", {}).get("description", "")),
                "keywords": result.get("metadata", {}).get("keywords", ""),
                "language": result.get("metadata", {}).get("language", ""),
            },
            "headings": {
                "h1_count": len([h for h in result.get("headings", []) if h.get("level") == 1]),
                "h2_count": len([h for h in result.get("headings", []) if h.get("level") == 2]),
                "h3_count": len([h for h in result.get("headings", []) if h.get("level") == 3]),
                "headings": result.get("headings", []),
            },
            "links": {
                "internal_count": 0,
                "external_count": 0,
                "total_count": len(result.get("links", [])),
            },
            "images": {
                "total_count": len(result.get("images", [])),
                "with_alt": len([img for img in result.get("images", []) if img.get("alt")]),
                "without_alt": len([img for img in result.get("images", []) if not img.get("alt")]),
            },
            "content": {
                "word_count": len(result.get("text", "").split()),
                "char_count": len(result.get("text", "")),
                "text_sample": result.get("text", "")[:500] + "..." if len(result.get("text", "")) > 500 else result.get("text", ""),
            },
        }
        
        # Categorize links
        for link in result.get("links", []):
            href = link.get("href", "")
            if href:
                if href.startswith(('#', '?')) or href.startswith('/'):
                    seo_info["links"]["internal_count"] += 1
                elif href.startswith(('http://', 'https://')):
                    seo_info["links"]["external_count"] += 1
        
        # SEO recommendations
        recommendations = []
        
        if not seo_info["basic"]["title"]:
            recommendations.append("Add a title tag")
        elif seo_info["basic"]["title_length"] < 30:
            recommendations.append("Title is too short (aim for 50-60 characters)")
        elif seo_info["basic"]["title_length"] > 70:
            recommendations.append("Title is too long (aim for 50-60 characters)")
        
        if not seo_info["basic"]["description"]:
            recommendations.append("Add a meta description")
        elif seo_info["basic"]["description_length"] < 120:
            recommendations.append("Description is too short (aim for 150-160 characters)")
        elif seo_info["basic"]["description_length"] > 170:
            recommendations.append("Description is too long (aim for 150-160 characters)")
        
        if seo_info["headings"]["h1_count"] == 0:
            recommendations.append("Add at least one H1 heading")
        elif seo_info["headings"]["h1_count"] > 1:
            recommendations.append("Consider having only one H1 heading")
        
        if seo_info["images"]["without_alt"] > 0:
            recommendations.append(f"Add alt text to {seo_info['images']['without_alt']} images")
        
        seo_info["recommendations"] = recommendations
        
        return seo_info