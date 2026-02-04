# cockatoo_v1/src/document_processing/extractors/markdown_extractor.py

"""
Markdown document extractor with support for CommonMark and GitHub Flavored Markdown.
"""

import os
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import re

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

try:
    import markdown
    from markdown.extensions import Extension
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    logger.warning("markdown library not installed. Markdown parsing will be limited.")

try:
    import frontmatter
    HAS_FRONTMATTER = True
except ImportError:
    HAS_FRONTMATTER = False
    logger.warning("python-frontmatter not installed. Front matter parsing will be limited.")


class MarkdownExtractor(BaseExtractor):
    """
    Markdown document extractor with front matter support.
    """
    
    def __init__(self, use_extensions: bool = True):
        """
        Initialize Markdown extractor.
        
        Args:
            use_extensions: Whether to use markdown extensions
        """
        super().__init__()
        self.use_extensions = use_extensions
        
        if not HAS_MARKDOWN:
            self.logger.warning(
                "markdown library is not installed. Install with: pip install markdown"
            )
        
        if not HAS_FRONTMATTER:
            self.logger.warning(
                "python-frontmatter is not installed. Front matter will not be parsed. "
                "Install with: pip install python-frontmatter"
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.md', '.markdown', '.mdown', '.mkd', '.mkdn', '.mdwn', '.mdt', '.mdtext']
    
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text and metadata from Markdown file.
        
        Args:
            file_path: Path to Markdown file
        
        Returns:
            Dictionary containing:
                - text: Extracted text content (HTML and plain text)
                - metadata: Markdown metadata
                - front_matter: Front matter data
                - headings: List of headings
                - links: List of links
                - code_blocks: List of code blocks
                - images: List of images
                - html: Generated HTML (if markdown library available)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        logger.info(f"Extracting Markdown: {file_path}")
        
        result = {
            "text": "",
            "html": "",
            "metadata": self.get_basic_metadata(file_path),
            "front_matter": {},
            "headings": [],
            "links": [],
            "code_blocks": [],
            "images": [],
            "tables": [],
            "lists": [],
            "blocks": [],
            "raw_content": "",
            "has_front_matter": False,
            "extraction_method": "markdown_lib" if HAS_MARKDOWN else "regex",
        }
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            content = ""
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except:
                    continue
        
        if not content:
            result["warnings"] = ["Failed to read Markdown file"]
            return result
        
        result["raw_content"] = content
        
        # Parse front matter if available
        if HAS_FRONTMATTER:
            try:
                parsed = frontmatter.loads(content)
                result["front_matter"] = parsed.metadata
                content = parsed.content
                result["has_front_matter"] = bool(parsed.metadata)
            except:
                pass  # No front matter or parsing failed
        
        # Extract using markdown library if available
        if HAS_MARKDOWN:
            try:
                result = self._extract_with_markdown_lib(content, result)
            except Exception as e:
                logger.error(f"Markdown library extraction failed: {e}")
                result = self._extract_with_regex(content, result)
        else:
            result = self._extract_with_regex(content, result)
        
        # Clean and post-process
        result["text"] = self.clean_text(result["text"])
        
        # Add language detection and summary
        if result["text"]:
            result["metadata"]["language"] = self.detect_language(result["text"])
            result["metadata"]["summary"] = self.extract_summary(result["text"])
        
        # Merge front matter into metadata
        if result["front_matter"]:
            result["metadata"].update(result["front_matter"])
        
        return result
    
    def _extract_with_markdown_lib(self, content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract using markdown library.
        
        Args:
            content: Markdown content
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        # Configure markdown extensions
        extensions = [
            'extra',
            'codehilite',
            'toc',
            'tables',
            'fenced_code',
            'footnotes',
            'attr_list',
            'def_list',
            'abbr',
            'md_in_html',
        ]
        
        # Convert markdown to HTML
        html = markdown.markdown(content, extensions=extensions, output_format='html5')
        result["html"] = html
        
        # Extract plain text from HTML
        text = self._html_to_text(html)
        result["text"] = text
        
        # Parse markdown structure
        result = self._parse_markdown_structure(content, result)
        
        return result
    
    def _extract_with_regex(self, content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback extraction using regex patterns.
        
        Args:
            content: Markdown content
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        # Extract plain text (remove markdown syntax)
        text = self._strip_markdown(content)
        result["text"] = text
        
        # Parse basic structure with regex
        result = self._parse_markdown_with_regex(content, result)
        
        return result
    
    def _html_to_text(self, html: str) -> str:
        """
        Convert HTML to plain text.
        
        Args:
            html: HTML content
            
        Returns:
            Plain text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Convert HTML entities
        import html as html_lib
        text = html_lib.unescape(text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _strip_markdown(self, markdown_text: str) -> str:
        """
        Strip markdown syntax to get plain text.
        
        Args:
            markdown_text: Markdown content
            
        Returns:
            Plain text
        """
        # Remove code blocks
        text = re.sub(r'```.*?```', '', markdown_text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        # Remove images
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        
        # Remove links (keep link text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove emphasis
        text = re.sub(r'(\*{1,3}|_{1,3})(.*?)\1', r'\2', text)
        
        # Remove blockquotes
        text = re.sub(r'^\s*>+', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Clean up
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _parse_markdown_structure(self, content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse markdown structure.
        
        Args:
            content: Markdown content
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        lines = content.split('\n')
        
        headings = []
        links = []
        code_blocks = []
        images = []
        tables = []
        lists = []
        blocks = []
        
        current_block = []
        in_code_block = False
        in_table = False
        code_language = ""
        table_rows = []
        
        for i, line in enumerate(lines):
            line = line.rstrip('\n')
            
            # Detect code blocks
            if line.startswith('```'):
                if in_code_block:
                    # End of code block
                    code_content = '\n'.join(current_block)
                    code_blocks.append({
                        "language": code_language,
                        "content": code_content,
                        "start_line": code_start_line,
                        "end_line": i,
                        "line_count": i - code_start_line - 1,
                    })
                    current_block = []
                    in_code_block = False
                    code_language = ""
                else:
                    # Start of code block
                    in_code_block = True
                    code_start_line = i
                    code_language = line[3:].strip() or "unknown"
                continue
            
            if in_code_block:
                current_block.append(line)
                continue
            
            # Detect headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                
                # Remove any trailing # symbols
                text = re.sub(r'\s*#+$', '', text)
                
                # Extract ID if present
                heading_id = ""
                if '{#' in text and '}' in text:
                    id_match = re.search(r'\{#([^}]+)\}', text)
                    if id_match:
                        heading_id = id_match.group(1)
                        text = text.replace(id_match.group(0), '').strip()
                
                headings.append({
                    "level": level,
                    "text": text,
                    "id": heading_id or self._slugify(text),
                    "line_number": i + 1,
                })
            
            # Detect links
            link_matches = re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line)
            for match in link_matches:
                links.append({
                    "text": match.group(1),
                    "url": match.group(2),
                    "line_number": i + 1,
                })
            
            # Detect images
            image_matches = re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            for match in image_matches:
                images.append({
                    "alt_text": match.group(1),
                    "url": match.group(2),
                    "line_number": i + 1,
                })
            
            # Detect tables
            if '|' in line and re.search(r'\|\s*-', line):
                # Table separator line
                in_table = True
                continue
            
            if in_table and '|' in line:
                # Table row
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                table_rows.append(cells)
            elif in_table and table_rows:
                # End of table
                if len(table_rows) > 1:
                    headers = table_rows[0]
                    data = table_rows[1:]
                    
                    tables.append({
                        "headers": headers,
                        "rows": data,
                        "row_count": len(data),
                        "column_count": len(headers),
                        "start_line": i - len(table_rows),
                        "end_line": i - 1,
                    })
                
                table_rows = []
                in_table = False
            
            # Detect lists
            list_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', line)
            if list_match:
                indent = len(list_match.group(1))
                marker = list_match.group(2)
                text = list_match.group(3)
                
                is_ordered = marker.endswith('.')
                
                lists.append({
                    "indent": indent,
                    "ordered": is_ordered,
                    "marker": marker,
                    "text": text,
                    "line_number": i + 1,
                })
        
        result.update({
            "headings": headings,
            "links": links,
            "code_blocks": code_blocks,
            "images": images,
            "tables": tables,
            "lists": lists,
        })
        
        return result
    
    def _parse_markdown_with_regex(self, content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse markdown structure using regex only.
        
        Args:
            content: Markdown content
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        headings = []
        links = []
        code_blocks = []
        images = []
        
        # Extract headings
        heading_pattern = r'^(#{1,6})\s+(.+?)(?:\s*#+)?$'
        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            
            headings.append({
                "level": level,
                "text": text,
                "id": self._slugify(text),
            })
        
        # Extract links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            links.append({
                "text": match.group(1),
                "url": match.group(2),
            })
        
        # Extract images
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(image_pattern, content):
            images.append({
                "alt_text": match.group(1),
                "url": match.group(2),
            })
        
        # Extract code blocks
        code_pattern = r'```(\w+)?\n(.*?)```'
        for match in re.finditer(code_pattern, content, re.DOTALL):
            code_blocks.append({
                "language": match.group(1) or "unknown",
                "content": match.group(2).strip(),
            })
        
        # Also match inline code
        inline_code_pattern = r'`([^`]+)`'
        for match in re.finditer(inline_code_pattern, content):
            code_blocks.append({
                "language": "inline",
                "content": match.group(1),
            })
        
        result.update({
            "headings": headings,
            "links": links,
            "code_blocks": code_blocks,
            "images": images,
        })
        
        return result
    
    def _slugify(self, text: str) -> str:
        """
        Convert text to URL-friendly slug.
        
        Args:
            text: Text to slugify
            
        Returns:
            Slugified text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace spaces and special characters
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        
        # Remove leading/trailing hyphens
        text = text.strip('-')
        
        return text
    
    def extract_toc(self, file_path: Union[str, Path], max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Extract table of contents from markdown.
        
        Args:
            file_path: Path to markdown file
            max_depth: Maximum heading depth to include
            
        Returns:
            Table of contents
        """
        result = self.extract(file_path)
        
        toc = []
        for heading in result.get("headings", []):
            if heading["level"] <= max_depth:
                toc.append({
                    "level": heading["level"],
                    "text": heading["text"],
                    "id": heading.get("id", self._slugify(heading["text"])),
                })
        
        return toc
    
    def validate_markdown(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate markdown syntax and structure.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            Validation results
        """
        result = self.extract(file_path)
        
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "stats": {
                "headings": len(result.get("headings", [])),
                "links": len(result.get("links", [])),
                "images": len(result.get("images", [])),
                "code_blocks": len(result.get("code_blocks", [])),
                "tables": len(result.get("tables", [])),
                "lists": len(result.get("lists", [])),
                "word_count": len(result.get("text", "").split()),
                "char_count": len(result.get("text", "")),
            },
        }
        
        # Check for common issues
        headings = result.get("headings", [])
        
        # Check heading hierarchy
        heading_levels = [h["level"] for h in headings]
        if heading_levels:
            if min(heading_levels) > 1:
                validation["warnings"].append("Document starts with H2 or deeper heading")
            
            # Check for skipped levels
            for i in range(len(heading_levels) - 1):
                if heading_levels[i + 1] > heading_levels[i] + 1:
                    validation["warnings"].append(
                        f"Heading level jumps from H{heading_levels[i]} to H{heading_levels[i + 1]}"
                    )
        
        # Check for broken links (simplified)
        links = result.get("links", [])
        for link in links:
            url = link.get("url", "")
            if url.startswith('http'):
                # External link - could check if valid
                pass
            elif url.startswith('#') and url != '#':
                # Internal link - check if target exists
                target_id = url[1:]
                if not any(h.get("id") == target_id for h in headings):
                    validation["warnings"].append(f"Broken internal link: {url}")
        
        # Check image alt text
        images = result.get("images", [])
        for image in images:
            if not image.get("alt_text"):
                validation["warnings"].append("Image missing alt text")
        
        return validation