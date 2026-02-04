# cockatoo_v1/src/document_processing/extractors/web_extractor.py

"""
Web content extractor for downloading and processing web pages and articles.
"""

import os
import logging
import re
import urllib.parse
from typing import Dict, Any, List, Union, Optional, Tuple
from pathlib import Path
import tempfile
import time
from datetime import datetime

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

# Try to import web scraping libraries
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests library not installed. Web scraping will be limited.")

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    logger.warning("BeautifulSoup4 not installed. HTML parsing will be limited.")

try:
    import readability
    from readability import Document
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False
    logger.warning("readability-lxml not installed. Article extraction will be limited.")

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    logger.warning("trafilatura not installed. Alternative article extraction will be limited.")


class WebExtractor(BaseExtractor):
    """
    Web content extractor for downloading and processing web pages.
    Supports article extraction, metadata parsing, and content cleaning.
    """
    
    def __init__(self, 
                 timeout: int = 30,
                 max_retries: int = 3,
                 user_agent: str = None,
                 use_readability: bool = True,
                 use_trafilatura: bool = True):
        """
        Initialize web extractor.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            user_agent: User agent string for requests
            use_readability: Use readability-lxml for article extraction
            use_trafilatura: Use trafilatura as fallback article extractor
        """
        super().__init__()
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        self.use_readability = use_readability and HAS_READABILITY
        self.use_trafilatura = use_trafilatura and HAS_TRAFILATURA
        
        if not HAS_REQUESTS:
            self.logger.warning(
                "requests library is not installed. Install with: pip install requests"
            )
        
        if not HAS_BEAUTIFULSOUP:
            self.logger.warning(
                "BeautifulSoup4 is not installed. Install with: pip install beautifulsoup4"
            )
        
        if use_readability and not HAS_READABILITY:
            self.logger.warning(
                "readability-lxml is not installed. Article extraction will be limited. "
                "Install with: pip install readability-lxml"
            )
        
        if use_trafilatura and not HAS_TRAFILATURA:
            self.logger.warning(
                "trafilatura is not installed. Alternative article extraction will be limited. "
                "Install with: pip install trafilatura"
            )
        
        # Create session with retry logic
        self.session = None
        if HAS_REQUESTS:
            self.session = self._create_session()
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported protocols/URL schemes
        """
        # Note: This extractor doesn't process local files, but URLs
        return ['http://', 'https://', 'www.']
    
    def extract(self, source: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract content from web URL or HTML file.
        
        Args:
            source: URL string or Path to HTML file
        
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: Web page metadata
                - html: Raw HTML content
                - article: Extracted article content (if applicable)
                - links: List of links on page
                - images: List of images
                - stats: Page statistics
                - download_info: Download information
        """
        result = {
            "text": "",
            "metadata": {},
            "html": "",
            "article": {},
            "links": [],
            "images": [],
            "stats": {},
            "download_info": {},
            "validation": {
                "is_valid": False,
                "errors": [],
                "warnings": [],
            },
            "extraction_method": "web_scraping",
        }
        
        # Determine if source is URL or file path
        source_str = str(source)
        
        if self._is_url(source_str):
            # Web URL
            result = self._extract_from_url(source_str, result)
        else:
            # Local HTML file
            file_path = Path(source)
            if file_path.exists():
                result = self._extract_from_file(file_path, result)
            else:
                result["validation"]["errors"].append(f"File not found: {source}")
        
        # Clean and post-process
        if result["text"]:
            result["text"] = self.clean_text(result["text"])
            result["metadata"]["language"] = self.detect_language(result["text"])
            result["metadata"]["summary"] = self.extract_summary(result["text"])
        
        return result
    
    def _is_url(self, source: str) -> bool:
        """
        Check if source is a valid URL.
        
        Args:
            source: Source string
            
        Returns:
            True if source appears to be a URL
        """
        source_lower = source.lower()
        return (source_lower.startswith('http://') or 
                source_lower.startswith('https://') or
                source_lower.startswith('www.'))
    
    def _create_session(self):
        """
        Create HTTP session with retry logic.
        
        Returns:
            requests.Session object
        """
        if not HAS_REQUESTS:
            return None
        
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    def _extract_from_url(self, url: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from URL.
        
        Args:
            url: Web URL
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        if not HAS_REQUESTS:
            result["validation"]["errors"].append("requests library required for URL extraction")
            return result
        
        # Normalize URL
        url = self._normalize_url(url)
        
        download_info = {
            "url": url,
            "start_time": datetime.now().isoformat(),
            "success": False,
            "status_code": None,
            "content_type": None,
            "content_length": 0,
            "download_time": 0,
        }
        
        try:
            start_time = time.time()
            
            # Make request
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                stream=False
            )
            
            download_time = time.time() - start_time
            
            download_info.update({
                "success": True,
                "status_code": response.status_code,
                "content_type": response.headers.get('content-type', ''),
                "content_length": len(response.content),
                "download_time": download_time,
                "final_url": response.url,
                "encoding": response.encoding,
                "headers": dict(response.headers),
            })
            
            # Check if successful
            if response.status_code != 200:
                result["validation"]["errors"].append(
                    f"HTTP {response.status_code}: {response.reason}"
                )
                result["download_info"] = download_info
                return result
            
            # Get HTML content
            html_content = response.text
            
            # Update result with HTML
            result["html"] = html_content
            result["download_info"] = download_info
            result["validation"]["is_valid"] = True
            
            # Parse HTML
            result = self._parse_html_content(html_content, url, result)
            
        except requests.exceptions.Timeout:
            result["validation"]["errors"].append(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError as e:
            result["validation"]["errors"].append(f"Connection error: {e}")
        except requests.exceptions.TooManyRedirects:
            result["validation"]["errors"].append("Too many redirects")
        except requests.exceptions.RequestException as e:
            result["validation"]["errors"].append(f"Request failed: {e}")
        except Exception as e:
            result["validation"]["errors"].append(f"Unexpected error: {e}")
        
        return result
    
    def _extract_from_file(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from local HTML file.
        
        Args:
            file_path: Path to HTML file
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            result["html"] = html_content
            result["validation"]["is_valid"] = True
            
            # Parse HTML
            result = self._parse_html_content(html_content, str(file_path), result)
            
            # Add file metadata
            result["metadata"].update(self.get_basic_metadata(file_path))
            
        except Exception as e:
            result["validation"]["errors"].append(f"Failed to read file: {e}")
        
        return result
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL.
        
        Args:
            url: Input URL
            
        Returns:
            Normalized URL
        """
        url = url.strip()
        
        # Add scheme if missing
        if url.startswith('www.'):
            url = 'https://' + url
        
        # Ensure URL is properly formatted
        parsed = urllib.parse.urlparse(url)
        
        # Reconstruct URL with proper components
        normalized = urllib.parse.urlunparse((
            parsed.scheme or 'https',
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        
        return normalized
    
    def _parse_html_content(self, html_content: str, source: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse HTML content and extract information.
        
        Args:
            html_content: HTML content as string
            source: Source URL or file path
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        # Extract using multiple methods
        methods_used = []
        
        # Method 1: Use readability for article extraction
        article_content = None
        if self.use_readability:
            try:
                article_content = self._extract_with_readability(html_content)
                if article_content and article_content.get("text"):
                    methods_used.append("readability")
            except Exception as e:
                logger.warning(f"Readability extraction failed: {e}")
        
        # Method 2: Use trafilatura as fallback
        if not article_content and self.use_trafilatura:
            try:
                article_content = self._extract_with_trafilatura(html_content)
                if article_content and article_content.get("text"):
                    methods_used.append("trafilatura")
            except Exception as e:
                logger.warning(f"Trafilatura extraction failed: {e}")
        
        # Method 3: Use BeautifulSoup for general parsing
        soup_content = None
        if HAS_BEAUTIFULSOUP:
            try:
                soup_content = self._extract_with_bs4(html_content, source)
                methods_used.append("beautifulsoup")
            except Exception as e:
                logger.warning(f"BeautifulSoup extraction failed: {e}")
        
        # Combine results
        if article_content:
            result["article"] = article_content
            result["text"] = article_content.get("text", "")
            result["metadata"].update(article_content.get("metadata", {}))
        
        if soup_content:
            # Use soup content if no article content was extracted
            if not result["text"]:
                result["text"] = soup_content.get("text", "")
            
            # Merge metadata
            result["metadata"].update(soup_content.get("metadata", {}))
            
            # Add links and images
            result["links"] = soup_content.get("links", [])
            result["images"] = soup_content.get("images", [])
        
        # Calculate statistics
        result["stats"] = {
            "html_size": len(html_content),
            "text_size": len(result["text"]),
            "compression_ratio": len(result["text"]) / len(html_content) if html_content else 0,
            "word_count": len(result["text"].split()),
            "link_count": len(result.get("links", [])),
            "image_count": len(result.get("images", [])),
            "extraction_methods": methods_used,
        }
        
        # Add source to metadata
        result["metadata"]["source"] = source
        
        return result
    
    def _extract_with_readability(self, html_content: str) -> Dict[str, Any]:
        """
        Extract article content using readability-lxml.
        
        Args:
            html_content: HTML content
            
        Returns:
            Article content and metadata
        """
        doc = Document(html_content)
        
        article_info = {
            "text": doc.summary(),
            "metadata": {
                "title": doc.title(),
                "short_title": doc.short_title(),
            }
        }
        
        return article_info
    
    def _extract_with_trafilatura(self, html_content: str) -> Dict[str, Any]:
        """
        Extract article content using trafilatura.
        
        Args:
            html_content: HTML content
            
        Returns:
            Article content and metadata
        """
        extracted = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            output_format='json'
        )
        
        if extracted:
            import json
            data = json.loads(extracted)
            
            article_info = {
                "text": data.get("text", ""),
                "metadata": {
                    "title": data.get("title", ""),
                    "author": data.get("author", ""),
                    "url": data.get("url", ""),
                    "hostname": data.get("hostname", ""),
                    "description": data.get("description", ""),
                    "sitename": data.get("sitename", ""),
                    "date": data.get("date", ""),
                    "categories": data.get("categories", ""),
                    "tags": data.get("tags", ""),
                    "fingerprint": data.get("fingerprint", ""),
                }
            }
            
            return article_info
        
        return {}
    
    def _extract_with_bs4(self, html_content: str, source: str) -> Dict[str, Any]:
        """
        Extract content using BeautifulSoup.
        
        Args:
            html_content: HTML content
            source: Source URL or file path
            
        Returns:
            Extracted content and metadata
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract metadata
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        metadata['title'] = title_tag.get_text(strip=True) if title_tag else ""
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                metadata[name.lower()] = content
        
        # Common metadata
        common_meta = {
            'description': soup.find('meta', {'name': 'description'}),
            'keywords': soup.find('meta', {'name': 'keywords'}),
            'author': soup.find('meta', {'name': 'author'}),
            'viewport': soup.find('meta', {'name': 'viewport'}),
            'og:title': soup.find('meta', {'property': 'og:title'}),
            'og:description': soup.find('meta', {'property': 'og:description'}),
            'og:image': soup.find('meta', {'property': 'og:image'}),
            'twitter:title': soup.find('meta', {'name': 'twitter:title'}),
            'twitter:description': soup.find('meta', {'name': 'twitter:description'}),
        }
        
        for key, tag in common_meta.items():
            if tag:
                metadata[key] = tag.get('content', '')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            href = link['href']
            
            # Make relative URLs absolute
            if not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                if self._is_url(source):
                    href = urllib.parse.urljoin(source, href)
            
            if link_text or href:
                links.append({
                    "text": link_text[:200],  # Limit text length
                    "url": href,
                    "title": link.get('title', ''),
                })
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            
            # Make relative URLs absolute
            if not src.startswith(('http://', 'https://', 'data:')):
                if self._is_url(source):
                    src = urllib.parse.urljoin(source, src)
            
            images.append({
                "src": src,
                "alt": img.get('alt', ''),
                "title": img.get('title', ''),
                "width": img.get('width', ''),
                "height": img.get('height', ''),
            })
        
        # Extract headings
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    "level": i,
                    "text": heading.get_text(strip=True),
                })
        
        return {
            "text": text,
            "metadata": metadata,
            "links": links,
            "images": images,
            "headings": headings,
        }
    
    def download_and_save(self, url: str, output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Download web page and save as HTML file.
        
        Args:
            url: Web URL to download
            output_dir: Directory to save the file
            
        Returns:
            Download information
        """
        if not HAS_REQUESTS:
            return {"error": "requests library required for download"}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "success": False,
            "url": url,
            "saved_path": "",
            "error": "",
            "download_info": {},
        }
        
        try:
            # Extract content
            extract_result = self.extract(url)
            
            if not extract_result["validation"]["is_valid"]:
                result["error"] = extract_result["validation"]["errors"][0] if extract_result["validation"]["errors"] else "Unknown error"
                return result
            
            # Generate filename from URL
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc.replace('www.', '')
            path = parsed_url.path.strip('/').replace('/', '_')
            
            if not path:
                path = "index"
            
            # Clean filename
            filename = f"{domain}_{path}.html"
            filename = re.sub(r'[^\w\-\.]', '_', filename)
            filename = filename[:100]  # Limit length
            
            # Save HTML file
            save_path = output_dir / filename
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(extract_result["html"])
            
            # Save metadata as JSON
            metadata_path = output_dir / f"{filename}.metadata.json"
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "url": url,
                    "title": extract_result["metadata"].get("title", ""),
                    "download_time": datetime.now().isoformat(),
                    "stats": extract_result["stats"],
                }, f, indent=2)
            
            result.update({
                "success": True,
                "saved_path": str(save_path),
                "metadata_path": str(metadata_path),
                "download_info": extract_result.get("download_info", {}),
            })
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def extract_news_article(self, url: str) -> Dict[str, Any]:
        """
        Specialized extraction for news articles.
        
        Args:
            url: News article URL
            
        Returns:
            Article information with enhanced metadata
        """
        result = self.extract(url)
        
        if not result["validation"]["is_valid"]:
            return result
        
        # Enhance article detection
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        
        # Try to extract date
        if not metadata.get("date"):
            date_patterns = [
                r'(\d{4}[-/]\d{2}[-/]\d{2})',
                r'(\d{2}[-/]\d{2}[-/]\d{4})',
                r'(\w+ \d{1,2}, \d{4})',
                r'(\d{1,2} \w+ \d{4})',
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text[:5000])
                if match:
                    metadata["extracted_date"] = match.group(1)
                    break
        
        # Try to extract author if not found
        if not metadata.get("author"):
            # Common author patterns
            author_patterns = [
                r'By\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'Written by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'Author:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            ]
            
            for pattern in author_patterns:
                match = re.search(pattern, text[:2000], re.IGNORECASE)
                if match:
                    metadata["extracted_author"] = match.group(1).strip()
                    break
        
        # Classify content type
        content_type = "unknown"
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['news', 'report', 'update', 'breaking']):
            content_type = "news"
        elif any(word in text_lower for word in ['blog', 'post', 'article']):
            content_type = "blog"
        elif any(word in text_lower for word in ['tutorial', 'guide', 'how-to', 'step by step']):
            content_type = "tutorial"
        elif any(word in text_lower for word in ['review', 'rating', 'score']):
            content_type = "review"
        
        metadata["content_type"] = content_type
        
        # Calculate reading time (words per minute = 200)
        word_count = len(text.split())
        reading_time = max(1, word_count // 200)
        metadata["reading_time_minutes"] = reading_time
        
        # Update result
        result["metadata"] = metadata
        
        return result
    
    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate URL and check accessibility.
        
        Args:
            url: URL to validate
            
        Returns:
            Validation results
        """
        if not HAS_REQUESTS:
            return {"is_valid": False, "error": "requests library required"}
        
        validation = {
            "is_valid": False,
            "url": url,
            "normalized_url": "",
            "reachable": False,
            "content_type": "",
            "size_bytes": 0,
            "load_time": 0,
            "errors": [],
        }
        
        try:
            # Normalize URL
            normalized = self._normalize_url(url)
            validation["normalized_url"] = normalized
            
            # Quick HEAD request to check
            start_time = time.time()
            response = self.session.head(
                normalized,
                timeout=10,
                allow_redirects=True
            )
            load_time = time.time() - start_time
            
            validation.update({
                "is_valid": True,
                "reachable": True,
                "status_code": response.status_code,
                "content_type": response.headers.get('content-type', ''),
                "load_time": load_time,
            })
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                validation["errors"].append(f"Not HTML content: {content_type}")
            
        except requests.exceptions.RequestException as e:
            validation["errors"].append(f"Request failed: {e}")
        except Exception as e:
            validation["errors"].append(f"Validation error: {e}")
        
        return validation