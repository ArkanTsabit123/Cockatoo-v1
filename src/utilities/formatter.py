# src/utilities/formatter.py

"""Comprehensive formatting utilities for text, documents, chat, and exports."""

import os
import re
import json
import textwrap
import html
import unicodedata
from datetime import datetime, date, time
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field


class FormatType(Enum):
    """Supported output formats."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    RICH_TEXT = "rich_text"
    TERMINAL = "terminal"


class LanguageCode(Enum):
    """Supported language codes."""
    ENGLISH = "en"
    INDONESIAN = "id"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"


@dataclass
class FormattingOptions:
    """Configuration options for formatters."""
    max_line_length: int = 80
    indent_spaces: int = 4
    tab_width: int = 4
    preserve_line_breaks: bool = True
    collapse_multiple_spaces: bool = True
    trim_trailing_whitespace: bool = True
    normalize_quotes: bool = True
    convert_dashes: bool = True
    
    default_format: FormatType = FormatType.PLAIN_TEXT
    encoding: str = "utf-8"
    line_ending: str = "\n"
    ensure_ascii: bool = False
    
    language: LanguageCode = LanguageCode.ENGLISH
    locale: str = "en_US"
    timezone: str = "UTC"
    
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    
    decimal_separator: str = "."
    thousands_separator: str = ","
    decimal_places: int = 2
    
    truncate_length: int = 200
    ellipsis_text: str = "..."
    show_line_numbers: bool = False
    syntax_highlighting: bool = True
    
    table_border: bool = True
    table_header: bool = True
    table_alignment: str = "left"
    
    include_metadata: bool = True
    include_timestamps: bool = True
    include_source_info: bool = True


class BaseFormatter:
    """Base class for all formatters with common formatting methods."""
    
    def __init__(self, options: Optional[FormattingOptions] = None):
        self.options = options or FormattingOptions()
    
    def clean_text(self, text: str, preserve_formatting: bool = False) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        if text.startswith('\ufeff'):
            text = text[1:]
        
        lines = text.split('\n')
        
        if preserve_formatting:
            cleaned_lines = [line.strip() for line in lines]
            return '\n'.join(cleaned_lines)
        
        cleaned_lines = []
        for line in lines:
            cleaned_line = ' '.join(line.split())
            cleaned_lines.append(cleaned_line)
        
        cleaned = '\n'.join(cleaned_lines)
        
        if self.options.normalize_quotes:
            quote_map = {
                '“': '"', '”': '"',
                '‘': "'", '’': "'",
                '«': '"', '»': '"',
                '‹': "'", '›': "'",
            }
            for smart, straight in quote_map.items():
                cleaned = cleaned.replace(smart, straight)
        
        if self.options.convert_dashes:
            dash_types = ['–', '—', '―', '‒']
            for dash in dash_types:
                cleaned = cleaned.replace(dash, '-')
        
        return cleaned
    
    def format_paragraph(self, text: str, max_length: Optional[int] = None,
                        justify: bool = False) -> str:
        """Format text as a paragraph with line wrapping."""
        if max_length is None:
            max_length = self.options.max_line_length
        
        cleaned = self.clean_text(text)
        
        return textwrap.fill(
            cleaned,
            width=max_length,
            expand_tabs=False,
            replace_whitespace=True,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=True,
            subsequent_indent='' if justify else ''
        )
    
    def truncate_with_ellipsis(self, text: str, max_chars: Optional[int] = None,
                              position: str = "end") -> str:
        """Truncate text and add ellipsis at specified position."""
        if max_chars is None:
            max_chars = self.options.truncate_length
        
        if not text or len(text) <= max_chars:
            return text or ""
        
        ellipsis = self.options.ellipsis_text
        
        if max_chars <= len(ellipsis):
            return ellipsis[:max_chars]
        
        if position == "start":
            return ellipsis + text[-(max_chars - len(ellipsis)):]
        elif position == "middle":
            half = (max_chars - len(ellipsis)) // 2
            return text[:half] + ellipsis + text[-(max_chars - len(ellipsis) - half):]
        else:
            return text[:max_chars - len(ellipsis)] + ellipsis
    
    def format_number(self, number: Union[int, float], 
                     as_integer: bool = False) -> str:
        """Format number with thousands separator and decimal places."""
        if as_integer:
            num_str = f"{int(number):,}"
        else:
            format_str = f"{{:,.{self.options.decimal_places}f}}"
            num_str = format_str.format(number)
        
        if self.options.thousands_separator != "," or self.options.decimal_separator != ".":
            if '.' in num_str:
                int_part, dec_part = num_str.split('.')
                int_part = int_part.replace(',', self.options.thousands_separator)
                num_str = f"{int_part}{self.options.decimal_separator}{dec_part}"
            else:
                num_str = num_str.replace(',', self.options.thousands_separator)
        
        return num_str
    
    def format_datetime(self, dt: Union[str, datetime, date, time, float, int],
                       format_str: Optional[str] = None) -> str:
        """Format datetime objects to string."""
        if isinstance(dt, datetime):
            dt_obj = dt
        elif isinstance(dt, date):
            dt_obj = datetime.combine(dt, datetime.min.time())
        elif isinstance(dt, time):
            dt_obj = datetime.combine(date.today(), dt)
        elif isinstance(dt, (int, float)):
            dt_obj = datetime.fromtimestamp(dt)
        elif isinstance(dt, str):
            try:
                dt_obj = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except ValueError:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                           "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
                    try:
                        dt_obj = datetime.strptime(dt, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return dt
        else:
            return str(dt)
        
        if format_str is None:
            if isinstance(dt, time):
                format_str = self.options.time_format
            elif isinstance(dt, date) and not isinstance(dt, datetime):
                format_str = self.options.date_format
            else:
                format_str = self.options.datetime_format
        
        return dt_obj.strftime(format_str)


class TextFormatter(BaseFormatter):
    """Formatter for plain text with lists, tables, and code blocks."""
    
    def format_list(self, items: List[Any], 
                   bullet_type: str = "bullet",
                   start_index: int = 1,
                   indent_level: int = 0) -> str:
        """Format a list with specified bullet type."""
        if not items:
            return ""
        
        indent = " " * (indent_level * self.options.indent_spaces)
        formatted_items = []
        
        for i, item in enumerate(items, start=start_index):
            item_text = str(item).strip()
            if not item_text:
                continue
            
            if bullet_type == "number":
                prefix = f"{i}."
            elif bullet_type == "letter":
                prefix = f"{chr(96 + i)}." if i <= 26 else f"{i}."
            elif bullet_type == "dash":
                prefix = "-"
            else:
                prefix = "•"
            
            formatted_items.append(f"{indent}{prefix} {item_text}")
        
        return '\n'.join(formatted_items)
    
    def format_table(self, data: List[List[Any]], 
                    headers: Optional[List[str]] = None,
                    alignments: Optional[List[str]] = None) -> str:
        """Format data as a table."""
        if not data:
            return ""
        
        if alignments is None:
            alignments = ['left'] * len(data[0])
        
        if headers:
            table_data = [headers] + data
        else:
            table_data = data
        
        column_widths = []
        for col_idx in range(len(table_data[0])):
            max_width = 0
            for row in table_data:
                cell_text = str(row[col_idx]) if col_idx < len(row) else ""
                max_width = max(max_width, len(cell_text))
            column_widths.append(max_width + 2)
        
        lines = []
        
        if headers and self.options.table_border:
            top_border = "┌" + "┬".join("─" * w for w in column_widths) + "┐"
            lines.append(top_border)
        
        if headers:
            header_cells = []
            for i, header in enumerate(headers):
                width = column_widths[i]
                alignment = alignments[i] if i < len(alignments) else 'left'
                
                if alignment == 'center':
                    cell = header.center(width - 2)
                elif alignment == 'right':
                    cell = header.rjust(width - 2)
                else:
                    cell = header.ljust(width - 2)
                
                header_cells.append(f" {cell} ")
            
            line = "│" + "│".join(header_cells) + "│"
            lines.append(line)
            
            if self.options.table_border:
                separator = "├" + "┼".join("─" * w for w in column_widths) + "┤"
                lines.append(separator)
        
        for row in data:
            row_cells = []
            for i, cell in enumerate(row):
                width = column_widths[i]
                cell_text = str(cell)
                alignment = alignments[i] if i < len(alignments) else 'left'
                
                if alignment == 'center':
                    cell_formatted = cell_text.center(width - 2)
                elif alignment == 'right':
                    cell_formatted = cell_text.rjust(width - 2)
                else:
                    cell_formatted = cell_text.ljust(width - 2)
                
                row_cells.append(f" {cell_formatted} ")
            
            line = "│" + "│".join(row_cells) + "│"
            lines.append(line)
        
        if self.options.table_border:
            bottom_border = "└" + "┴".join("─" * w for w in column_widths) + "┘"
            lines.append(bottom_border)
        
        return '\n'.join(lines)
    
    def format_code_block(self, code: str, language: str = "",
                         line_numbers: Optional[bool] = None) -> str:
        """Format code as a code block."""
        if line_numbers is None:
            line_numbers = self.options.show_line_numbers
        
        lines = code.rstrip().split('\n')
        
        if line_numbers:
            max_line_num = len(lines)
            line_num_width = len(str(max_line_num))
            formatted_lines = []
            
            for i, line in enumerate(lines, 1):
                line_num = f"{i:{line_num_width}}"
                formatted_lines.append(f"{line_num} │ {line}")
            
            formatted_code = '\n'.join(formatted_lines)
        else:
            formatted_code = '\n'.join(lines)
        
        if language:
            return f"```{language}\n{formatted_code}\n```"
        else:
            return f"```\n{formatted_code}\n```"
    
    def generate_table_of_contents(self, headings: List[Dict[str, Any]],
                                  max_depth: int = 3) -> str:
        """Generate a table of contents from headings."""
        if not headings:
            return ""
        
        toc_lines = ["## Table of Contents", ""]
        
        for heading in headings:
            level = heading.get('level', 1)
            text = heading.get('text', '')
            anchor = heading.get('anchor', '')
            
            if level > max_depth:
                continue
            
            indent = "  " * (level - 1)
            if anchor:
                toc_lines.append(f"{indent}- [{text}](#{anchor})")
            else:
                anchor_text = re.sub(r'[^\w\s-]', '', text.lower())
                anchor_text = re.sub(r'[-\s]+', '-', anchor_text).strip('-')
                toc_lines.append(f"{indent}- [{text}](#{anchor_text})")
        
        return '\n'.join(toc_lines)


class DocumentFormatter(BaseFormatter):
    """Formatter for document metadata and chunks."""
    
    def format_document_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format document metadata as markdown."""
        lines = ["## Document Information", ""]
        
        if 'title' in metadata:
            lines.append(f"**Title:** {metadata['title']}")
        if 'author' in metadata:
            lines.append(f"**Author:** {metadata['author']}")
        if 'date' in metadata:
            lines.append(f"**Date:** {self.format_datetime(metadata['date'])}")
        if 'file_name' in metadata:
            lines.append(f"**File:** {metadata['file_name']}")
        if 'file_size' in metadata:
            lines.append(f"**Size:** {self.format_file_size(metadata['file_size'])}")
        if 'file_type' in metadata:
            lines.append(f"**Type:** {metadata['file_type'].upper()}")
        if 'chunk_count' in metadata:
            lines.append(f"**Chunks:** {metadata['chunk_count']}")
        if 'word_count' in metadata:
            lines.append(f"**Words:** {self.format_number(metadata['word_count'])}")
        if 'language' in metadata:
            lines.append(f"**Language:** {metadata['language']}")
        if 'upload_date' in metadata:
            lines.append(f"**Uploaded:** {self.format_datetime(metadata['upload_date'])}")
        if 'processed_at' in metadata:
            lines.append(f"**Processed:** {self.format_datetime(metadata['processed_at'])}")
        if 'tags' in metadata and isinstance(metadata['tags'], list):
            lines.append(f"**Tags:** {', '.join(metadata['tags'])}")
        if 'summary' in metadata and metadata['summary']:
            lines.extend(["", "### Summary", metadata['summary']])
        
        return '\n'.join(lines)
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        elif size < 10:
            return f"{size:.2f} {units[unit_index]}"
        elif size < 100:
            return f"{size:.1f} {units[unit_index]}"
        else:
            return f"{int(size)} {units[unit_index]}"
    
    def format_chunk_preview(self, chunk: Dict[str, Any], 
                            max_lines: int = 5,
                            show_metadata: bool = True) -> str:
        """Format a document chunk preview."""
        lines = []
        
        if show_metadata:
            metadata = chunk.get('metadata', {})
            if metadata:
                source = metadata.get('source_file', 'Unknown')
                chunk_idx = metadata.get('chunk_index', 0)
                total = metadata.get('total_chunks', 1)
                lines.append(f"**Chunk {chunk_idx + 1}/{total} from {source}**")
                lines.append("")
        
        text = chunk.get('text', '')
        if text:
            text_lines = text.split('\n')
            content_lines = []
            line_count = 0
            for line in text_lines:
                if line_count >= max_lines:
                    break
                content_lines.append(line)
                if line.strip():
                    line_count += 1
            
            lines.extend(content_lines)
            
            remaining_lines = text_lines[len(content_lines):]
            if any(l.strip() for l in remaining_lines):
                lines.append(self.options.ellipsis_text)
        
        score = chunk.get('score')
        if score is not None:
            lines.append("")
            lines.append(f"*Relevance: {score:.2%}*")
        
        return '\n'.join(lines)
    
    def format_citation(self, source: Dict[str, Any], 
                       format_type: str = "inline") -> str:
        """Format a source citation."""
        if format_type == "inline":
            author = source.get('author', 'Unknown')
            year = source.get('year', '')
            return f"[{author}, {year}]" if year else f"[{author}]"
        
        elif format_type == "footnote":
            author = source.get('author', 'Unknown')
            title = source.get('title', 'Unknown Title')
            year = source.get('year', '')
            pages = source.get('pages', '')
            
            parts = [author]
            if year:
                parts.append(f"({year})")
            parts.append(f"\"{title}\"")
            if pages:
                parts.append(f"pp. {pages}")
            
            return ' '.join(parts)
        
        else:
            lines = []
            if 'author' in source:
                lines.append(f"**Author:** {source['author']}")
            if 'title' in source:
                lines.append(f"**Title:** {source['title']}")
            if 'year' in source:
                lines.append(f"**Year:** {source['year']}")
            if 'publisher' in source:
                lines.append(f"**Publisher:** {source['publisher']}")
            if 'url' in source:
                lines.append(f"**URL:** {source['url']}")
            if 'pages' in source:
                lines.append(f"**Pages:** {source['pages']}")
            
            return '\n'.join(lines)


class ChatMessageFormatter(BaseFormatter):
    """Formatter for chat messages and conversations."""
    
    def __init__(self, options: Optional[FormattingOptions] = None):
        super().__init__(options)
        self.role_prefixes = {
            'user': 'You: ',
            'assistant': 'cockatoo_v1: ', 
            'system': 'System: ',
            'error': 'Error: '
        }
    
    def format_message(self, message: Dict[str, Any], 
                      include_timestamp: Optional[bool] = None) -> str:
        """Format a single chat message."""
        if include_timestamp is None:
            include_timestamp = self.options.include_timestamps
        
        role = message.get('role', 'user')
        content = message.get('content', '')
        timestamp = message.get('timestamp')
        
        prefix = self.role_prefixes.get(role, '')
        
        if include_timestamp and timestamp and isinstance(timestamp, datetime):
            time_str = timestamp.strftime("%H:%M")
            prefix = f"[{time_str}] {prefix}"
        
        if isinstance(content, dict):
            if 'text' in content:
                formatted_content = content['text']
                if 'sources' in content and content['sources']:
                    sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in content['sources'])
                    formatted_content += sources_text
            elif 'answer' in content:
                formatted_content = content['answer']
                if 'sources' in content and content['sources']:
                    sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in content['sources'])
                    formatted_content += sources_text
            else:
                formatted_content = str(content)
        else:
            formatted_content = str(content)
        
        formatted_content = self.clean_text(formatted_content, preserve_formatting=True)
        
        lines = formatted_content.split('\n')
        if len(lines) == 1:
            return f"{prefix}{formatted_content}"
        
        result = [f"{prefix}{lines[0]}"]
        indent = " " * len(prefix)
        
        for line in lines[1:]:
            if line.strip():
                result.append(f"{indent}{line}")
            else:
                result.append("")
        
        return '\n'.join(result)
    
    def format_conversation(self, messages: List[Dict[str, Any]],
                           include_metadata: Optional[bool] = None) -> str:
        """Format a conversation of multiple messages."""
        if include_metadata is None:
            include_metadata = self.options.include_metadata
        
        formatted_messages = []
        
        if include_metadata and messages:
            first_msg_time = messages[0].get('timestamp')
            last_msg_time = messages[-1].get('timestamp')
            
            if first_msg_time and last_msg_time:
                header = f"## Conversation ({len(messages)} messages)"
                formatted_messages.append(header)
                formatted_messages.append("")
        
        for msg in messages:
            formatted_messages.append(self.format_message(msg))
            formatted_messages.append("")
        
        return '\n'.join(formatted_messages).strip()
    
    def format_source_reference(self, source: Dict[str, Any], 
                               format_type: str = "compact") -> str:
        """Format a source reference."""
        if format_type == "compact":
            doc_name = source.get('document_name', 'Unknown Document')
            page = source.get('page')
            confidence = source.get('confidence')
            
            ref = f"[{doc_name}" + (f", Page {page}]" if page else "]")
            if confidence:
                ref += f" ({confidence:.0%})"
            
            return ref
        
        elif format_type == "detailed":
            doc_name = source.get('document_name', 'Unknown Document')
            page = source.get('page')
            text = source.get('text', '')
            confidence = source.get('confidence')
            
            lines = [f"**{doc_name}**"]
            if page:
                lines.append(f"*Page {page}*")
            if confidence:
                lines.append(f"*Relevance: {confidence:.0%}*")
            if text:
                preview = self.truncate_with_ellipsis(text, 150)
                lines.append(f"> {preview}")
            
            return '\n'.join(lines)
        
        else:
            return source.get('document_name', 'Source')


class ExportFormatter(BaseFormatter):
    """Formatter for exporting content to various formats."""
    
    def to_markdown(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """Export content to markdown format."""
        lines = []
        
        if metadata:
            lines.append("---")
            for key, value in metadata.items():
                if value is not None:
                    lines.append(f"{key}: {value}")
            lines.append("---")
            lines.append("")
        
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'role' in item:
                    role = item['role']
                    msg_content = item.get('content', '')
                    lines.append(f"### {role.capitalize()}")
                    lines.append("")
                    if isinstance(msg_content, dict):
                        if 'text' in msg_content:
                            text_content = msg_content['text']
                            lines.append(text_content)
                            if 'sources' in msg_content and msg_content['sources']:
                                lines.append("")
                                lines.append("**Sources:**")
                                for source in msg_content['sources']:
                                    lines.append(f"- {source}")
                        elif 'answer' in msg_content:
                            lines.append(msg_content['answer'])
                            if 'sources' in msg_content and msg_content['sources']:
                                lines.append("")
                                lines.append("**Sources:**")
                                for source in msg_content['sources']:
                                    lines.append(f"- {source}")
                        else:
                            lines.append(str(msg_content))
                    else:
                        lines.append(str(msg_content))
                    lines.append("")
                else:
                    lines.append(f"- {item}")
        elif isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"## {key}")
                lines.append("")
                if isinstance(value, (list, dict)):
                    lines.append(self.to_markdown(value))
                else:
                    lines.append(str(value))
                lines.append("")
        else:
            lines.append(str(content))
        
        return '\n'.join(lines).strip()
    
    def to_html(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """Export content to HTML format."""
        lines = ['<!DOCTYPE html>', '<html>', '<head>']
        
        if metadata:
            lines.append('<meta charset="UTF-8">')
            if 'title' in metadata:
                lines.append(f'<title>{html.escape(metadata["title"])}</title>')
            if 'author' in metadata:
                lines.append(f'<meta name="author" content="{html.escape(metadata["author"])}">')
            if 'date' in metadata:
                lines.append(f'<meta name="date" content="{html.escape(str(metadata["date"]))}">')
        
        lines.append('<style>')
        lines.append('body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }')
        lines.append('h1, h2, h3 { color: #333; }')
        lines.append('.message { margin-bottom: 20px; padding: 10px; border-left: 3px solid #007bff; }')
        lines.append('.user { background-color: #f0f8ff; }')
        lines.append('.assistant { background-color: #f9f9f9; }')
        lines.append('.source { font-size: 0.9em; color: #666; margin-top: 5px; }')
        lines.append('</style>')
        lines.append('</head>')
        lines.append('<body>')
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'role' in item:
                    role = item['role']
                    msg_content = item.get('content', '')
                    lines.append(f'<div class="message {role}">')
                    lines.append(f'<strong>{role.capitalize()}:</strong>')
                    
                    if isinstance(msg_content, dict):
                        content_text = msg_content.get('text', str(msg_content))
                    else:
                        content_text = str(msg_content)
                    
                    lines.append(f'<div>{html.escape(content_text)}</div>')
                    
                    if 'sources' in item and item['sources']:
                        lines.append('<div class="source">')
                        lines.append('<strong>Sources:</strong>')
                        for source in item['sources']:
                            lines.append(f'<div>{html.escape(str(source))}</div>')
                        lines.append('</div>')
                    
                    lines.append('</div>')
                else:
                    lines.append(f'<p>{html.escape(str(item))}</p>')
        elif isinstance(content, str):
            html_content = html.escape(content)
            html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
            html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)
            html_content = re.sub(r'`(.+?)`', r'<code>\1</code>', html_content)
            html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
            lines.append(html_content)
        else:
            lines.append(f'<p>{html.escape(str(content))}</p>')
        
        lines.append('</body>')
        lines.append('</html>')
        
        return '\n'.join(lines)
    
    def to_json(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """Export content to JSON format."""
        export_data = {}
        
        if metadata:
            export_data['metadata'] = metadata
        
        if isinstance(content, dict) and len(content) == 1 and 'content' not in content:
            export_data['content'] = content
        else:
            export_data['content'] = content
        
        def json_serializer(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, time):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        def make_serializable(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (datetime, date, time)):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            if isinstance(obj, (set, frozenset)):
                return [make_serializable(item) for item in obj]
            return str(obj)
        
        export_data = make_serializable(export_data)
        
        return json.dumps(export_data, indent=2, 
                        ensure_ascii=self.options.ensure_ascii,
                        default=json_serializer)
    
    def to_plain_text(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """Export content to plain text format."""
        lines = []
        
        if metadata:
            lines.append("=" * 60)
            for key, value in metadata.items():
                if value is not None:
                    lines.append(f"{key}: {value}")
            lines.append("=" * 60)
            lines.append("")
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'role' in item and 'content' in item:
                    role = item['role'].upper()
                    msg_content = item['content']
                    if isinstance(msg_content, dict):
                        msg_content = msg_content.get('text', str(msg_content))
                    lines.append(f"[{role}]")
                    lines.append(str(msg_content))
                    lines.append("")
                else:
                    lines.append(f"- {item}")
        elif isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"{key.upper()}:")
                lines.append(str(value))
                lines.append("")
        else:
            lines.append(str(content))
        
        return '\n'.join(lines)


class FormatterFactory:
    """Factory for creating and caching formatter instances."""
    
    _formatters = {}
    
    @classmethod
    def get_formatter(cls, formatter_type: str = "text", 
                     options: Optional[FormattingOptions] = None) -> BaseFormatter:
        """Get a formatter instance of the specified type."""
        if formatter_type not in cls._formatters:
            cls._create_formatter(formatter_type, options)
        return cls._formatters[formatter_type]
    
    @classmethod
    def _create_formatter(cls, formatter_type: str, 
                         options: Optional[FormattingOptions] = None):
        formatter_classes = {
            'text': TextFormatter,
            'document': DocumentFormatter,
            'chat': ChatMessageFormatter,
            'export': ExportFormatter,
            'markdown': TextFormatter,
        }
        
        formatter_class = formatter_classes.get(formatter_type, TextFormatter)
        cls._formatters[formatter_type] = formatter_class(options)
    
    @classmethod
    def format_content(cls, content: Any, format_type: str = "text",
                      formatter_type: str = "text", **kwargs) -> str:
        """Format content using the specified formatter and output format."""
        formatter = cls.get_formatter(formatter_type)
        
        if format_type == "markdown":
            if formatter_type == "chat":
                export_formatter = cls.get_formatter('export')
                return export_formatter.to_markdown(content, **kwargs)
            elif hasattr(formatter, 'to_markdown'):
                return formatter.to_markdown(content, **kwargs)
            return str(content)
        elif format_type == "html" and hasattr(formatter, 'to_html'):
            return formatter.to_html(content, **kwargs)
        elif format_type == "json":
            export_formatter = cls.get_formatter('export')
            
            if isinstance(content, dict):
                return export_formatter.to_json(content, **kwargs)
            else:
                return export_formatter.to_json({"content": content}, **kwargs)
        elif format_type == "plain_text" and hasattr(formatter, 'to_plain_text'):
            return formatter.to_plain_text(content, **kwargs)
        
        return str(content)


def format_text(text: str, max_length: Optional[int] = None, **kwargs) -> str:
    """Convenience function to format text as a paragraph."""
    formatter = FormatterFactory.get_formatter('text')
    return formatter.format_paragraph(text, max_length, **kwargs)


def format_document_metadata(metadata: Dict[str, Any]) -> str:
    """Convenience function to format document metadata."""
    formatter = FormatterFactory.get_formatter('document')
    return formatter.format_document_metadata(metadata)


def format_chat_message(message: Dict[str, Any], **kwargs) -> str:
    """Convenience function to format a chat message."""
    formatter = FormatterFactory.get_formatter('chat')
    return formatter.format_message(message, **kwargs)


def export_to_markdown(content: Any, metadata: Optional[Dict] = None) -> str:
    """Convenience function to export content to markdown."""
    formatter = FormatterFactory.get_formatter('export')
    return formatter.to_markdown(content, metadata)


default_text_formatter = TextFormatter()
default_document_formatter = DocumentFormatter()
default_chat_formatter = ChatMessageFormatter()
default_export_formatter = ExportFormatter()