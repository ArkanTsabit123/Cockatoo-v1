# src/ai_engine/prompt_templates.py

"""Prompt templates for various AI tasks.

Provides template management and rendering for prompts.
"""

import json
import re
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Prompt template with variables."""
    name: str
    template: str
    description: str = ""
    category: str = "general"
    variables: Set[str] = None
    
    def __post_init__(self):
        """Extract variables from template."""
        if self.variables is None:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> Set[str]:
        """Extract variable names from template.
        
        Variables are in format {{variable_name}}
        
        Returns:
            Set of variable names
        """
        pattern = r'\{\{([^}]+)\}\}'
        matches = re.findall(pattern, self.template)
        return set(match.strip() for match in matches)
    
    def format(self, **kwargs) -> str:
        """Format template with variables.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted prompt
            
        Raises:
            ValueError: If missing required variables
        """
        missing = self.variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        
        return result
    
    def validate(self) -> bool:
        """Validate template.
        
        Returns:
            True if valid
        """
        # Check for unclosed braces
        if self.template.count("{{") != self.template.count("}}"):
            logger.error(f"Template {self.name} has mismatched braces")
            return False
        
        # Check for invalid variable names
        for var in self.variables:
            if not var.isidentifier():
                logger.error(f"Template {self.name} has invalid variable name: {var}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["variables"] = list(self.variables)
        return data


class PromptManager:
    """Manager for prompt templates."""
    
    # Predefined templates
    DEFAULT_TEMPLATES = {
        "qa": PromptTemplate(
            name="qa",
            template="""Context: {{context}}

Question: {{question}}

Answer the question based on the context provided. If the answer cannot be found in the context, say "I cannot find this information in the provided context."

Answer:""",
            description="Question answering template",
            category="rag"
        ),
        
        "summarize": PromptTemplate(
            name="summarize",
            template="""Please summarize the following text in a {{style}} manner:

{{document}}

Summary:""",
            description="Document summarization template",
            category="summarization"
        ),
        
        "tagging": PromptTemplate(
            name="tagging",
            template="""Generate relevant tags for the following text:

{{document}}

Existing tags (if any): {{existing_tags}}

Generate 3-5 relevant tags that capture the main topics and themes. Return only the tags as a comma-separated list.

Tags:""",
            description="Auto-tagging template",
            category="tagging"
        ),
        
        "conversation": PromptTemplate(
            name="conversation",
            template="""{{history}}

Human: {{query}}

Assistant:""",
            description="Conversation template",
            category="conversation"
        ),
        
        "extract_keywords": PromptTemplate(
            name="extract_keywords",
            template="""Extract the most important keywords from the following text:

{{document}}

Return exactly {{num_keywords}} keywords as a comma-separated list.

Keywords:""",
            description="Keyword extraction template",
            category="extraction"
        ),
        
        "rewrite_query": PromptTemplate(
            name="rewrite_query",
            template="""Given the conversation history and the latest query, rewrite the query to be standalone and self-contained.

History: {{history}}
Latest query: {{query}}

Rewritten query:""",
            description="Query rewriting template",
            category="rag"
        ),
        
        "classify": PromptTemplate(
            name="classify",
            template="""Classify the following text into one of these categories: {{categories}}

Text: {{text}}

Category:""",
            description="Text classification template",
            category="classification"
        ),
        
        "translate": PromptTemplate(
            name="translate",
            template="""Translate the following text from {{source_language}} to {{target_language}}:

{{text}}

Translation:""",
            description="Translation template",
            category="translation"
        )
    }
    
    def __init__(self, templates_path: Optional[Union[str, Path]] = None):
        """Initialize prompt manager.
        
        Args:
            templates_path: Path to templates file
        """
        self.templates_path = Path(templates_path) if templates_path else None
        self.templates = {}
        self._lock = threading.RLock()
        
        # Load default templates
        self._load_defaults()
        
        # Load custom templates if exists
        if self.templates_path and self.templates_path.exists():
            self.load_templates()
    
    def _load_defaults(self) -> None:
        """Load default templates."""
        for name, template in self.DEFAULT_TEMPLATES.items():
            self.templates[name] = template
    
    def load_templates(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Load templates from file.
        
        Args:
            path: Path to templates file
            
        Returns:
            True if successful
        """
        load_path = Path(path) if path else self.templates_path
        if not load_path or not load_path.exists():
            return False
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                for name, template_data in data.items():
                    template = PromptTemplate(
                        name=name,
                        template=template_data["template"],
                        description=template_data.get("description", ""),
                        category=template_data.get("category", "general")
                    )
                    if template.validate():
                        self.templates[name] = template
            
            logger.info(f"Loaded {len(data)} templates from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            return False
    
    def save_templates(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Save templates to file.
        
        Args:
            path: Path to save to
            
        Returns:
            True if successful
        """
        save_path = Path(path) if path else self.templates_path
        if not save_path:
            return False
        
        try:
            data = {}
            for name, template in self.templates.items():
                data[name] = template.to_dict()
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(data)} templates to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
            return False
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate or None
        """
        return self.templates.get(name)
    
    def add_template(self, template: PromptTemplate, overwrite: bool = False) -> bool:
        """Add or update template.
        
        Args:
            template: PromptTemplate instance
            overwrite: Overwrite existing
            
        Returns:
            True if added
        """
        with self._lock:
            if template.name in self.templates and not overwrite:
                logger.warning(f"Template {template.name} already exists")
                return False
            
            if not template.validate():
                return False
            
            self.templates[template.name] = template
            logger.info(f"Added template: {template.name}")
            return True
    
    def remove_template(self, name: str) -> bool:
        """Remove template.
        
        Args:
            name: Template name
            
        Returns:
            True if removed
        """
        with self._lock:
            if name in self.templates:
                del self.templates[name]
                logger.info(f"Removed template: {name}")
                return True
            return False
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """List template names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of template names
        """
        if category:
            return [name for name, t in self.templates.items() if t.category == category]
        return list(self.templates.keys())
    
    def render(self, name: str, **kwargs) -> str:
        """Render template with variables.
        
        Args:
            name: Template name
            **kwargs: Variable values
            
        Returns:
            Rendered prompt
            
        Raises:
            ValueError: If template not found or missing variables
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template not found: {name}")
        
        return template.format(**kwargs)
    
    def get_categories(self) -> List[str]:
        """Get all categories.
        
        Returns:
            List of categories
        """
        categories = set(t.category for t in self.templates.values())
        return sorted(categories)


# Global instance getter
def get_prompt_manager(templates_path: Optional[Union[str, Path]] = None) -> PromptManager:
    """Get prompt manager instance.
    
    Args:
        templates_path: Path to templates file
        
    Returns:
        PromptManager instance
    """
    return PromptManager(templates_path)