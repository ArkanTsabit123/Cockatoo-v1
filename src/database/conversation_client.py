# cockatoo_v1/src/database/conversation_client.py
"""
Conversation database operations module.

Handles conversation and message CRUD operations with SQLite database.
Provides methods for creating, retrieving, updating, and deleting conversations
with proper JSON serialization and parameterized queries for security.
"""
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


class ConversationClient:
    """Conversation-specific database operations."""
    
    def __init__(self, sqlite_client: SQLiteClient) -> None:
        self.client = sqlite_client
        logger.info("Conversation client initialized")
    
    def create_conversation(self, title: Optional[str] = None) -> Optional[str]:
        """Create a new conversation."""
        try:
            conversation_id = f"conv_{uuid.uuid4().hex[:16]}"
            
            # Parameterized query
            query = """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """
            
            now = datetime.now()
            params = (conversation_id, title, now, now)
            
            self.client.execute_query(query, params)
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            return None
    
    def add_message(self, conversation_id: str, message_data: Dict[str, Any]) -> Optional[str]:
        """Add message to conversation with parameterized query."""
        try:
            message_id = message_data.get('id') or f"msg_{uuid.uuid4().hex[:16]}"
            
            # Parameterized query
            query = """
            INSERT INTO messages (
                id, conversation_id, role, content, tokens,
                model_used, sources_json, processing_time_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                message_id,
                conversation_id,
                message_data.get('role', 'user'),
                message_data.get('content', ''),
                message_data.get('tokens', 0),
                message_data.get('model_used', ''),
                json.dumps(message_data.get('sources', [])),
                message_data.get('processing_time_ms', 0),
                message_data.get('created_at', datetime.now())
            )
            
            self.client.execute_query(query, params)
            
            # Update conversation message count
            self._update_conversation_stats(conversation_id)
            
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return None
    
    def get_conversation(self, conversation_id: str, include_messages: bool = True) -> Optional[Dict[str, Any]]:
        """Get conversation by ID with parameterized query."""
        try:
            # Parameterized query
            query = "SELECT * FROM conversations WHERE id = ?"
            results = self.client.execute_query(query, (conversation_id,))
            
            if not results:
                return None
            
            conversation = dict(results[0])
            
            # Get messages if requested
            if include_messages:
                messages_query = """
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY created_at ASC
                """
                messages_results = self.client.execute_query(messages_query, (conversation_id,))
                
                conversation['messages'] = [
                    self._process_message_row(row) for row in messages_results
                ]
            
            # Parse JSON fields
            if conversation.get('tags_json'):
                try:
                    conversation['tags'] = json.loads(conversation['tags_json'])
                except json.JSONDecodeError:
                    conversation['tags'] = []
            else:
                conversation['tags'] = []
            
            # Remove JSON field
            conversation.pop('tags_json', None)
            
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None
    
    def list_conversations(self, page: int = 1, page_size: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        """List conversations with pagination using parameterized queries."""
        try:
            # Count total with parameterized query
            count_query = "SELECT COUNT(*) as total FROM conversations"
            count_result = self.client.execute_query(count_query, ())
            total = count_result[0]['total'] if count_result else 0
            
            # Get paginated results with parameterized query
            offset = (page - 1) * page_size
            query = """
            SELECT * FROM conversations 
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """
            
            results = self.client.execute_query(query, (page_size, offset))
            
            conversations = []
            for row in results:
                conv = dict(row)
                
                # Parse JSON fields
                if conv.get('tags_json'):
                    try:
                        conv['tags'] = json.loads(conv['tags_json'])
                    except json.JSONDecodeError:
                        conv['tags'] = []
                else:
                    conv['tags'] = []
                
                # Remove JSON field
                conv.pop('tags_json', None)
                
                conversations.append(conv)
            
            return conversations, total
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return [], 0
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation with parameterized query."""
        try:
            query = "DELETE FROM conversations WHERE id = ?"
            self.client.execute_query(query, (conversation_id,))
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False
    
    def _update_conversation_stats(self, conversation_id: str) -> None:
        """Update conversation statistics with parameterized query."""
        try:
            # Count messages and sum tokens with parameterized query
            query = """
            SELECT 
                COUNT(*) as message_count,
                SUM(tokens) as total_tokens
            FROM messages 
            WHERE conversation_id = ?
            """
            
            result = self.client.execute_query(query, (conversation_id,))
            
            if result:
                message_count = result[0]['message_count'] or 0
                total_tokens = result[0]['total_tokens'] or 0
                
                # Update conversation with parameterized query
                update_query = """
                UPDATE conversations 
                SET message_count = ?, total_tokens = ?, updated_at = ?
                WHERE id = ?
                """
                
                self.client.execute_query(
                    update_query,
                    (message_count, total_tokens, datetime.now(), conversation_id)
                )
                
        except Exception as e:
            logger.error(f"Failed to update conversation stats: {e}")
    
    def _process_message_row(self, row: Dict) -> Dict[str, Any]:
        """Process database row into message dictionary."""
        try:
            message = dict(row)
            
            # Parse sources JSON
            if message.get('sources_json'):
                try:
                    message['sources'] = json.loads(message['sources_json'])
                except json.JSONDecodeError:
                    message['sources'] = []
            else:
                message['sources'] = []
            
            # Remove JSON field
            message.pop('sources_json', None)
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to process message row: {e}")
            return dict(row)
    
    def bulk_add_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> List[str]:
        """Bulk add messages with parameterized queries."""
        try:
            query = """
            INSERT INTO messages (
                id, conversation_id, role, content, tokens,
                model_used, sources_json, processing_time_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params_list = []
            message_ids = []
            
            for msg_data in messages:
                message_id = msg_data.get('id') or f"msg_{uuid.uuid4().hex[:16]}"
                message_ids.append(message_id)
                
                params = (
                    message_id,
                    conversation_id,
                    msg_data.get('role', 'user'),
                    msg_data.get('content', ''),
                    msg_data.get('tokens', 0),
                    msg_data.get('model_used', ''),
                    json.dumps(msg_data.get('sources', [])),
                    msg_data.get('processing_time_ms', 0),
                    msg_data.get('created_at', datetime.now())
                )
                params_list.append(params)
            
            success = self.client.execute_many(query, params_list)
            
            if success:
                self._update_conversation_stats(conversation_id)
                return message_ids
            else:
                return []
            
        except Exception as e:
            logger.error(f"Failed to bulk add messages: {e}")
            return []