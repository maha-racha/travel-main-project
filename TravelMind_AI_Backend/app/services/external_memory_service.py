import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import func
from app.core.config import Settings
from .database_manager import DatabaseManager
from app.models.models import ChatMemory, ConversationMessage, UserInfo
import json

from app.core.logger import get_logger

logger = get_logger(__name__)


class ExternalMemoryService:
    def __init__(self, settings: Settings, db_manager: DatabaseManager):
        self.settings = settings
        self.db = db_manager
        self.db = db_manager
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=0.3, # Keeping low temp for consistent summaries
        )
        self.message_threshold = 10

    def add_user_message(self, user_id: str, message: str, max_tokens: Optional[int] = None, trip_id: Optional[int] = None, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0):
        logger.info(f"User message: {message}")
        self._store_temp_message(user_id, "user", message, trip_id, input_tokens, output_tokens, total_tokens)
        self._check_and_update_memory(user_id, max_tokens)

    def add_assistant_message(self, user_id: str, message: str, max_tokens: Optional[int] = None, trip_id: Optional[int] = None, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0):
        logger.info(f"Assistant message: {message}")
        self._store_temp_message(user_id, "assistant", message, trip_id, input_tokens, output_tokens, total_tokens)
        self._check_and_update_memory(user_id, max_tokens)

    def get_memory_summary(self, user_id: str) -> Optional[str]:
        session = self.db.get_session()
        try:
            memory = session.query(ChatMemory).filter_by(
                user_id=user_id
            ).order_by(ChatMemory.updated_at.desc()).first()
            
            return memory.memory_summary if memory else None
        finally:
            session.close()

    def _store_temp_message(self, user_id: str, role: str, content: str, trip_id: Optional[int] = None, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0):
        session = self.db.get_session()
        try:
            message = ConversationMessage(
                user_id=user_id,
                trip_id=trip_id,
                role=role,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            )
            session.add(message)
            session.commit()
        finally:
            session.close()

    def get_recent_messages(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        session = self.db.get_session()
        try:
            messages_query = session.query(ConversationMessage).filter(
                ConversationMessage.user_id == user_id,
                ConversationMessage.trip_id.is_(None)
            ).order_by(ConversationMessage.created_at.desc()).limit(limit).all()
            
            messages = []
            for msg in reversed(messages_query):
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            return messages
        finally:
            session.close()

    def _count_unsummarized_messages(self, user_id: str) -> int:
        session = self.db.get_session()
        try:
            # Get the last summary time
            last_memory = session.query(func.max(ChatMemory.updated_at)).filter_by(
                user_id=user_id
            ).scalar()
            
            # Count unsummarized messages
            query = session.query(func.count(ConversationMessage.id)).filter(
                ConversationMessage.user_id == user_id,
                ConversationMessage.trip_id.is_(None)
            )
            
            if last_memory:
                query = query.filter(ConversationMessage.created_at > last_memory)
            
            return query.scalar() or 0
        finally:
            session.close()

    def _check_and_update_memory(self, user_id: str, max_tokens: Optional[int] = None):
        message_count = self._count_unsummarized_messages(user_id)
        
        if message_count >= self.message_threshold:
            self._generate_and_store_summary(user_id, max_tokens)

    def _generate_and_store_summary(self, user_id: str, max_tokens: Optional[int] = None):
        recent_messages = self.get_recent_messages(user_id, self.message_threshold)
        logger.info("Genearating and storing summary")
        if not recent_messages:
            return

        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in recent_messages
        ])

        existing_summary = self.get_memory_summary(user_id)
        logger.info(f'Existing summary {existing_summary}')
        if existing_summary:
            template = """You create concise memory summaries for travel conversations.

Current memory summary:
{existing_summary}

New conversation messages:
{conversation_text}

Update the memory summary by incorporating the new information. Focus on:
- User preferences and interests
- Travel plans and decisions
- Important context for future conversations
- Key facts mentioned

Keep the updated summary concise (150-250 words)."""
        else:
            template = """You create concise memory summaries for travel conversations.

Conversation:
{conversation_text}

Create a memory summary focusing on:
- User preferences and interests
- Travel plans and decisions
- Important context for future conversations
- Key facts mentioned

Keep the summary concise (150-250 words)."""

        try:
            effective_max_tokens = max_tokens or 350

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm

            response = chain.invoke({
                "existing_summary": existing_summary if existing_summary else "",
                "conversation_text": conversation_text
            })
            
            new_summary = response.content
            logger.info(f'New summmary generated: {new_summary}')
            session = self.db.get_session()
            try:
                if existing_summary:
                    # Update existing memory
                    memory = session.query(ChatMemory).filter_by(user_id=user_id).first()
                    if memory:
                        memory.memory_summary = new_summary
                        memory.updated_at = datetime.utcnow()
                else:
                    # Create new memory
                    memory = ChatMemory(
                        user_id=user_id,
                        memory_summary=new_summary
                    )
                    session.add(memory)
                
                session.commit()
            finally:
                session.close()

            self._cleanup_old_messages(user_id)

            logger.info(f"Updated memory summary for user {user_id}")

        except Exception as e:
            logger.error(f"Error generating memory summary: {str(e)}")

    # TODO: Why do we need to class old messages
    def _cleanup_old_messages(self, user_id: str):
        session = self.db.get_session()
        try:
            # Get the 5 most recent messages to keep
            recent_messages = session.query(ConversationMessage.id).filter(
                ConversationMessage.user_id == user_id,
                ConversationMessage.trip_id.is_(None)
            ).order_by(ConversationMessage.created_at.desc()).limit(5).all()
            
            recent_ids = [msg.id for msg in recent_messages]
            
            # Delete old messages
            if recent_ids:
                session.query(ConversationMessage).filter(
                    ConversationMessage.user_id == user_id,
                    ConversationMessage.trip_id.is_(None),
                    ~ConversationMessage.id.in_(recent_ids)
                ).delete(synchronize_session=False)
            
            session.commit()
        finally:
            session.close()

    # TODO: Why do we need clear_memory even that is not used here.
    def clear_memory(self, user_id: str):
        session = self.db.get_session()
        try:
            # Delete chat memory
            session.query(ChatMemory).filter_by(user_id=user_id).delete()
            
            # Delete conversation messages
            session.query(ConversationMessage).filter(
                ConversationMessage.user_id == user_id,
                ConversationMessage.trip_id.is_(None)
            ).delete()
            
            session.commit()
            logger.info(f"Cleared memory for user {user_id}")
        finally:
            session.close()

    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info:
                return {
                    "current_city": user_info.current_city,
                    "preferences": user_info.preferences,
                    "language": user_info.language,
                    "amenities": json.loads(user_info.amenities) if user_info.amenities else [],
                    "seen_items": json.loads(user_info.seen_items) if user_info.seen_items else [],
                    "budget": user_info.budget
                }
            return {"current_city": None, "preferences": None, "language": None, "amenities": [], "seen_items": [], "budget": None}
        finally:
            session.close()

    def update_user_city(self, user_id: str, city: str):
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info:
                user_info.current_city = city
                user_info.updated_at = datetime.utcnow()
            else:
                user_info = UserInfo(user_id=user_id, current_city=city)
                session.add(user_info)
            session.commit()
        finally:
            session.close()

    def update_user_preferences(self, user_id: str, preferences: str):
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info:
                # Append to existing preferences or overwrite? 
                # For now let's just append if it exists, or maybe the agent sends the full string?
                # Let's assume the agent might send a new fact.
                # Actually, simpler to just append with a newline if it exists.
                if user_info.preferences:
                    user_info.preferences = f"{user_info.preferences}\n{preferences}"
                else:
                    user_info.preferences = preferences
                user_info.updated_at = datetime.utcnow()
            else:
                user_info = UserInfo(user_id=user_id, preferences=preferences)
                session.add(user_info)
            session.commit()
        finally:
            session.close()

    def update_user_language(self, user_id: str, language: str):
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info:
                user_info.language = language
                user_info.updated_at = datetime.utcnow()
            else:
                user_info = UserInfo(user_id=user_id, language=language)
                session.add(user_info)
            session.commit()
        finally:
            session.close()

    def add_seen_items(self, user_id: str, item_ids: List[str]):
        if not item_ids:
            return
            
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if not user_info:
                user_info = UserInfo(user_id=user_id)
                session.add(user_info)
            
            # Load existing
            current_seen = []
            if user_info.seen_items:
                try:
                    current_seen = json.loads(user_info.seen_items)
                except:
                    current_seen = []
            
            # Append new uniqe items
            updated = False
            for item_id in item_ids:
                if item_id not in current_seen:
                    current_seen.append(item_id)
                    updated = True
            
            if updated:
                user_info.seen_items = json.dumps(current_seen)
                user_info.updated_at = datetime.utcnow()
                session.commit()
                logger.info(f"Added {len(item_ids)} items to seen_items for user {user_id}")
        except Exception as e:
            logger.error(f"Error adding seen items: {e}")
        finally:
            session.close()

    def get_seen_items(self, user_id: str) -> List[str]:
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info and user_info.seen_items:
                try:
                    return json.loads(user_info.seen_items)
                except:
                    return []
            return []
        finally:
            session.close()

    def update_user_amenities(self, user_id: str, amenities: List[str]):
        """Save or replace user's amenities preferences."""
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info:
                user_info.amenities = json.dumps(amenities)
                user_info.updated_at = datetime.utcnow()
            else:
                user_info = UserInfo(user_id=user_id, amenities=json.dumps(amenities))
                session.add(user_info)
            session.commit()
            logger.info(f"Updated amenities for user {user_id}: {amenities}")
        except Exception as e:
            logger.error(f"Error updating user amenities: {e}")
        finally:
            session.close()

    def get_user_amenities(self, user_id: str) -> List[str]:
        """Retrieve user's saved amenities."""
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info and user_info.amenities:
                try:
                    return json.loads(user_info.amenities)
                except:
                    return []
            return []
        finally:
            session.close()

    def update_user_budget(self, user_id: str, budget: str):
        """Save user's budget preference for accommodations."""
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info:
                user_info.budget = budget
                user_info.updated_at = datetime.utcnow()
            else:
                user_info = UserInfo(user_id=user_id, budget=budget)
                session.add(user_info)
            session.commit()
            logger.info(f"Updated budget for user {user_id}: {budget}")
        except Exception as e:
            logger.error(f"Error updating user budget: {e}")
        finally:
            session.close()

    def get_user_budget(self, user_id: str) -> Optional[str]:
        """Retrieve user's saved budget."""
        session = self.db.get_session()
        try:
            user_info = session.query(UserInfo).filter_by(user_id=user_id).first()
            if user_info:
                return user_info.budget
            return None
        finally:
            session.close()