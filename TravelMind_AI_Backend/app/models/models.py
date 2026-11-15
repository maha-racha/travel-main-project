from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Trip(Base):
    """Trip model representing a user's travel plan."""
    __tablename__ = "trips"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Required Fields
    user_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    
    # Optional Fields
    start_date = Column(String)
    end_date = Column(String)
    budget = Column(String)
    preferences = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    itinerary_days = relationship("ItineraryDay", back_populates="trip", cascade="all, delete-orphan")
    conversation_memory = relationship("ConversationMemory", back_populates="trip", cascade="all, delete-orphan")
    conversation_messages = relationship("ConversationMessage", back_populates="trip", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Trip(id={self.id}, user_id={self.user_id}, title={self.title})>"


class ItineraryDay(Base):
    """Itinerary day model representing a single day in a trip."""
    __tablename__ = "itinerary_days"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    trip_id = Column(Integer, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False)
    
    # Required Fields
    day_number = Column(Integer, nullable=False)
    
    # Optional Fields
    date = Column(String)
    description = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trip = relationship("Trip", back_populates="itinerary_days")
    itinerary_items = relationship("ItineraryItem", back_populates="day", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ItineraryDay(id={self.id}, trip_id={self.trip_id}, day_number={self.day_number})>"


class ItineraryItem(Base):
    """Itinerary item model representing an activity or event in a day."""
    __tablename__ = "itinerary_items"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    day_id = Column(Integer, ForeignKey("itinerary_days.id", ondelete="CASCADE"), nullable=False)
    
    # Required Fields
    name = Column(String, nullable=False)
    
    # Optional Fields
    activity_type = Column(String)
    description = Column(Text)
    location = Column(String)
    time = Column(String)
    booking_info = Column(Text)
    order_index = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    day = relationship("ItineraryDay", back_populates="itinerary_items")

    def __repr__(self):
        return f"<ItineraryItem(id={self.id}, name={self.name}, day_id={self.day_id})>"


class ConversationMemory(Base):
    """Conversation memory model for storing conversation summaries."""
    __tablename__ = "conversation_memory"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    trip_id = Column(Integer, ForeignKey("trips.id", ondelete="CASCADE"))
    
    # Required Fields
    user_id = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    
    # Optional Fields
    message_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trip = relationship("Trip", back_populates="conversation_memory")

    def __repr__(self):
        return f"<ConversationMemory(id={self.id}, user_id={self.user_id}, messages={self.message_count})>"


class ConversationMessage(Base):
    """Conversation message model for storing individual chat messages."""
    __tablename__ = "conversation_messages"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    trip_id = Column(Integer, ForeignKey("trips.id", ondelete="CASCADE"))
    
    # Required Fields
    user_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    
    # Optional Fields
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trip = relationship("Trip", back_populates="conversation_messages")

    def __repr__(self):
        return f"<ConversationMessage(id={self.id}, user_id={self.user_id}, role={self.role})>"


class SyncMetadata(Base):
    """Sync metadata model for tracking document synchronization."""
    __tablename__ = "sync_metadata"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Required Fields
    last_sync_time = Column(DateTime, nullable=False)
    
    # Optional Fields
    documents_synced = Column(Integer, default=0)
    chunks_synced = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    added = Column(Integer, default=0)
    updated = Column(Integer, default=0)
    deleted = Column(Integer, default=0)
    skipped = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SyncMetadata(id={self.id}, documents={self.documents_synced}, chunks={self.chunks_synced})>"


class ChatMemory(Base):
    """Chat memory model for storing user conversation context."""
    __tablename__ = "chat_memory"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Required Fields
    user_id = Column(String, nullable=False)
    memory_summary = Column(Text, nullable=False)
    
    # Timestamps
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ChatMemory(id={self.id}, user_id={self.user_id})>"


class UserInfo(Base):
    """User info model for storing user preferences and current city."""
    __tablename__ = "user_info"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Required Fields
    user_id = Column(String, unique=True, nullable=False, index=True)
    
    # Optional Fields
    current_city = Column(String)
    language = Column(String)
    preferences = Column(Text)
    amenities = Column(Text, default="[]") # JSON list of amenity strings
    seen_items = Column(Text, default="[]") # JSON list of item IDs
    budget = Column(String) # Price range for accommodations (e.g., "100-200")
    
    # Timestamps
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<UserInfo(id={self.id}, user_id={self.user_id}, city={self.current_city})>"