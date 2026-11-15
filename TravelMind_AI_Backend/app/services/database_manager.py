"""Database session management and configuration."""
import sqlite3
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.models.models import Base


class DatabaseManager:
    """
    Database manager that provides both SQLAlchemy sessions and legacy sqlite3 connections.
    
    This class supports both the new SQLAlchemy ORM approach and the legacy raw SQL approach
    during the migration period.
    """
    
    def __init__(self, database_url: str = "sqlite:///tourism_chatbot.db"):
        """
        Initialize the database manager.
        
        Args:
            database_url: SQLAlchemy database URL (default: sqlite:///tourism_chatbot.db)
        """
        self.database_url = database_url
        
        # Create SQLAlchemy engine
        self.engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},  # Needed for SQLite
            echo=False  # Set to True for SQL query logging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Extract database path for legacy sqlite3 connections
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url.replace("sqlite:///", "")
        else:
            self.db_path = "tourism_chatbot.db"
    
    def create_tables(self):
        """Create all tables defined in SQLAlchemy models."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """
        Get a new SQLAlchemy session.
        
        Returns:
            SQLAlchemy Session object
            
        Note:
            Caller is responsible for closing the session.
        """
        return self.SessionLocal()
    
    def get_connection(self):
        """
        Get a legacy sqlite3 connection.
        
        This method is provided for backward compatibility during migration.
        New code should use get_session() instead.
        
        Returns:
            sqlite3.Connection object
            
        Note:
            Caller is responsible for closing the connection.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.
    
    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            items = db.query(Item).all()
            return items
    
    Yields:
        SQLAlchemy Session object
    """
    db = None
    try:
        # This will be initialized in main.py
        from app.main import db_manager
        db = db_manager.get_session()
        yield db
    finally:
        if db:
            db.close()
