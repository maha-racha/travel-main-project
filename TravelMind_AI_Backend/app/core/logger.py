import logging
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with a dedicated file handler.
    Loggers will save logs to data/logs/{name}.log
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if a FileHandler for this specific file already exists to prevent duplicates
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and 
        Path(h.baseFilename).name == f"{name}.log"
        for h in logger.handlers
    )
    
    if not has_file_handler:
        # Create data/logs directory if it doesn't exist
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
