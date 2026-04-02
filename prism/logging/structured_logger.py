"""
Structured logging system for PRISM.
All components emit structured JSON events to file, console, and ring buffer.
"""
import json
import os
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler as StdRotatingFileHandler
from pathlib import Path
from typing import Optional

from .console_handler import ConsoleHandler
from .log_config import LogConfig


class StructuredLogger:
    """
    Single logging integration point for all PRISM components.
    
    Three outputs:
    - Rotating JSON-lines log file
    - Console handler (formatted for human readability)
    - In-memory ring buffer (for programmatic access and test assertions)
    """
    
    def __init__(self, config: LogConfig):
        self.config = config
        
        # Ensure log directory exists
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up file handler (rotating)
        self._file_handler = StdRotatingFileHandler(
            config.log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3
        )
        self._file_handler.setLevel(config.file_level)
        
        # Set up console handler
        self._console_handler = ConsoleHandler(level=config.console_level)
        
        # Set up ring buffer
        self._ring_buffer: deque = deque(maxlen=config.ring_buffer_size)
        
        self._level_priority = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3}
    
    def emit(self, level: str, event: str, **fields):
        """
        Emit a structured log event.
        
        Args:
            level: Log level (DEBUG, INFO, WARN, ERROR)
            event: Event type/name
            **fields: Additional structured fields
        """
        record = {
            'ts': datetime.utcnow().isoformat(),
            'level': level,
            'event': event,
            **fields
        }
        
        # Add to ring buffer (always)
        self._ring_buffer.append(record)
        
        # Write to file
        file_priority = self._level_priority.get(self.config.file_level, 0)
        record_priority = self._level_priority.get(level, 1)
        if record_priority >= file_priority:
            self._file_handler.emit(self._make_log_record(json.dumps(record) + '\n'))
        
        # Write to console
        console_priority = self._level_priority.get(self.config.console_level, 1)
        if record_priority >= console_priority:
            formatted = self._console_handler.format(record)
            if formatted:
                self._console_handler.write(formatted)
    
    def _make_log_record(self, message: str):
        """Create a LogRecord for the file handler."""
        import logging
        record = logging.LogRecord(
            name='prism',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        return record
    
    # Convenience methods
    def debug(self, event: str, **kw):
        """Emit a DEBUG level event."""
        self.emit('DEBUG', event, **kw)
    
    def info(self, event: str, **kw):
        """Emit an INFO level event."""
        self.emit('INFO', event, **kw)
    
    def warn(self, event: str, **kw):
        """Emit a WARN level event."""
        self.emit('WARN', event, **kw)
    
    def error(self, event: str, **kw):
        """Emit an ERROR level event."""
        self.emit('ERROR', event, **kw)
    
    def get_events(self, event_type: Optional[str] = None) -> list[dict]:
        """
        Get events from the ring buffer.
        
        Args:
            event_type: Filter by event type. If None, return all events.
        
        Returns:
            List of event dictionaries
        """
        if event_type:
            return [e for e in self._ring_buffer if e.get('event') == event_type]
        return list(self._ring_buffer)
    
    def clear_buffer(self):
        """Clear the ring buffer (useful for tests)."""
        self._ring_buffer.clear()

    def close(self):
        """Close all handlers, releasing file handles. Required on Windows before cleanup."""
        self._file_handler.close()
        self._console_handler.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()