import sys
from typing import TextIO


class ConsoleHandler:
    """Formats and writes log records to console in human-readable format."""
    
    LEVEL_COLORS = {
        'DEBUG': '\033[36m',  # cyan
        'INFO':  '\033[32m',  # green
        'WARN':  '\033[33m',  # yellow
        'ERROR': '\033[31m',  # red
    }
    RESET = '\033[0m'
    
    def __init__(self, level: str = 'INFO', stream: TextIO = None):
        self.level = level
        self.stream = stream or sys.stderr
        self.level_priority = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3}
        self.min_priority = self.level_priority.get(level, 1)
    
    def write(self, formatted_message: str):
        """Write a formatted message to the console."""
        self.stream.write(formatted_message)
        self.stream.flush()
    
    def format(self, record: dict) -> str:
        """Format a log record for console display."""
        level = record.get('level', 'INFO')
        
        # Skip if below minimum level
        if self.level_priority.get(level, 1) < self.min_priority:
            return ''
        
        event = record.get('event', '')
        color = self.LEVEL_COLORS.get(level, '')
        
        # Format: [LEVEL] event: field=value field2=value2
        parts = [f"{color}[{level}]{self.RESET} {event}"]
        
        # Add non-standard fields
        skip_fields = {'ts', 'level', 'event'}
        for key, value in record.items():
            if key not in skip_fields:
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + '...'
                parts.append(f"{key}={value}")
        
        return ' '.join(parts) + '\n'

    def close(self):
        """No-op close for interface compatibility. stderr is not owned by this handler."""
        pass