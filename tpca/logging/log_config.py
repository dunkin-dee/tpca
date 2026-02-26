from dataclasses import dataclass


@dataclass
class LogConfig:
    """Configuration for the StructuredLogger."""
    
    log_file: str = '.tpca_cache/tpca.log'
    console_level: str = 'INFO'   # DEBUG|INFO|WARN|ERROR
    file_level: str = 'DEBUG'     # always verbose to file
    ring_buffer_size: int = 1000
    include_prompt_text: bool = False  # redact prompts from logs by default
