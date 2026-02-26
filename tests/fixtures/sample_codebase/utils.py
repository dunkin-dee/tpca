"""
Utility functions for the application.
"""
import hashlib
import secrets
from typing import Optional


def generate_token(user_id: str, secret: str, expiry: int = 3600) -> str:
    """
    Generate a signed JWT token.
    
    Args:
        user_id: User identifier
        secret: Secret key for signing
        expiry: Token expiry in seconds
    
    Returns:
        Signed JWT token string
    """
    import json
    import base64
    import time
    
    payload = {
        'user_id': user_id,
        'exp': int(time.time()) + expiry,
        'iat': int(time.time())
    }
    
    payload_json = json.dumps(payload)
    encoded = base64.b64encode(payload_json.encode()).decode()
    
    # Simple signature
    signature = _sign_token(encoded, secret)
    
    return f"header.{encoded}.{signature}"


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password with salt.
    
    Args:
        password: Plain text password
        salt: Optional salt (generated if not provided)
    
    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Use SHA256 for hashing
    combined = f"{password}{salt}".encode('utf-8')
    hashed = hashlib.sha256(combined).hexdigest()
    
    return hashed, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password: Plain text password to verify
        hashed: Stored hash
        salt: Salt used for hashing
    
    Returns:
        True if password matches
    """
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == hashed


def _sign_token(payload: str, secret: str) -> str:
    """
    Sign a token payload.
    
    Args:
        payload: Token payload to sign
        secret: Secret key
    
    Returns:
        Signature string
    """
    combined = f"{payload}{secret}".encode('utf-8')
    return hashlib.sha256(combined).hexdigest()[:32]


def format_error(error: Exception, include_traceback: bool = False) -> dict:
    """
    Format an exception as a JSON-serializable dict.
    
    Args:
        error: Exception to format
        include_traceback: Whether to include traceback
    
    Returns:
        Error dict
    """
    result = {
        'error': type(error).__name__,
        'message': str(error)
    }
    
    if include_traceback:
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result
