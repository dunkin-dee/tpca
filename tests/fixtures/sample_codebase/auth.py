"""
Authentication module for the application.
"""
from typing import Optional


class BaseAuth:
    """Base authentication class."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config


class Auth(BaseAuth):
    """
    Main authentication handler.
    Validates JWT tokens and manages user sessions.
    """
    
    def __init__(self, config: dict):
        """
        Initialize authentication with configuration.
        
        Args:
            config: Configuration dictionary with jwt_secret and expiry
        """
        super().__init__(config)
        self.jwt_secret = config.get('jwt_secret', 'default_secret')
        self.expiry_seconds = config.get('expiry', 3600)
    
    def validate_token(self, token: str) -> bool:
        """
        Validates a JWT token.
        
        Args:
            token: JWT token string to validate
        
        Returns:
            True if valid, False otherwise
        
        Raises:
            ValueError: If token is expired or malformed
        """
        if not token:
            return False
        
        payload = self._decode_payload(token)
        if not payload:
            return False
        
        # Check expiry
        import time
        if time.time() > payload.get('exp', 0):
            raise ValueError("Token expired")
        
        return True
    
    def refresh_token(self, token: str) -> str:
        """
        Refresh an existing token with a new expiry.
        
        Args:
            token: Existing JWT token
        
        Returns:
            New JWT token string
        """
        payload = self._decode_payload(token)
        if payload:
            return self._encode_token(payload)
        return ""
    
    def _decode_payload(self, token: str) -> Optional[dict]:
        """
        Decode JWT payload.
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded payload dict or None if invalid
        """
        # Simplified - real implementation would use PyJWT
        try:
            import base64
            parts = token.split('.')
            if len(parts) != 3:
                return None
            payload_part = parts[1]
            # Add padding if needed
            padding = 4 - len(payload_part) % 4
            if padding != 4:
                payload_part += '=' * padding
            decoded = base64.b64decode(payload_part)
            import json
            return json.loads(decoded)
        except Exception:
            return None
    
    def _encode_token(self, payload: dict) -> str:
        """
        Encode payload to JWT.
        
        Args:
            payload: Dictionary to encode
        
        Returns:
            JWT token string
        """
        # Simplified implementation
        import base64
        import json
        import time
        
        payload['exp'] = int(time.time()) + self.expiry_seconds
        payload_json = json.dumps(payload)
        encoded = base64.b64encode(payload_json.encode()).decode()
        
        # Simplified: header.payload.signature
        return f"header.{encoded}.signature"
