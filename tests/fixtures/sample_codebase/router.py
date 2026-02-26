"""
HTTP routing and request handling.
"""
from typing import Callable, Dict, Any


class Request:
    """Represents an HTTP request."""
    
    def __init__(self, path: str, method: str, headers: dict):
        self.path = path
        self.method = method
        self.headers = headers
        self.body = None


class Response:
    """Represents an HTTP response."""
    
    def __init__(self, status: int, body: Any, headers: dict = None):
        self.status = status
        self.body = body
        self.headers = headers or {}


class Router:
    """
    Main HTTP router.
    Routes requests to registered handlers.
    """
    
    def __init__(self, auth_handler):
        """
        Initialize router with authentication.
        
        Args:
            auth_handler: Auth instance for token validation
        """
        self.auth_handler = auth_handler
        self.routes: Dict[str, Callable] = {}
    
    def register(self, path: str, handler: Callable) -> None:
        """
        Register a handler for a path.
        
        Args:
            path: URL path pattern
            handler: Callable that handles the request
        """
        self.routes[path] = handler
    
    def route(self, request: Request) -> Response:
        """
        Route a request to the appropriate handler.
        
        Args:
            request: HTTP request to route
        
        Returns:
            HTTP response from handler
        """
        # Check authentication if auth token is present
        auth_token = request.headers.get('Authorization', '')
        if auth_token:
            if not self._validate_auth(auth_token):
                return Response(401, {'error': 'Unauthorized'})
        
        # Find handler
        handler = self.routes.get(request.path)
        if not handler:
            return self._not_found(request)
        
        # Execute handler
        try:
            return handler(request)
        except Exception as e:
            return self._error_response(e)
    
    def _validate_auth(self, token: str) -> bool:
        """
        Validate authentication token.
        
        Args:
            token: JWT token from request
        
        Returns:
            True if valid
        """
        try:
            # Strip 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            return self.auth_handler.validate_token(token)
        except ValueError:
            return False
    
    def _not_found(self, request: Request) -> Response:
        """Generate 404 response."""
        return Response(404, {
            'error': 'Not found',
            'path': request.path
        })
    
    def _error_response(self, error: Exception) -> Response:
        """Generate error response."""
        return Response(500, {
            'error': 'Internal server error',
            'message': str(error)
        })
