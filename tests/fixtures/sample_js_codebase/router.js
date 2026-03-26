/**
 * HTTP Router — maps URL paths to handler functions.
 * Integrates with Auth for protected route validation.
 */

const { Auth } = require('./auth');

/**
 * Simple path-based HTTP router with middleware support.
 */
class Router {
    constructor() {
        this._routes = new Map();
        this._auth = null;
    }

    /**
     * Attach an Auth instance for protected routes.
     * @param {Auth} auth - Configured Auth instance.
     */
    useAuth(auth) {
        this._auth = auth;
    }

    /**
     * Register a handler for a given path.
     * @param {string} path - URL path pattern (e.g. '/api/users').
     * @param {Function} handler - Request handler (req, res) => void.
     * @param {Object} [options] - Options: { protected: boolean }.
     */
    register(path, handler, options = {}) {
        this._routes.set(path, { handler, options });
    }

    /**
     * Route a request to the appropriate handler.
     * Returns a response-like object: { status, body }.
     * @param {Object} request - { path, method, headers, body }.
     * @returns {{ status: number, body: any }}
     */
    route(request) {
        const route = this._routes.get(request.path);
        if (!route) {
            return { status: 404, body: { error: 'Not found' } };
        }

        if (route.options.protected) {
            const authed = this._validateAuth(request);
            if (!authed) {
                return { status: 401, body: { error: 'Unauthorized' } };
            }
        }

        try {
            return route.handler(request);
        } catch (err) {
            return { status: 500, body: { error: err.message } };
        }
    }

    // ── Private ────────────────────────────────────────────────────────────

    /**
     * Validate the Bearer token in request headers.
     * @param {Object} request
     * @returns {boolean}
     */
    _validateAuth(request) {
        if (!this._auth) return false;
        const authHeader = (request.headers || {}).authorization || '';
        const token = authHeader.replace(/^Bearer\s+/i, '');
        return this._auth.validateToken(token);
    }
}

module.exports = { Router };
