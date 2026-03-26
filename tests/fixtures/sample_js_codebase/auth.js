/**
 * Authentication module.
 * Handles JWT token creation, validation, and refresh.
 */

const crypto = require('crypto');
const { hashPassword, verifyPassword } = require('./utils');

/**
 * Core authentication class.
 * Manages JWT lifecycle using a shared config.
 */
class Auth {
    /**
     * @param {Object} config - Configuration object with secret and expirySeconds.
     */
    constructor(config) {
        this.secret = config.secret;
        this.expirySeconds = config.expirySeconds || 3600;
        this._tokenStore = new Map();
    }

    /**
     * Validate a JWT token string.
     * Returns true if the token is valid and not expired.
     * @param {string} token - The JWT token to validate.
     * @returns {boolean}
     */
    validateToken(token) {
        try {
            const payload = this._decodePayload(token);
            if (!payload) return false;
            const now = Math.floor(Date.now() / 1000);
            if (payload.exp && payload.exp < now) {
                throw new Error('Token expired');
            }
            return true;
        } catch (err) {
            return false;
        }
    }

    /**
     * Refresh a valid token, extending its expiry.
     * Raises if the token is already invalid.
     * @param {string} token - Existing valid token.
     * @returns {string} New token with extended expiry.
     */
    refreshToken(token) {
        if (!this.validateToken(token)) {
            throw new Error('Cannot refresh invalid token');
        }
        const payload = this._decodePayload(token);
        return this._createToken(payload.sub, payload.role);
    }

    /**
     * Create a new signed token for a given subject.
     * @param {string} subject - User identifier.
     * @param {string} [role='user'] - User role.
     * @returns {string}
     */
    createToken(subject, role = 'user') {
        return this._createToken(subject, role);
    }

    // ── Private helpers ────────────────────────────────────────────────────

    _decodePayload(token) {
        if (!token || typeof token !== 'string') return null;
        const parts = token.split('.');
        if (parts.length !== 3) return null;
        try {
            const raw = Buffer.from(parts[1], 'base64').toString('utf-8');
            return JSON.parse(raw);
        } catch {
            return null;
        }
    }

    _createToken(subject, role) {
        const now = Math.floor(Date.now() / 1000);
        const payload = {
            sub: subject,
            role,
            iat: now,
            exp: now + this.expirySeconds,
        };
        const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64');
        const body   = Buffer.from(JSON.stringify(payload)).toString('base64');
        const sig    = crypto
            .createHmac('sha256', this.secret)
            .update(`${header}.${body}`)
            .digest('base64');
        return `${header}.${body}.${sig}`;
    }
}

module.exports = { Auth };
