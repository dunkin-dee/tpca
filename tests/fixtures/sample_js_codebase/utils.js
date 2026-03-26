/**
 * Utility functions — shared across the application.
 */

const crypto = require('crypto');

/**
 * Hash a plain-text password using SHA-256 + salt.
 * Returns a hex digest string.
 * @param {string} password - Plain-text password.
 * @param {string} [salt] - Optional salt; generated if omitted.
 * @returns {{ hash: string, salt: string }}
 */
function hashPassword(password, salt = null) {
    const usedSalt = salt || crypto.randomBytes(16).toString('hex');
    const hash = crypto
        .createHmac('sha256', usedSalt)
        .update(password)
        .digest('hex');
    return { hash, salt: usedSalt };
}

/**
 * Verify a plain-text password against a stored hash + salt.
 * @param {string} password - Plain-text password to check.
 * @param {string} hash - Stored hash string.
 * @param {string} salt - Salt used when hash was created.
 * @returns {boolean}
 */
function verifyPassword(password, hash, salt) {
    const { hash: computed } = hashPassword(password, salt);
    return computed === hash;
}

/**
 * Generate a URL-safe random token of *byteLength* bytes.
 * @param {number} [byteLength=32]
 * @returns {string}
 */
function generateToken(byteLength = 32) {
    return crypto.randomBytes(byteLength).toString('base64url');
}

/**
 * Simple rate-limiter: returns true if the key has not exceeded *maxCalls*
 * within the last *windowMs* milliseconds.
 * @param {Map} store - Shared in-memory store for tracking calls.
 * @param {string} key - Identifier (e.g. IP address or user ID).
 * @param {number} maxCalls
 * @param {number} windowMs
 * @returns {boolean}
 */
function checkRateLimit(store, key, maxCalls, windowMs) {
    const now = Date.now();
    const record = store.get(key) || { count: 0, resetAt: now + windowMs };
    if (now > record.resetAt) {
        record.count = 0;
        record.resetAt = now + windowMs;
    }
    record.count += 1;
    store.set(key, record);
    return record.count <= maxCalls;
}

module.exports = { hashPassword, verifyPassword, generateToken, checkRateLimit };
