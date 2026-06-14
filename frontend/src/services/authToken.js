/**
 * In-memory holder for the short-lived access token.
 *
 * The access token is deliberately NOT persisted (XSS-safe): it lives only in
 * this module for the lifetime of the page and is restored via a silent
 * `/auth/refresh` on app start. The auth store is the single writer; the API
 * client and PhentrieveService are readers.
 *
 * @module services/authToken
 */

let accessToken = null;

/** @returns {string|null} the current access token, or null if signed out */
export function getAccessToken() {
  return accessToken;
}

/** @param {string|null} token */
export function setAccessToken(token) {
  accessToken = token || null;
}

export function clearAccessToken() {
  accessToken = null;
}
