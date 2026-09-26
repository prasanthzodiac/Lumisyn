const configuredBase = import.meta.env.VITE_API_BASE_URL || "";
export const API_BASE_URL = configuredBase.replace(/\/$/, "");

export function apiUrl(path) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

export async function apiFetch(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the HTTP error when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response;
}
