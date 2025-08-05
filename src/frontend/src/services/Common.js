export const BASE_API_URL = process.env.REACT_APP_BASE_API_URL;

export function epochToJsDate(ts) {
    let dt = new Date(ts)
    return dt.toLocaleDateString() + " " + dt.toLocaleTimeString();
}