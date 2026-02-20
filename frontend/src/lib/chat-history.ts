/**
 * Chat History Service
 * 
 * Manages chat history persistence in localStorage with course-specific keys.
 * Implements getHistory, saveHistory, clearHistory functions.
 * 
 * Requirements: 4.1, 4.2, 4.3
 */

import { ChatMessage, ChunkReference } from "./api";

/**
 * Extended message interface for history storage
 * Includes timestamp, response time, and source references
 */
export interface StoredMessage extends ChatMessage {
  timestamp: string;
  responseTime?: number;
  sources?: ChunkReference[];
}

/**
 * Chat history storage format for a course
 */
export interface CourseHistory {
  courseId: number;
  courseName?: string;
  messages: StoredMessage[];
  lastUpdated: string;
}

const STORAGE_KEY_PREFIX = "student_chat_history_";

/**
 * Get the localStorage key for a specific course
 */
function getStorageKey(courseId: number): string {
  return `${STORAGE_KEY_PREFIX}${courseId}`;
}

/**
 * Check if localStorage is available
 */
function isLocalStorageAvailable(): boolean {
  try {
    const testKey = "__test__";
    localStorage.setItem(testKey, testKey);
    localStorage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get chat history for a specific course
 * Returns an empty array if no history exists or on error
 * 
 * @param courseId - The course ID to get history for
 * @returns Array of stored messages
 */
export function getHistory(courseId: number): StoredMessage[] {
  if (!isLocalStorageAvailable()) {
    return [];
  }

  try {
    const storageKey = getStorageKey(courseId);
    const saved = localStorage.getItem(storageKey);
    
    if (!saved) {
      return [];
    }

    const parsed = JSON.parse(saved);
    
    // Handle both old format (array) and new format (CourseHistory object)
    if (Array.isArray(parsed)) {
      return parsed as StoredMessage[];
    }
    
    if (parsed && typeof parsed === "object" && Array.isArray(parsed.messages)) {
      return parsed.messages as StoredMessage[];
    }

    return [];
  } catch {
    // Invalid JSON or other error, return empty array
    return [];
  }
}

/**
 * Save chat history for a specific course
 * Overwrites any existing history for the course
 * 
 * @param courseId - The course ID to save history for
 * @param messages - Array of messages to save
 * @param courseName - Optional course name for metadata
 */
export function saveHistory(
  courseId: number, 
  messages: StoredMessage[], 
  courseName?: string
): void {
  if (!isLocalStorageAvailable()) {
    return;
  }

  try {
    const storageKey = getStorageKey(courseId);
    const history: CourseHistory = {
      courseId,
      courseName,
      messages,
      lastUpdated: new Date().toISOString(),
    };
    localStorage.setItem(storageKey, JSON.stringify(history));
  } catch (error) {
    // Handle quota exceeded or other storage errors
    console.error("Failed to save chat history:", error);
  }
}

/**
 * Clear chat history for a specific course
 * 
 * @param courseId - The course ID to clear history for
 */
export function clearHistory(courseId: number): void {
  if (!isLocalStorageAvailable()) {
    return;
  }

  try {
    const storageKey = getStorageKey(courseId);
    localStorage.removeItem(storageKey);
  } catch (error) {
    console.error("Failed to clear chat history:", error);
  }
}

/**
 * Get all course IDs that have stored chat history
 * 
 * @returns Array of course IDs with stored history
 */
export function getAllCourseIds(): number[] {
  if (!isLocalStorageAvailable()) {
    return [];
  }

  const courseIds: number[] = [];
  
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(STORAGE_KEY_PREFIX)) {
        const courseIdStr = key.substring(STORAGE_KEY_PREFIX.length);
        const courseId = Number.parseInt(courseIdStr, 10);
        if (!Number.isNaN(courseId)) {
          courseIds.push(courseId);
        }
      }
    }
  } catch {
    // Ignore errors
  }

  return courseIds;
}

/**
 * Get the full course history object including metadata
 * 
 * @param courseId - The course ID to get history for
 * @returns CourseHistory object or null if not found
 */
export function getCourseHistory(courseId: number): CourseHistory | null {
  if (!isLocalStorageAvailable()) {
    return null;
  }

  try {
    const storageKey = getStorageKey(courseId);
    const saved = localStorage.getItem(storageKey);
    
    if (!saved) {
      return null;
    }

    const parsed = JSON.parse(saved);
    
    // Handle old format (array) - convert to new format
    if (Array.isArray(parsed)) {
      return {
        courseId,
        messages: parsed as StoredMessage[],
        lastUpdated: new Date().toISOString(),
      };
    }
    
    if (parsed && typeof parsed === "object") {
      return parsed as CourseHistory;
    }

    return null;
  } catch {
    return null;
  }
}

/**
 * Create a new stored message with timestamp
 * 
 * @param role - Message role (user or assistant)
 * @param content - Message content
 * @param sources - Optional source references
 * @param responseTime - Optional response time in ms
 * @returns StoredMessage object
 */
export function createMessage(
  role: "user" | "assistant",
  content: string,
  sources?: ChunkReference[],
  responseTime?: number
): StoredMessage {
  return {
    role,
    content,
    timestamp: new Date().toISOString(),
    sources,
    responseTime,
  };
}

/**
 * Format timestamp for display
 * 
 * @param timestamp - ISO timestamp string
 * @returns Formatted time string (HH:MM)
 */
export function formatTimestamp(timestamp?: string): string {
  if (!timestamp) return "";
  
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString("tr-TR", { 
      hour: "2-digit", 
      minute: "2-digit" 
    });
  } catch {
    return "";
  }
}

/**
 * Format response time for display
 * 
 * @param ms - Response time in milliseconds
 * @returns Formatted string (e.g., "1.2s" or "500ms")
 */
export function formatResponseTime(ms?: number): string {
  if (!ms) return "";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
