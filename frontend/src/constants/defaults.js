/**
 * Default values for query parameters and thresholds.
 * Single source of truth — used by QueryInterface and API calls.
 */

// Query defaults
export const DEFAULT_NUM_RESULTS = 10
export const DEFAULT_SIMILARITY_THRESHOLD = 0.5
export const DEFAULT_SPLIT_THRESHOLD = 0.5

// Chunking defaults
export const DEFAULT_CHUNK_RETRIEVAL_THRESHOLD = 0.7
export const DEFAULT_AGGREGATED_TERM_CONFIDENCE = 0.75
export const DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD = 120
export const DEFAULT_WINDOW_SIZE = 3
export const DEFAULT_STEP_SIZE = 1
export const DEFAULT_MIN_SEGMENT_LENGTH = 2
export const DEFAULT_NUM_RESULTS_PER_CHUNK = 3

// Similarity score quality thresholds
export const SCORE_EXCELLENT = 0.9
export const SCORE_GOOD = 0.75
export const SCORE_MODERATE = 0.6
export const SCORE_LOW = 0.4
export const SCORE_ANIMATION_TRIGGER = 0.85
