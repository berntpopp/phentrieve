import { ref } from 'vue';
import {
  DEFAULT_NUM_RESULTS,
  DEFAULT_SIMILARITY_THRESHOLD,
  DEFAULT_SPLIT_THRESHOLD,
  DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
  DEFAULT_AGGREGATED_TERM_CONFIDENCE,
  DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD,
  DEFAULT_WINDOW_SIZE,
  DEFAULT_STEP_SIZE,
  DEFAULT_MIN_SEGMENT_LENGTH,
  DEFAULT_NUM_RESULTS_PER_CHUNK,
} from '../constants/defaults';

/**
 * Composable for managing advanced query options state.
 * Extracted from QueryInterface.vue to reduce component size.
 */
export function useAdvancedOptions() {
  const showAdvancedOptions = ref(false);
  const numResults = ref(DEFAULT_NUM_RESULTS);
  const similarityThreshold = ref(DEFAULT_SIMILARITY_THRESHOLD);
  const splitThreshold = ref(DEFAULT_SPLIT_THRESHOLD);
  const chunkRetrievalThreshold = ref(DEFAULT_CHUNK_RETRIEVAL_THRESHOLD);
  const aggregatedTermConfidence = ref(DEFAULT_AGGREGATED_TERM_CONFIDENCE);
  const inputTextLengthThreshold = ref(DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD);
  const windowSize = ref(DEFAULT_WINDOW_SIZE);
  const stepSize = ref(DEFAULT_STEP_SIZE);
  const minSegmentLength = ref(DEFAULT_MIN_SEGMENT_LENGTH);
  const numResultsPerChunk = ref(DEFAULT_NUM_RESULTS_PER_CHUNK);

  function toggleAdvancedOptions() {
    showAdvancedOptions.value = !showAdvancedOptions.value;
  }

  function resetToDefaults() {
    similarityThreshold.value = DEFAULT_SIMILARITY_THRESHOLD;
  }

  return {
    showAdvancedOptions,
    numResults,
    similarityThreshold,
    splitThreshold,
    chunkRetrievalThreshold,
    aggregatedTermConfidence,
    inputTextLengthThreshold,
    windowSize,
    stepSize,
    minSegmentLength,
    numResultsPerChunk,
    toggleAdvancedOptions,
    resetToDefaults,
  };
}
