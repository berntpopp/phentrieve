import { describe, it, expect } from 'vitest';
import { useAdvancedOptions } from '../../composables/useAdvancedOptions';
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
} from '../../constants/defaults';

describe('useAdvancedOptions', () => {
  it('returns all expected refs and functions', () => {
    const result = useAdvancedOptions();
    expect(result).toHaveProperty('showAdvancedOptions');
    expect(result).toHaveProperty('numResults');
    expect(result).toHaveProperty('similarityThreshold');
    expect(result).toHaveProperty('splitThreshold');
    expect(result).toHaveProperty('chunkRetrievalThreshold');
    expect(result).toHaveProperty('aggregatedTermConfidence');
    expect(result).toHaveProperty('inputTextLengthThreshold');
    expect(result).toHaveProperty('windowSize');
    expect(result).toHaveProperty('stepSize');
    expect(result).toHaveProperty('minSegmentLength');
    expect(result).toHaveProperty('numResultsPerChunk');
    expect(result).toHaveProperty('resetToDefaults');
  });

  it('initializes with correct default values', () => {
    const opts = useAdvancedOptions();
    expect(opts.showAdvancedOptions.value).toBe(false);
    expect(opts.numResults.value).toBe(DEFAULT_NUM_RESULTS);
    expect(opts.similarityThreshold.value).toBe(DEFAULT_SIMILARITY_THRESHOLD);
    expect(opts.splitThreshold.value).toBe(DEFAULT_SPLIT_THRESHOLD);
    expect(opts.chunkRetrievalThreshold.value).toBe(DEFAULT_CHUNK_RETRIEVAL_THRESHOLD);
    expect(opts.aggregatedTermConfidence.value).toBe(DEFAULT_AGGREGATED_TERM_CONFIDENCE);
    expect(opts.inputTextLengthThreshold.value).toBe(DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD);
    expect(opts.windowSize.value).toBe(DEFAULT_WINDOW_SIZE);
    expect(opts.stepSize.value).toBe(DEFAULT_STEP_SIZE);
    expect(opts.minSegmentLength.value).toBe(DEFAULT_MIN_SEGMENT_LENGTH);
    expect(opts.numResultsPerChunk.value).toBe(DEFAULT_NUM_RESULTS_PER_CHUNK);
  });

  it('resetToDefaults resets similarity threshold', () => {
    const { similarityThreshold, resetToDefaults } = useAdvancedOptions();
    similarityThreshold.value = 0.9;
    resetToDefaults();
    expect(similarityThreshold.value).toBe(DEFAULT_SIMILARITY_THRESHOLD);
  });

  it('refs are independent across calls', () => {
    const opts1 = useAdvancedOptions();
    const opts2 = useAdvancedOptions();
    opts1.numResults.value = 99;
    expect(opts2.numResults.value).toBe(DEFAULT_NUM_RESULTS);
  });
});
