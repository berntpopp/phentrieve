import { describe, it, expect } from 'vitest';
import { shallowMount } from '@vue/test-utils';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import AdvancedOptionsPanel from '../../components/AdvancedOptionsPanel.vue';
import {
  DEFAULT_AGGREGATED_TERM_CONFIDENCE,
  DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
  DEFAULT_MIN_SEGMENT_LENGTH,
  DEFAULT_NUM_RESULTS_PER_CHUNK,
  DEFAULT_SIMILARITY_THRESHOLD,
  DEFAULT_SPLIT_THRESHOLD,
  DEFAULT_STEP_SIZE,
  DEFAULT_WINDOW_SIZE,
} from '../../constants/defaults';

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

describe('AdvancedOptionsPanel', () => {
  it('takes its prop defaults from constants/defaults.js (#426)', () => {
    const wrapper = shallowMount(AdvancedOptionsPanel, { global: { plugins: [i18n] } });

    expect(wrapper.props()).toMatchObject({
      similarityThreshold: DEFAULT_SIMILARITY_THRESHOLD,
      windowSize: DEFAULT_WINDOW_SIZE,
      stepSize: DEFAULT_STEP_SIZE,
      chunkRetrievalThreshold: DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
      aggregatedTermConfidence: DEFAULT_AGGREGATED_TERM_CONFIDENCE,
      splitThreshold: DEFAULT_SPLIT_THRESHOLD,
      minSegmentLength: DEFAULT_MIN_SEGMENT_LENGTH,
      numResultsPerChunk: DEFAULT_NUM_RESULTS_PER_CHUNK,
    });
  });
});
