import { describe, it, expect } from 'vitest'
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
  SCORE_EXCELLENT,
  SCORE_GOOD,
  SCORE_MODERATE,
  SCORE_LOW,
  SCORE_ANIMATION_TRIGGER,
} from '../constants/defaults'
import { HPO_TERM_URL, GITHUB_REPO_URL, PHENTRIEVE_PRODUCTION_URL } from '../constants/urls'

describe('constants/defaults', () => {
  it('exports numeric query defaults', () => {
    expect(DEFAULT_NUM_RESULTS).toBe(10)
    expect(DEFAULT_SIMILARITY_THRESHOLD).toBe(0.5)
    expect(DEFAULT_SPLIT_THRESHOLD).toBe(0.5)
  })

  it('exports numeric chunking defaults', () => {
    expect(DEFAULT_CHUNK_RETRIEVAL_THRESHOLD).toBe(0.7)
    expect(DEFAULT_AGGREGATED_TERM_CONFIDENCE).toBe(0.75)
    expect(DEFAULT_INPUT_TEXT_LENGTH_THRESHOLD).toBe(120)
    expect(DEFAULT_WINDOW_SIZE).toBe(3)
    expect(DEFAULT_STEP_SIZE).toBe(1)
    expect(DEFAULT_MIN_SEGMENT_LENGTH).toBe(2)
    expect(DEFAULT_NUM_RESULTS_PER_CHUNK).toBe(3)
  })

  it('exports score thresholds in descending order', () => {
    expect(SCORE_EXCELLENT).toBeGreaterThan(SCORE_GOOD)
    expect(SCORE_GOOD).toBeGreaterThan(SCORE_MODERATE)
    expect(SCORE_MODERATE).toBeGreaterThan(SCORE_LOW)
  })

  it('exports animation trigger threshold', () => {
    expect(SCORE_ANIMATION_TRIGGER).toBe(0.85)
  })
})

describe('constants/urls', () => {
  it('HPO_TERM_URL generates correct URL', () => {
    expect(HPO_TERM_URL('HP:0001234')).toBe('https://hpo.jax.org/browse/term/HP:0001234')
  })

  it('GITHUB_REPO_URL is a valid URL', () => {
    expect(GITHUB_REPO_URL).toContain('github.com')
    expect(GITHUB_REPO_URL).toContain('phentrieve')
  })

  it('PHENTRIEVE_PRODUCTION_URL is a valid URL', () => {
    expect(PHENTRIEVE_PRODUCTION_URL).toContain('https://')
  })
})
