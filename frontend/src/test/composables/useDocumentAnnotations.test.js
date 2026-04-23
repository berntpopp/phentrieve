import { describe, expect, it } from 'vitest';
import {
  buildMarkedSegments,
  buildSelectedAnnotationDecorations,
  getChunkAnnotationDetails,
  getSpanAnnotations,
  needsFallbackMarks,
} from '../../composables/useDocumentAnnotations';

describe('useDocumentAnnotations', () => {
  it('normalizes span annotations and merges adjacent fallback segments', () => {
    const normalizedChunk = {
      chunk_id: 1,
      text: 'Developmental delay was present.',
      evidence_mode: 'span',
      annotations: [
        { start_char: -4, end_char: 19, matched_text_in_chunk: 'Developmental delay' },
        { start_char: 19, end_char: 19, matched_text_in_chunk: 'ignored zero-width range' },
        { start_char: 5, end_char: 200, id: 'ann-2' },
      ],
    };

    expect(getSpanAnnotations(normalizedChunk)).toEqual([
      {
        start_char: 0,
        end_char: 19,
        matched_text_in_chunk: 'Developmental delay',
        id: 'annotation-1-0',
      },
      {
        start_char: 5,
        end_char: normalizedChunk.text.length,
        id: 'ann-2',
      },
    ]);

    const segmentedChunk = {
      chunk_id: 1,
      text: 'Developmental delay was present.',
      evidence_mode: 'span',
      annotations: [{ start_char: 0, end_char: 19, matched_text_in_chunk: 'Developmental delay' }],
    };

    expect(buildMarkedSegments(segmentedChunk, new Set())).toHaveLength(2);
  });

  it('builds selected annotation decoration data for fallback segments', () => {
    const decorations = buildSelectedAnnotationDecorations({
      annotations: [
        { id: 'ann-2', start_char: 5, end_char: 29 },
        {
          id: 'ann-1',
          start_char: 0,
          end_char: 19,
          matched_text_in_chunk: 'Developmental delay',
        },
      ],
      selectedAnnotationIds: new Set(['ann-1']),
      segmentStart: 5,
      segmentEnd: 19,
      text: 'opmental delay',
    });

    expect(decorations).toEqual([
      {
        id: 'ann-1',
        selected: true,
        detailText: 'Developmental delay',
        start_char: 0,
        end_char: 19,
      },
      {
        id: 'ann-2',
        selected: false,
        detailText: 'opmental delay',
        start_char: 5,
        end_char: 29,
      },
    ]);
  });

  it('builds chunk annotation details and fallback mark support from normalized spans', () => {
    const chunk = {
      chunk_id: 2,
      text: 'Seizures and more seizures were reported.',
      evidence_mode: 'span',
      annotations: [
        { id: 'shared-ann', start_char: 0, end_char: 8, matched_text_in_chunk: 'Seizures' },
        { id: 'shared-ann', start_char: 18, end_char: 26, matched_text_in_chunk: 'seizures' },
      ],
    };

    expect(needsFallbackMarks(chunk, false)).toBe(true);
    expect(needsFallbackMarks(chunk, true)).toBe(false);
    expect(getChunkAnnotationDetails(chunk)).toEqual([
      {
        id: 'shared-ann',
        detailText: 'Seizures',
      },
    ]);
  });
});
