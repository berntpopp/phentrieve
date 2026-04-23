import { describe, expect, it } from 'vitest';
import {
  buildUserNoteSegments,
  formatDocumentSummaryMeta,
  resolveChunkOffsetsInNote,
  resolveMatchedTextRange,
  summarizeDocumentQuery,
} from '../../composables/useUserNoteAnnotations';

describe('useUserNoteAnnotations', () => {
  it('maps chunk offsets into note segments', () => {
    const note = 'Patient had seizures. Developmental delay documented.';
    const chunks = [
      { chunk_id: 1, text: 'Patient had seizures.' },
      { chunk_id: 2, text: 'Developmental delay documented.' },
    ];
    const terms = [
      {
        hpo_id: 'HP:0001250',
        name: 'Seizure',
        text_attributions: [{ chunk_id: 1, start_char: 12, end_char: 20 }],
      },
    ];

    const segments = buildUserNoteSegments({
      note,
      chunks,
      terms,
      activePhenotypeId: null,
    });

    expect(segments.some((segment) => segment.highlighted)).toBe(true);
    expect(segments.find((segment) => segment.highlighted)?.text).toBe('seizures');
  });

  it('resolves chunk offsets in order within the note', () => {
    const offsets = resolveChunkOffsetsInNote('alpha beta gamma', [
      { chunk_id: 1, text: 'alpha beta' },
      { chunk_id: 2, text: 'gamma' },
    ]);

    expect(offsets.get(1)).toBe(0);
    expect(offsets.get('1')).toBe(0);
    expect(offsets.get(2)).toBe(11);
  });

  it('falls back to matched text lookup when chunk offsets are unavailable', () => {
    expect(resolveMatchedTextRange('Patient had seizures in clinic.', 'seizures')).toEqual({
      start: 12,
      end: 20,
    });
  });

  it('formats document summary metadata from word count', () => {
    expect(formatDocumentSummaryMeta('Patient had recurrent seizures.')).toBe('4 words');
  });

  it('summarizes long document text for the collapsed preview', () => {
    const longQuery =
      'Patient had recurrent seizures and developmental delay documented across multiple visits with additional family history details.';

    expect(summarizeDocumentQuery(longQuery)).toContain('...');
    expect(summarizeDocumentQuery('short note')).toBe('short note');
  });
});
