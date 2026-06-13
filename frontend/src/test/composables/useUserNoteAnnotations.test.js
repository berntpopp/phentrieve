import { describe, expect, it } from 'vitest';
import {
  buildUserNoteSegments,
  buildSegmentsFromAnnotations,
  computeSelectionOffsets,
  deriveFindingsFromAnnotations,
  formatDocumentSummaryMeta,
  resolveChunkOffsetsInNote,
  resolveMatchedTextRange,
  seedAnnotationsFromResponse,
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

  it('keeps every phenotype highlighted while one is active (no hover collapse)', () => {
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
      {
        hpo_id: 'HP:0001263',
        name: 'Global developmental delay',
        text_attributions: [{ chunk_id: 2, start_char: 0, end_char: 19 }],
      },
    ];

    const highlightedTexts = (activePhenotypeId) =>
      buildUserNoteSegments({ note, chunks, terms, activePhenotypeId })
        .filter((segment) => segment.highlighted)
        .map((segment) => segment.text);

    // Both mentions are highlighted at rest...
    expect(highlightedTexts(null)).toEqual(['seizures', 'Developmental delay']);
    // ...and stay highlighted even when one phenotype is the active/hovered one.
    expect(highlightedTexts('HP:0001250')).toEqual(['seizures', 'Developmental delay']);
    expect(highlightedTexts('HP:0001263')).toEqual(['seizures', 'Developmental delay']);
  });

  it('produces identical segment structure regardless of the active phenotype', () => {
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
      {
        hpo_id: 'HP:0001263',
        name: 'Global developmental delay',
        text_attributions: [{ chunk_id: 2, start_char: 0, end_char: 19 }],
      },
    ];

    const keysFor = (activePhenotypeId) =>
      buildUserNoteSegments({ note, chunks, terms, activePhenotypeId }).map(
        (segment) => segment.key
      );

    // Stable keys => no DOM re-creation on hover => mouseleave always fires => no stuck highlight.
    expect(keysFor('HP:0001250')).toEqual(keysFor(null));
    expect(keysFor('HP:0001263')).toEqual(keysFor(null));
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

describe('seedAnnotationsFromResponse', () => {
  const note = 'She has microcephaly and feeding problems.';
  const response = {
    processed_chunks: [{ chunk_id: 1, text: 'She has microcephaly and feeding problems.' }],
    aggregated_hpo_terms: [
      {
        hpo_id: 'HP:0000252',
        name: 'Microcephaly',
        status: 'present',
        confidence: 1,
        text_attributions: [
          { chunk_id: 1, start_char: 8, end_char: 20, matched_text_in_chunk: 'microcephaly' },
        ],
      },
      {
        hpo_id: 'HP:0011968',
        name: 'Feeding difficulties',
        status: 'present',
        confidence: 0.9,
        text_attributions: [
          { chunk_id: 1, start_char: 25, end_char: 41, matched_text_in_chunk: 'feeding problems' },
        ],
      },
    ],
  };

  it('builds note-relative auto annotations with origin auto', () => {
    const result = seedAnnotationsFromResponse({ note, response });
    expect(result).toHaveLength(2);
    const micro = result.find((a) => a.hpoId === 'HP:0000252');
    expect(micro.label).toBe('Microcephaly');
    expect(micro.origin).toBe('auto');
    expect(micro.status).toBe('affirmed');
    expect(micro.confidence).toBe(1);
    expect(micro.spans[0]).toMatchObject({ start: 8, end: 20, text: 'microcephaly' });
  });

  it('maps absent/present status to negated/affirmed', () => {
    const r = seedAnnotationsFromResponse({
      note,
      response: {
        ...response,
        aggregated_hpo_terms: [{ ...response.aggregated_hpo_terms[0], status: 'absent' }],
      },
    });
    expect(r[0].status).toBe('negated');
  });

  it('keeps a term with no resolvable span (empty spans) so it still appears in findings', () => {
    const r = seedAnnotationsFromResponse({
      note,
      response: {
        processed_chunks: [{ chunk_id: 9, text: 'unrelated' }],
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:9999999',
            name: 'Inferred term',
            status: 'present',
            confidence: 0.5,
            text_attributions: [
              { chunk_id: 9, start_char: 0, end_char: 0, matched_text_in_chunk: 'nowhere-in-note' },
            ],
          },
        ],
      },
    });
    expect(r).toHaveLength(1);
    expect(r[0].spans).toHaveLength(0);
    expect(deriveFindingsFromAnnotations(r)).toHaveLength(1);
  });

  it('falls back to matched-text search when offsets do not resolve', () => {
    const r = seedAnnotationsFromResponse({
      note,
      response: {
        processed_chunks: [{ chunk_id: 9, text: 'unrelated' }],
        aggregated_hpo_terms: [
          {
            hpo_id: 'HP:0000252',
            name: 'Microcephaly',
            status: 'present',
            confidence: 1,
            text_attributions: [
              { chunk_id: 9, start_char: 0, end_char: 0, matched_text_in_chunk: 'microcephaly' },
            ],
          },
        ],
      },
    });
    expect(r[0].spans[0]).toMatchObject({ start: 8, end: 20 });
  });
});

describe('buildSegmentsFromAnnotations', () => {
  const note = 'She has microcephaly and feeding problems.';
  const annotations = [
    {
      id: 'a1',
      hpoId: 'HP:0000252',
      label: 'Microcephaly',
      status: 'affirmed',
      origin: 'auto',
      spans: [{ start: 8, end: 20, text: 'microcephaly' }],
    },
    {
      id: 'a2',
      hpoId: 'HP:0011968',
      label: 'Feeding difficulties',
      status: 'affirmed',
      origin: 'manual',
      spans: [{ start: 25, end: 41, text: 'feeding problems' }],
    },
  ];

  it('emits highlighted + plain segments in order with termIds and annotationIds', () => {
    const segs = buildSegmentsFromAnnotations(note, annotations);
    const highlighted = segs.filter((s) => s.highlighted);
    expect(highlighted).toHaveLength(2);
    expect(highlighted[0].termIds).toContain('HP:0000252');
    expect(highlighted[0].annotationIds).toContain('a1');
    expect(segs.map((s) => s.text).join('')).toBe(note);
  });

  it('merges overlapping spans and unions termIds', () => {
    const overlap = [
      {
        id: 'x',
        hpoId: 'HP:1',
        label: 'A',
        status: 'affirmed',
        origin: 'auto',
        spans: [{ start: 8, end: 16, text: note.slice(8, 16) }],
      },
      {
        id: 'y',
        hpoId: 'HP:2',
        label: 'B',
        status: 'affirmed',
        origin: 'auto',
        spans: [{ start: 12, end: 20, text: note.slice(12, 20) }],
      },
    ];
    const segs = buildSegmentsFromAnnotations(note, overlap).filter((s) => s.highlighted);
    expect(segs).toHaveLength(1);
    expect(segs[0].termIds).toEqual(expect.arrayContaining(['HP:1', 'HP:2']));
    expect(segs[0].annotationIds).toEqual(expect.arrayContaining(['x', 'y']));
  });

  it('returns a single plain segment when there are no annotations', () => {
    const segs = buildSegmentsFromAnnotations(note, []);
    expect(segs).toEqual([{ key: 'plain-note', text: note, highlighted: false }]);
  });

  it('degrades to plain text when stored span text no longer matches the note (redacted reload)', () => {
    // Note text not persisted; stored offsets/text refer to the original note.
    const segs = buildSegmentsFromAnnotations('[redacted]', annotations);
    expect(segs.some((s) => s.highlighted)).toBe(false);
    expect(segs.map((s) => s.text).join('')).toBe('[redacted]');
  });
});

describe('deriveFindingsFromAnnotations', () => {
  it('dedupes by hpoId, marks the term manual if any span is manual', () => {
    const findings = deriveFindingsFromAnnotations([
      {
        id: 'a1',
        hpoId: 'HP:1',
        label: 'A',
        status: 'affirmed',
        origin: 'auto',
        confidence: 1,
        spans: [{ start: 0, end: 1, text: 'x' }],
      },
      {
        id: 'a2',
        hpoId: 'HP:1',
        label: 'A',
        status: 'affirmed',
        origin: 'manual',
        confidence: null,
        spans: [{ start: 2, end: 3, text: 'y' }],
      },
      {
        id: 'a3',
        hpoId: 'HP:2',
        label: 'B',
        status: 'negated',
        origin: 'manual',
        confidence: null,
        spans: [{ start: 4, end: 5, text: 'z' }],
      },
    ]);
    expect(findings).toHaveLength(2);
    expect(findings.find((f) => f.hpo_id === 'HP:1').origin).toBe('manual');
    expect(findings.find((f) => f.hpo_id === 'HP:2').status).toBe('negated');
  });

  it('includes terms without spans (findings can exceed note highlights)', () => {
    const findings = deriveFindingsFromAnnotations([
      { id: 'a1', hpoId: 'HP:1', label: 'A', status: 'affirmed', origin: 'auto', spans: [] },
    ]);
    expect(findings).toHaveLength(1);
    expect(findings[0].hpo_id).toBe('HP:1');
  });
});

describe('computeSelectionOffsets', () => {
  it('maps a selection range to note-relative offsets across mixed nodes', () => {
    const container = document.createElement('div');
    container.innerHTML = 'She has <mark>microcephaly</mark> today';
    document.body.appendChild(container);

    const markText = container.querySelector('mark').firstChild;
    const range = document.createRange();
    range.setStart(markText, 0);
    range.setEnd(markText, 'microcephaly'.length);

    // 'She has ' is 8 chars, 'microcephaly' is 12 chars
    expect(computeSelectionOffsets(container, range)).toEqual({ start: 8, end: 20 });
    document.body.removeChild(container);
  });

  it('returns null for a collapsed range', () => {
    const container = document.createElement('div');
    container.textContent = 'plain text';
    const range = document.createRange();
    range.setStart(container.firstChild, 3);
    range.setEnd(container.firstChild, 3);
    expect(computeSelectionOffsets(container, range)).toBeNull();
  });
});
