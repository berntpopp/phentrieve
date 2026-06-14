import { setActivePinia, createPinia } from 'pinia';
import { beforeEach, describe, it, expect, vi } from 'vitest';

vi.mock('../../services/PhentrieveService', () => ({
  default: {
    queryHpo: vi.fn(async () => ({
      query_assertion_status: 'affirmed',
      results: [{ hpo_id: 'HP:9', label: 'Z', similarity: 0.8 }],
    })),
  },
}));
import PhentrieveService from '../../services/PhentrieveService';
import { useFullTextCuration } from '../../composables/useFullTextCuration';

const item = {
  id: 't1',
  response: {
    processed_chunks: [{ chunk_id: 1, text: 'She has microcephaly.' }],
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
    ],
  },
};

describe('useFullTextCuration', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    PhentrieveService.queryHpo.mockClear();
  });

  it('seeds and exposes segments + findings', () => {
    const c = useFullTextCuration('t1');
    c.ensureSeeded(item, 'She has microcephaly.');
    expect(c.findings.value).toHaveLength(1);
    expect(c.findings.value[0].hpo_id).toBe('HP:0000252');
    expect(c.segments.value.some((s) => s.highlighted)).toBe(true);
  });

  it('does not seed while the response is still loading, then seeds when it arrives', () => {
    const c = useFullTextCuration('t1');
    // Component mounted during the loading state: response not yet present.
    c.ensureSeeded({ id: 't1', response: null }, 'She has microcephaly.');
    expect(c.findings.value).toHaveLength(0);
    // Response arrives: seeding now populates findings (the bug was locking in 0).
    c.ensureSeeded(item, 'She has microcephaly.');
    expect(c.findings.value).toHaveLength(1);
  });

  it('requery calls the service with curation defaults and returns candidates + assertion', async () => {
    const c = useFullTextCuration('t1');
    const res = await c.requery('microcephaly', {
      model_name: 'm',
      language: 'en',
      num_results: 8,
      similarity_threshold: 0.1,
    });
    expect(PhentrieveService.queryHpo).toHaveBeenCalledWith(
      expect.objectContaining({ text: 'microcephaly', include_details: true, num_results: 8 })
    );
    expect(res.assertion).toBe('affirmed');
    expect(res.results[0].hpo_id).toBe('HP:9');
  });

  it('replace + remove + add mutate findings', () => {
    const c = useFullTextCuration('t1');
    c.ensureSeeded(item, 'She has microcephaly.');
    const annId = c.annotations.value[0].id;

    c.replace(annId, { hpo_id: 'HP:9', label: 'Z' }, 'negated');
    expect(c.findings.value[0].hpo_id).toBe('HP:9');
    expect(c.findings.value[0].origin).toBe('manual');

    c.remove(annId);
    expect(c.findings.value).toHaveLength(0);

    c.addManual({ start: 0, end: 3, text: 'She' }, { hpo_id: 'HP:7', label: 'M' }, 'affirmed');
    expect(c.findings.value[0].hpo_id).toBe('HP:7');
    expect(c.canUndo.value).toBe(true);
  });
});
