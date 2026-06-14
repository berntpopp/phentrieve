import { setActivePinia, createPinia } from 'pinia';
import { beforeEach, describe, it, expect } from 'vitest';
import { useFullTextCurationStore } from '../../stores/fullTextCuration';

const seedAnn = () => [
  {
    id: 'auto-HP:1-0',
    hpoId: 'HP:1',
    label: 'A',
    status: 'affirmed',
    origin: 'auto',
    confidence: 1,
    spans: [{ start: 0, end: 3, text: 'aaa' }],
  },
];

describe('fullTextCuration store', () => {
  beforeEach(() => setActivePinia(createPinia()));

  it('seeds a turn once (idempotent) and reads annotations', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.seedTurn('t1', []); // ignored, already seeded
    expect(s.isSeeded('t1')).toBe(true);
    expect(s.annotationsForTurn('t1')).toHaveLength(1);
  });

  it('removeAnnotation drops it and supports undo', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.removeAnnotation('t1', 'auto-HP:1-0');
    expect(s.annotationsForTurn('t1')).toHaveLength(0);
    expect(s.canUndo('t1')).toBe(true);
    s.undo('t1');
    expect(s.annotationsForTurn('t1')).toHaveLength(1);
  });

  it('replaceTerm records replacedFrom and flips origin to manual', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.replaceTerm('t1', 'auto-HP:1-0', { hpoId: 'HP:9', label: 'Z', status: 'negated' });
    const ann = s.annotationsForTurn('t1')[0];
    expect(ann.hpoId).toBe('HP:9');
    expect(ann.label).toBe('Z');
    expect(ann.status).toBe('negated');
    expect(ann.origin).toBe('manual');
    expect(ann.replacedFrom).toMatchObject({ hpoId: 'HP:1', label: 'A' });
  });

  it('addManual appends a manual annotation with a unique id', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', []);
    s.addManual(
      't1',
      { start: 5, end: 9, text: 'cccc' },
      { hpoId: 'HP:7', label: 'M', status: 'affirmed' }
    );
    s.addManual(
      't1',
      { start: 10, end: 14, text: 'dddd' },
      { hpoId: 'HP:7', label: 'M', status: 'affirmed' }
    );
    const anns = s.annotationsForTurn('t1');
    expect(anns).toHaveLength(2);
    expect(anns[0].origin).toBe('manual');
    expect(anns[0].spans[0]).toMatchObject({ start: 5, end: 9 });
    expect(anns[0].id).not.toBe(anns[1].id);
  });

  it('revert restores an auto annotation to its seeded value', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.replaceTerm('t1', 'auto-HP:1-0', { hpoId: 'HP:9', label: 'Z', status: 'affirmed' });
    s.revert('t1', 'auto-HP:1-0');
    expect(s.annotationsForTurn('t1')[0].hpoId).toBe('HP:1');
    expect(s.annotationsForTurn('t1')[0].origin).toBe('auto');
  });

  it('dropTurn clears state for evicted history items', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.dropTurn('t1');
    expect(s.isSeeded('t1')).toBe(false);
  });
});
