/**
 * Persisted full-text curation store, keyed by conversation turn id.
 *
 * Holds a note-relative annotation model per turn plus a per-turn undo stack and
 * a frozen seed snapshot (used for revert-to-auto). The store is the single
 * source of truth shared by the note curator and the findings receipt, so both
 * stay in sync. Curation survives reload via pinia-plugin-persistedstate.
 */
import { defineStore } from 'pinia';
import { ref } from 'vue';

const MAX_UNDO = 50;

function clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

export const useFullTextCurationStore = defineStore(
  'fullTextCuration',
  () => {
    // turnId -> { seeded, seed: Annotation[], annotations: Annotation[], undoStack: Annotation[][] }
    const turns = ref({});
    // Monotonic counter for manual annotation ids; persisted so ids stay unique across reloads.
    const seq = ref(0);

    function ensure(turnId) {
      if (!turns.value[turnId]) {
        turns.value[turnId] = { seeded: false, seed: [], annotations: [], undoStack: [] };
      }
      return turns.value[turnId];
    }

    function pushUndo(turn) {
      turn.undoStack.push(clone(turn.annotations));
      if (turn.undoStack.length > MAX_UNDO) turn.undoStack.shift();
    }

    function seedTurn(turnId, annotations) {
      const turn = ensure(turnId);
      if (turn.seeded) return;
      turn.seed = clone(annotations) || [];
      turn.annotations = clone(annotations) || [];
      turn.seeded = true;
    }

    function isSeeded(turnId) {
      return Boolean(turns.value[turnId]?.seeded);
    }

    function annotationsForTurn(turnId) {
      return turns.value[turnId]?.annotations ?? [];
    }

    function canUndo(turnId) {
      return (turns.value[turnId]?.undoStack?.length ?? 0) > 0;
    }

    function removeAnnotation(turnId, id) {
      const turn = ensure(turnId);
      pushUndo(turn);
      turn.annotations = turn.annotations.filter((a) => a.id !== id);
    }

    function replaceTerm(turnId, id, term) {
      const turn = ensure(turnId);
      pushUndo(turn);
      turn.annotations = turn.annotations.map((a) => {
        if (a.id !== id) return a;
        const replacedFrom = a.replacedFrom || { hpoId: a.hpoId, label: a.label };
        return {
          ...a,
          hpoId: term.hpoId,
          label: term.label,
          status: term.status || a.status,
          origin: 'manual',
          replacedFrom,
        };
      });
    }

    function addManual(turnId, span, term) {
      const turn = ensure(turnId);
      pushUndo(turn);
      seq.value += 1;
      turn.annotations = [
        ...turn.annotations,
        {
          id: `manual-${term.hpoId}-${seq.value}`,
          hpoId: term.hpoId,
          label: term.label,
          status: term.status || 'affirmed',
          spans: [clone(span)],
          origin: 'manual',
          confidence: null,
        },
      ];
    }

    function revert(turnId, id) {
      const turn = ensure(turnId);
      const original = turn.seed.find((a) => a.id === id);
      if (!original) return;
      pushUndo(turn);
      turn.annotations = turn.annotations.map((a) => (a.id === id ? clone(original) : a));
    }

    function undo(turnId) {
      const turn = turns.value[turnId];
      if (!turn || turn.undoStack.length === 0) return;
      turn.annotations = turn.undoStack.pop();
    }

    function dropTurn(turnId) {
      delete turns.value[turnId];
    }

    return {
      turns,
      seq,
      seedTurn,
      isSeeded,
      annotationsForTurn,
      canUndo,
      removeAnnotation,
      replaceTerm,
      addManual,
      revert,
      undo,
      dropTurn,
    };
  },
  {
    persist: {
      key: 'phentrieve-fulltext-curation',
      storage: localStorage,
      pick: ['turns', 'seq'],
    },
  }
);
