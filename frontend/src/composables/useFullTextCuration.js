/**
 * Orchestration composable for full-text annotation curation of a single turn.
 *
 * Owns the data side of curation: seeds the persisted store from the API
 * response, exposes reactive note `segments` and `findings` derived from the
 * annotation model, runs single-term re-queries through PhentrieveService, and
 * applies mutations (replace / remove / revert / addManual / undo). Visual state
 * (popover/dialog) is owned by the consuming component.
 */
import { computed, ref } from 'vue';
import PhentrieveService from '../services/PhentrieveService';
import { useFullTextCurationStore } from '../stores/fullTextCuration';
import {
  seedAnnotationsFromResponse,
  buildSegmentsFromAnnotations,
  deriveFindingsFromAnnotations,
} from './useUserNoteAnnotations';

export function useFullTextCuration(turnId) {
  const store = useFullTextCurationStore();
  const noteText = ref('');

  function setNoteText(text) {
    noteText.value = typeof text === 'string' ? text : '';
  }

  function ensureSeeded(item, text) {
    setNoteText(text);
    if (store.isSeeded(turnId)) return;
    store.seedTurn(
      turnId,
      seedAnnotationsFromResponse({ note: noteText.value, response: item?.response })
    );
  }

  const annotations = computed(() => store.annotationsForTurn(turnId));
  const findings = computed(() => deriveFindingsFromAnnotations(annotations.value));
  const segments = computed(() => buildSegmentsFromAnnotations(noteText.value, annotations.value));
  const canUndo = computed(() => store.canUndo(turnId));

  async function requery(text, options = {}) {
    const payload = {
      text,
      model_name: options.model_name,
      language: options.language ?? null,
      num_results: options.num_results ?? 8,
      similarity_threshold: options.similarity_threshold ?? 0.1,
      query_assertion_language: options.language ?? null,
      detect_query_assertion: true,
      include_details: true,
    };
    const data = await PhentrieveService.queryHpo(payload);
    return {
      assertion: data?.query_assertion_status ?? null,
      results: Array.isArray(data?.results) ? data.results : [],
    };
  }

  function annotationById(id) {
    return annotations.value.find((a) => a.id === id) || null;
  }

  function replace(annotationId, term, assertion) {
    store.replaceTerm(turnId, annotationId, {
      hpoId: term.hpo_id,
      label: term.label,
      status: assertion,
    });
  }

  function remove(annotationId) {
    store.removeAnnotation(turnId, annotationId);
  }

  function revert(annotationId) {
    store.revert(turnId, annotationId);
  }

  function undo() {
    store.undo(turnId);
  }

  function addManual(span, term, assertion) {
    store.addManual(turnId, span, { hpoId: term.hpo_id, label: term.label, status: assertion });
  }

  return {
    ensureSeeded,
    setNoteText,
    noteText,
    annotations,
    findings,
    segments,
    canUndo,
    requery,
    annotationById,
    replace,
    remove,
    revert,
    undo,
    addManual,
  };
}
