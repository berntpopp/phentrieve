export function normalizeSelectedTerm(term) {
  if (
    term == null ||
    typeof term.hpo_id !== 'string' ||
    typeof term.name !== 'string' ||
    (term.confidence != null && typeof term.confidence !== 'number')
  ) {
    return null;
  }

  return {
    hpoId: term.hpo_id,
    name: term.name,
    confidence: term.confidence ?? null,
  };
}
