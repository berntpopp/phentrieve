export function summarizeDocumentQuery(query) {
  if (typeof query !== 'string') {
    return '';
  }

  const compact = query.replace(/\s+/g, ' ').trim();
  if (compact.length <= 120) {
    return compact;
  }

  return `${compact.slice(0, 117)}...`;
}

export function formatDocumentSummaryMeta(query) {
  if (typeof query !== 'string') {
    return '';
  }

  const charCount = query.trim().length;
  if (charCount === 0) {
    return '';
  }

  const wordCount = query.trim().split(/\s+/).filter(Boolean).length;

  return `${wordCount} words`;
}

export function resolveChunkOffsetsInNote(noteText, chunks) {
  const offsets = new Map();
  let searchFrom = 0;

  chunks.forEach((chunk) => {
    const chunkText =
      typeof chunk?.text === 'string'
        ? chunk.text
        : typeof chunk?.chunk_text === 'string'
          ? chunk.chunk_text
          : '';

    if (!chunkText || chunk?.chunk_id == null) {
      return;
    }

    let offset = noteText.indexOf(chunkText, searchFrom);
    if (offset === -1) {
      offset = noteText.indexOf(chunkText);
    }
    if (offset === -1) {
      return;
    }

    offsets.set(chunk.chunk_id, offset);
    offsets.set(String(chunk.chunk_id), offset);
    searchFrom = Math.max(offset + 1, searchFrom);
  });

  return offsets;
}

export function resolveMatchedTextRange(noteText, matchedText) {
  if (typeof matchedText !== 'string' || matchedText.trim() === '') {
    return null;
  }

  const normalizedNote = noteText.toLowerCase();
  const normalizedMatch = matchedText.toLowerCase();
  const start = normalizedNote.indexOf(normalizedMatch);
  if (start === -1) {
    return null;
  }

  return {
    start,
    end: start + matchedText.length,
  };
}

export function buildUserNoteSegments({ note, chunks, terms }) {
  const fallbackText = typeof note === 'string' ? note : '';
  const normalizedChunks = Array.isArray(chunks) ? chunks : [];
  const normalizedTerms = Array.isArray(terms) ? terms : [];
  const termLabels = new Map(
    normalizedTerms
      .filter((term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string')
      .map((term) => [term.hpo_id, term.name])
  );

  if (!fallbackText || normalizedChunks.length === 0) {
    return [{ key: 'fallback-note', text: fallbackText, highlighted: false }];
  }

  const chunkOffsets = resolveChunkOffsetsInNote(fallbackText, normalizedChunks);
  // Always build highlights for every term. The hovered/active phenotype is
  // emphasised purely via a CSS class in the template (driven by the
  // activePhenotypeId prop), so the segment list stays stable across hovers:
  // no annotations collapse and the <mark> DOM nodes are never re-created
  // (which previously dropped mouseleave and left a highlight stuck).
  const highlights = normalizedTerms
    .flatMap((term) =>
      (Array.isArray(term.text_attributions) ? term.text_attributions : []).map((attr) => ({
        ...attr,
        termId: term.hpo_id,
      }))
    )
    .map((attr) => {
      const offset =
        chunkOffsets.get(attr?.chunk_id) ??
        chunkOffsets.get(String(attr?.chunk_id)) ??
        chunkOffsets.get(Number(attr?.chunk_id));

      if (offset != null) {
        const start = offset + Math.max(0, attr.start_char ?? 0);
        const end = offset + Math.max(0, attr.end_char ?? 0);
        return {
          start: Math.max(0, Math.min(start, fallbackText.length)),
          end: Math.max(0, Math.min(end, fallbackText.length)),
          termIds: [attr.termId],
        };
      }

      const resolved = resolveMatchedTextRange(fallbackText, attr?.matched_text_in_chunk);
      return resolved
        ? {
            ...resolved,
            termIds: [attr.termId],
          }
        : null;
    })
    .filter(Boolean)
    .map((attr) => ({
      start: attr.start,
      end: attr.end,
      termIds: attr.termIds,
    }))
    .filter((attr) => attr.end > attr.start)
    .sort((left, right) => left.start - right.start);

  if (highlights.length === 0) {
    return [{ key: 'plain-note', text: fallbackText, highlighted: false }];
  }

  const merged = [];
  for (const range of highlights) {
    const previous = merged[merged.length - 1];
    if (previous && range.start <= previous.end) {
      previous.end = Math.max(previous.end, range.end);
      previous.termIds = [...new Set([...previous.termIds, ...range.termIds])];
    } else {
      merged.push({ ...range });
    }
  }

  const segments = [];
  let cursor = 0;
  merged.forEach((range, index) => {
    if (range.start > cursor) {
      segments.push({
        key: `plain-${index}-${cursor}`,
        text: fallbackText.slice(cursor, range.start),
        highlighted: false,
      });
    }
    segments.push({
      key: `mark-${index}-${range.start}`,
      text: fallbackText.slice(range.start, range.end),
      highlighted: true,
      termIds: range.termIds,
      tooltip: range.termIds
        .map((termId) => `${termLabels.get(termId) || termId} (${termId})`)
        .join(', '),
    });
    cursor = range.end;
  });

  if (cursor < fallbackText.length) {
    segments.push({
      key: `plain-tail-${cursor}`,
      text: fallbackText.slice(cursor),
      highlighted: false,
    });
  }

  return segments;
}

/**
 * Map an API aggregated-term assertion status onto the curation model's
 * affirmed/negated/uncertain/unknown vocabulary.
 */
export function normalizeAnnotationStatus(status) {
  if (status === 'absent' || status === 'negated') return 'negated';
  if (status === 'uncertain') return 'uncertain';
  if (status === 'unknown') return 'unknown';
  return 'affirmed';
}

function parseConfidence(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

/**
 * Seed the note-relative annotation model from a full-text processing response.
 * Chunk-relative text attributions are converted to note offsets (falling back
 * to a matched-text search). Returns one annotation per aggregated HPO term that
 * resolves to at least one span. All seeded annotations have origin 'auto'.
 */
export function seedAnnotationsFromResponse({ note, response }) {
  const noteText = typeof note === 'string' ? note : '';
  const chunks = Array.isArray(response?.processed_chunks) ? response.processed_chunks : [];
  const terms = Array.isArray(response?.aggregated_hpo_terms) ? response.aggregated_hpo_terms : [];
  // Findings should not depend on note text; only span resolution does. When the
  // note is empty we still create span-less annotations so terms reach findings.
  if (terms.length === 0) return [];

  const chunkOffsets = resolveChunkOffsetsInNote(noteText, chunks);

  return terms
    .filter((term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string')
    .map((term, index) => {
      const spans = (Array.isArray(term.text_attributions) ? term.text_attributions : [])
        .map((attr) => {
          const base =
            chunkOffsets.get(attr?.chunk_id) ??
            chunkOffsets.get(String(attr?.chunk_id)) ??
            chunkOffsets.get(Number(attr?.chunk_id));

          if (base != null) {
            const start = Math.max(
              0,
              Math.min(base + Math.max(0, attr.start_char ?? 0), noteText.length)
            );
            const end = Math.max(
              0,
              Math.min(base + Math.max(0, attr.end_char ?? 0), noteText.length)
            );
            if (end > start) return { start, end, text: noteText.slice(start, end) };
          }

          const resolved = resolveMatchedTextRange(noteText, attr?.matched_text_in_chunk);
          return resolved
            ? { ...resolved, text: noteText.slice(resolved.start, resolved.end) }
            : null;
        })
        .filter(Boolean)
        .sort((a, b) => a.start - b.start);

      // Keep terms even when no note span resolves: they still belong in the
      // findings list (e.g. inferred terms without offsets). Such annotations
      // simply produce no highlight in the note.
      const confidence = parseConfidence(term.confidence);
      return {
        id: `auto-${term.hpo_id}-${index}`,
        hpoId: term.hpo_id,
        label: term.name,
        status: normalizeAnnotationStatus(term.status),
        spans,
        origin: 'auto',
        confidence,
      };
    });
}

/**
 * Derive ordered note segments from the annotation model. Overlapping spans are
 * merged and their term/annotation ids unioned, so a highlighted run can carry
 * multiple linked phenotypes (preserving the multi-term tooltip behavior).
 */
export function buildSegmentsFromAnnotations(noteText, annotations) {
  const text = typeof noteText === 'string' ? noteText : '';
  const labelById = new Map();
  const ranges = [];

  (Array.isArray(annotations) ? annotations : []).forEach((ann) => {
    if (!ann || typeof ann.hpoId !== 'string') return;
    labelById.set(ann.hpoId, ann.label);
    (Array.isArray(ann.spans) ? ann.spans : []).forEach((span) => {
      if (!span || span.end <= span.start || span.end > text.length) return;
      // Only highlight when the stored span still matches the current note text.
      // The clinical note is not persisted (redacted on reload), so stale offsets
      // must degrade to plain text instead of slicing garbage out of "[redacted]".
      if (typeof span.text === 'string' && text.slice(span.start, span.end) !== span.text) return;
      ranges.push({
        start: span.start,
        end: span.end,
        termIds: [ann.hpoId],
        annotationIds: [ann.id],
      });
    });
  });

  if (ranges.length === 0) return [{ key: 'plain-note', text, highlighted: false }];

  ranges.sort((left, right) => left.start - right.start);

  const merged = [];
  for (const range of ranges) {
    const previous = merged[merged.length - 1];
    if (previous && range.start <= previous.end) {
      previous.end = Math.max(previous.end, range.end);
      previous.termIds = [...new Set([...previous.termIds, ...range.termIds])];
      previous.annotationIds = [...new Set([...previous.annotationIds, ...range.annotationIds])];
    } else {
      merged.push({ ...range });
    }
  }

  const segments = [];
  let cursor = 0;
  merged.forEach((range, index) => {
    if (range.start > cursor) {
      segments.push({
        key: `plain-${index}-${cursor}`,
        text: text.slice(cursor, range.start),
        highlighted: false,
      });
    }
    segments.push({
      key: `mark-${index}-${range.start}`,
      text: text.slice(range.start, range.end),
      highlighted: true,
      termIds: range.termIds,
      annotationIds: range.annotationIds,
      tooltip: range.termIds.map((id) => `${labelById.get(id) || id} (${id})`).join(', '),
    });
    cursor = range.end;
  });

  if (cursor < text.length) {
    segments.push({ key: `plain-tail-${cursor}`, text: text.slice(cursor), highlighted: false });
  }

  return segments;
}

/**
 * Collapse the annotation model into a deduped, term-level findings list for the
 * response receipt. A term is marked 'manual' if any of its annotations is
 * manual. Annotations with no spans are excluded.
 */
export function deriveFindingsFromAnnotations(annotations) {
  const byTerm = new Map();

  (Array.isArray(annotations) ? annotations : []).forEach((ann) => {
    if (!ann || typeof ann.hpoId !== 'string') return;
    const existing = byTerm.get(ann.hpoId);
    if (!existing) {
      byTerm.set(ann.hpoId, {
        hpo_id: ann.hpoId,
        name: ann.label,
        label: ann.label,
        status: ann.status,
        confidence: ann.confidence ?? null,
        origin: ann.origin,
      });
    } else if (ann.origin === 'manual') {
      existing.origin = 'manual';
    }
  });

  return [...byTerm.values()];
}

/**
 * Compute note-relative {start, end} character offsets for a DOM selection range
 * within a container whose text content equals the note text. Walks the
 * container's text nodes in document order and accumulates lengths until it
 * reaches the range's start/end containers. Returns null if the range cannot be
 * resolved or is collapsed.
 */
export function computeSelectionOffsets(container, range) {
  if (!container || !range) return null;

  const doc = container.ownerDocument || (typeof document !== 'undefined' ? document : null);
  if (!doc || typeof doc.createTreeWalker !== 'function') return null;

  const showText = typeof NodeFilter !== 'undefined' ? NodeFilter.SHOW_TEXT : 4;
  const walker = doc.createTreeWalker(container, showText);

  let start = null;
  let end = null;
  let acc = 0;
  let node = walker.nextNode();
  while (node) {
    const length = node.textContent.length;
    if (node === range.startContainer) start = acc + range.startOffset;
    if (node === range.endContainer) end = acc + range.endOffset;
    acc += length;
    node = walker.nextNode();
  }

  if (start == null || end == null) return null;
  if (end < start) {
    const swap = start;
    start = end;
    end = swap;
  }
  return end > start ? { start, end } : null;
}
