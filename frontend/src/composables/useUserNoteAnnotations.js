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
