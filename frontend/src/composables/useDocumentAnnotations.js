function getChunkText(chunk) {
  return typeof chunk?.text === 'string' ? chunk.text : '';
}

function getChunkAnnotations(chunk) {
  return Array.isArray(chunk?.annotations) ? [...chunk.annotations] : [];
}

function normalizeSpanAnnotation(annotation, chunk, index) {
  const textLength = getChunkText(chunk).length;
  const start = Math.max(0, Math.min(annotation?.start_char ?? 0, textLength));
  const end = Math.max(start, Math.min(annotation?.end_char ?? 0, textLength));

  if (end <= start) {
    return null;
  }

  return {
    ...annotation,
    id: annotation?.id || `annotation-${chunk?.chunk_id}-${index}`,
    start_char: start,
    end_char: end,
  };
}

function toSelectedAnnotationSet(selectedAnnotationIds) {
  return selectedAnnotationIds instanceof Set
    ? selectedAnnotationIds
    : new Set(Array.isArray(selectedAnnotationIds) ? selectedAnnotationIds : []);
}

export function getSpanAnnotations(chunk) {
  if ((chunk?.evidence_mode || 'chunk') !== 'span') {
    return [];
  }

  return getChunkAnnotations(chunk)
    .filter((annotation) => annotation?.start_char != null && annotation?.end_char != null)
    .map((annotation, index) => normalizeSpanAnnotation(annotation, chunk, index))
    .filter(Boolean);
}

export function getChunkAnnotationDetails(chunk, spanAnnotations = getSpanAnnotations(chunk)) {
  const detailsById = new Map();

  spanAnnotations.forEach((annotation) => {
    if (!detailsById.has(annotation.id)) {
      detailsById.set(annotation.id, {
        id: annotation.id,
        detailText: annotation.matched_text_in_chunk || '',
      });
    }
  });

  return [...detailsById.values()];
}

export function needsFallbackMarks(chunk, supportsCustomHighlight) {
  return getSpanAnnotations(chunk).length > 0 && !supportsCustomHighlight;
}

export function buildSelectedAnnotationDecorations({
  annotations,
  selectedAnnotationIds,
  segmentStart,
  segmentEnd,
  text,
}) {
  const selectedAnnotationSet = toSelectedAnnotationSet(selectedAnnotationIds);

  return (Array.isArray(annotations) ? annotations : [])
    .filter(
      (annotation) => annotation.start_char < segmentEnd && annotation.end_char > segmentStart
    )
    .map((annotation) => ({
      id: annotation.id,
      selected: selectedAnnotationSet.has(annotation.id),
      detailText: annotation.matched_text_in_chunk || text,
      start_char: annotation.start_char,
      end_char: annotation.end_char,
    }))
    .sort((left, right) => {
      if (left.start_char !== right.start_char) {
        return left.start_char - right.start_char;
      }

      return right.end_char - left.end_char;
    });
}

export function buildMarkedSegments(chunk, selectedAnnotationIds = new Set()) {
  const text = getChunkText(chunk);
  const annotations = getSpanAnnotations(chunk).sort(
    (left, right) => left.start_char - right.start_char
  );
  const boundaries = Array.from(
    new Set([
      0,
      text.length,
      ...annotations.flatMap((annotation) => [annotation.start_char, annotation.end_char]),
    ])
  ).sort((left, right) => left - right);
  const segments = [];
  let previousBoundary = boundaries[0];

  for (const end of boundaries.slice(1)) {
    const start = previousBoundary;
    previousBoundary = end;

    if (end <= start) {
      continue;
    }

    const activeAnnotations = buildSelectedAnnotationDecorations({
      annotations,
      selectedAnnotationIds,
      segmentStart: start,
      segmentEnd: end,
      text: text.slice(start, end),
    });
    const nextSegment = {
      key: `segment-${start}-${end}`,
      text: text.slice(start, end),
      annotations: activeAnnotations,
    };
    const previousSegment = segments[segments.length - 1];
    const previousSignature =
      previousSegment?.annotations?.map((annotation) => annotation.id).join('|') || '';
    const nextSignature = activeAnnotations.map((annotation) => annotation.id).join('|');

    if (previousSegment && previousSignature === nextSignature) {
      previousSegment.text += nextSegment.text;
      previousSegment.key = `${previousSegment.key}-${end}`;
      continue;
    }

    segments.push(nextSegment);
  }

  return segments;
}
