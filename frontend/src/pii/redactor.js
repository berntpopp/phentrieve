function mergeFindings(findings, includeReviewFindings) {
  const selected = findings
    .filter((finding) => finding.confidence === 'high' || includeReviewFindings)
    .sort((a, b) => a.start - b.start || a.end - b.end);

  const merged = [];
  for (const finding of selected) {
    const previous = merged[merged.length - 1];
    if (previous && finding.start < previous.end) {
      const previousLength = previous.end - previous.start;
      const findingLength = finding.end - finding.start;
      previous.end = Math.max(previous.end, finding.end);
      if (
        previous.confidence !== 'high' &&
        (finding.confidence === 'high' || findingLength > previousLength)
      ) {
        previous.category = finding.category;
        previous.confidence = finding.confidence;
        previous.redactionToken = finding.redactionToken;
      }
      continue;
    }
    merged.push({ ...finding });
  }
  return merged;
}

function summarize(findings) {
  return findings.reduce(
    (summary, finding) => {
      summary[finding.confidence][finding.category] =
        (summary[finding.confidence][finding.category] ?? 0) + 1;
      return summary;
    },
    { high: {}, review: {} }
  );
}

export function redactPiiFindings(text, findings, { includeReviewFindings = false } = {}) {
  const source = String(text ?? '');
  const selected = mergeFindings(findings, includeReviewFindings);
  if (selected.length === 0) {
    return { text: source, changed: false, summary: { high: {}, review: {} } };
  }

  let cursor = 0;
  let redacted = '';
  for (const finding of selected) {
    redacted += source.slice(cursor, finding.start);
    redacted += finding.redactionToken;
    cursor = finding.end;
  }
  redacted += source.slice(cursor);

  return {
    text: redacted,
    changed: redacted !== source,
    summary: summarize(selected),
  };
}
