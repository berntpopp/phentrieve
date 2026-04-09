/**
 * Composable for file download functionality.
 * Replaces 3x duplicated document.createElement('a') pattern.
 */
export function useFileDownload() {
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function downloadText(content, filename, mimeType = 'text/plain') {
    const blob = new Blob([content], { type: mimeType });
    downloadBlob(blob, filename);
  }

  function downloadJson(data, filename) {
    const content = JSON.stringify(data, null, 2);
    downloadText(content, filename, 'application/json');
  }

  return { downloadBlob, downloadText, downloadJson };
}
