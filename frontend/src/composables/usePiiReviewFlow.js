import { ref } from 'vue';
import { redactPiiFindings, scanPii } from '../pii';

export function usePiiReviewFlow({ logService = null } = {}) {
  const piiReviewDialogVisible = ref(false);
  const pendingPiiSubmission = ref(null);

  function openPiiReview(pendingSubmission) {
    pendingPiiSubmission.value = pendingSubmission;
    piiReviewDialogVisible.value = true;
  }

  function cancelPiiReview() {
    piiReviewDialogVisible.value = false;
    pendingPiiSubmission.value = null;
  }

  function scanSubmissionForPii({
    text,
    locale = 'en',
    useTextProcessMode = false,
    isAutoSubmit = false,
  }) {
    const scanResult = scanPii(text, { locale });

    if (!scanResult.hasFindings) {
      return { hasFindings: false, scanResult };
    }

    const pendingSubmission = {
      text,
      useTextProcessMode,
      isAutoSubmit,
      scanResult,
    };

    openPiiReview(pendingSubmission);
    logService?.info?.('PII review required before submission', {
      mode: useTextProcessMode ? 'textProcess' : 'query',
      textLength: String(text ?? '').length,
      piiSummary: scanResult.summary,
    });

    return { hasFindings: true, scanResult, pendingSubmission };
  }

  async function continueWithPiiRedaction({ submitQueryText } = {}) {
    const pending = pendingPiiSubmission.value;

    if (!pending) {
      return null;
    }

    const redaction = redactPiiFindings(pending.text, pending.scanResult.findings, {
      includeReviewFindings: false,
    });

    cancelPiiReview();

    if (typeof submitQueryText === 'function') {
      await submitQueryText({
        currentQuery: redaction.text,
        rawQueryForHistory: pending.text,
        redactedQueryForHistory: redaction.text,
        useTextProcessMode: pending.useTextProcessMode,
        isAutoSubmit: pending.isAutoSubmit,
        piiScanResult: pending.scanResult,
        redactionApplied: redaction.changed,
      });
    }

    return redaction;
  }

  function redactPiiInInput({ setQueryText } = {}) {
    const pending = pendingPiiSubmission.value;

    if (!pending) {
      return null;
    }

    const redaction = redactPiiFindings(pending.text, pending.scanResult.findings, {
      includeReviewFindings: true,
    });

    if (typeof setQueryText === 'function') {
      setQueryText(redaction.text);
    }

    cancelPiiReview();

    return redaction;
  }

  return {
    piiReviewDialogVisible,
    pendingPiiSubmission,
    openPiiReview,
    cancelPiiReview,
    scanSubmissionForPii,
    continueWithPiiRedaction,
    redactPiiInInput,
  };
}
