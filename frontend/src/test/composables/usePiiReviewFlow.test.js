import { describe, expect, it, vi } from 'vitest';
import { usePiiReviewFlow } from '../../composables/usePiiReviewFlow';

describe('usePiiReviewFlow', () => {
  it('opens a review dialog and stores pending submission metadata when PII is detected', () => {
    const logService = { info: vi.fn() };
    const flow = usePiiReviewFlow({ logService });

    const result = flow.scanSubmissionForPii({
      text: 'MRN: AB-123456 with seizures',
      locale: 'en',
      useTextProcessMode: false,
      isAutoSubmit: false,
    });

    expect(result.hasFindings).toBe(true);
    expect(flow.piiReviewDialogVisible.value).toBe(true);
    expect(flow.pendingPiiSubmission.value).toMatchObject({
      text: 'MRN: AB-123456 with seizures',
      useTextProcessMode: false,
      isAutoSubmit: false,
    });
    expect(flow.pendingPiiSubmission.value.scanResult.summary.high.medical_record).toBe(1);
  });

  it('returns a clean scan result without opening review when no PII is detected', () => {
    const flow = usePiiReviewFlow();

    const result = flow.scanSubmissionForPii({
      text: 'short syndrome query',
      locale: 'en',
      useTextProcessMode: false,
      isAutoSubmit: false,
    });

    expect(result.hasFindings).toBe(false);
    expect(result.scanResult.hasFindings).toBe(false);
    expect(flow.piiReviewDialogVisible.value).toBe(false);
    expect(flow.pendingPiiSubmission.value).toBeNull();
  });

  it('continues with local redaction and submits redacted text while preserving raw history metadata', async () => {
    const flow = usePiiReviewFlow();
    const submitQueryText = vi.fn().mockResolvedValue(undefined);

    flow.scanSubmissionForPii({
      text: 'MRN: AB-123456 with seizures',
      locale: 'en',
      useTextProcessMode: true,
      isAutoSubmit: true,
    });

    await flow.continueWithPiiRedaction({ submitQueryText });

    expect(submitQueryText).toHaveBeenCalledWith(
      expect.objectContaining({
        currentQuery: expect.stringContaining('[REDACTED_MRN]'),
        rawQueryForHistory: 'MRN: AB-123456 with seizures',
        redactedQueryForHistory: expect.stringContaining('[REDACTED_MRN]'),
        useTextProcessMode: true,
        isAutoSubmit: true,
        redactionApplied: true,
      })
    );
    expect(flow.piiReviewDialogVisible.value).toBe(false);
    expect(flow.pendingPiiSubmission.value).toBeNull();
  });

  it('redacts PII in the input without submitting', () => {
    const flow = usePiiReviewFlow();
    const setQueryText = vi.fn();
    const submitQueryText = vi.fn();

    flow.scanSubmissionForPii({
      text: 'Email jane@example.org and seizures',
      locale: 'en',
      useTextProcessMode: false,
      isAutoSubmit: false,
    });

    const redaction = flow.redactPiiInInput({ setQueryText });

    expect(redaction.text).toContain('[REDACTED_EMAIL]');
    expect(setQueryText).toHaveBeenCalledWith(expect.stringContaining('[REDACTED_EMAIL]'));
    expect(submitQueryText).not.toHaveBeenCalled();
    expect(flow.piiReviewDialogVisible.value).toBe(false);
    expect(flow.pendingPiiSubmission.value).toBeNull();
  });

  it('cancels a pending review without mutating input or submitting', () => {
    const flow = usePiiReviewFlow();

    flow.openPiiReview({
      text: 'MRN: AB-123456 with seizures',
      useTextProcessMode: false,
      isAutoSubmit: false,
      scanResult: { hasFindings: true, findings: [], summary: { high: { mrn: 1 }, review: {} } },
    });

    flow.cancelPiiReview();

    expect(flow.piiReviewDialogVisible.value).toBe(false);
    expect(flow.pendingPiiSubmission.value).toBeNull();
  });
});
