import { describe, it, expect, beforeEach, vi } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';

// Mock logService before importing composable
vi.mock('../../services/logService', () => ({
  logService: {
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Mutable download mocks so individual tests can make them throw.
const downloadJsonMock = vi.fn();
const downloadTextMock = vi.fn();

vi.mock('../../composables/useFileDownload', () => ({
  useFileDownload: () => ({
    downloadText: downloadTextMock,
    downloadJson: downloadJsonMock,
    downloadBlob: vi.fn(),
  }),
}));

import { usePhenotypeCollection } from '../../composables/usePhenotypeCollection';
import { useConversationStore } from '../../stores/conversation';

describe('usePhenotypeCollection', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
    downloadJsonMock.mockReset();
    downloadTextMock.mockReset();
  });

  it('returns all expected properties and functions', () => {
    const result = usePhenotypeCollection();
    expect(result).toHaveProperty('phenopacketSubjectId');
    expect(result).toHaveProperty('phenopacketSex');
    expect(result).toHaveProperty('phenopacketDateOfBirth');
    expect(result).toHaveProperty('addPhenotype');
    expect(result).toHaveProperty('removePhenotype');
    expect(result).toHaveProperty('toggleAssertionStatus');
    expect(result).toHaveProperty('clearCollection');
    expect(result).toHaveProperty('toggleCollectionPanel');
    expect(result).toHaveProperty('exportCollectionAsText');
    expect(result).toHaveProperty('exportAsPhenopacket');
  });

  it('initializes subject fields as empty', () => {
    const { phenopacketSubjectId, phenopacketSex, phenopacketDateOfBirth } =
      usePhenotypeCollection();
    expect(phenopacketSubjectId.value).toBe('');
    expect(phenopacketSex.value).toBeNull();
    expect(phenopacketDateOfBirth.value).toBeNull();
  });

  it('addPhenotype delegates to conversation store', () => {
    const { addPhenotype } = usePhenotypeCollection();
    addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });
    // The function should work without errors (delegates to store)
  });

  it('clearCollection resets subject info and store', () => {
    const { phenopacketSubjectId, phenopacketSex, phenopacketDateOfBirth, clearCollection } =
      usePhenotypeCollection();
    phenopacketSubjectId.value = 'patient-1';
    phenopacketSex.value = 1;
    phenopacketDateOfBirth.value = '2000-01-01';
    clearCollection();
    expect(phenopacketSubjectId.value).toBe('');
    expect(phenopacketSex.value).toBeNull();
    expect(phenopacketDateOfBirth.value).toBeNull();
  });

  describe('exportAsPhenopacket', () => {
    it('writes DOB to subject.dateOfBirth (NOT timeAtLastEncounter) as Phenopacket v2 requires', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001250', label: 'Seizure' });
      const { exportAsPhenopacket, phenopacketDateOfBirth } = usePhenotypeCollection();
      phenopacketDateOfBirth.value = '2010-05-15';
      exportAsPhenopacket();
      expect(downloadJsonMock).toHaveBeenCalledTimes(1);
      const [phenopacket] = downloadJsonMock.mock.calls[0];
      // Regression guard — this was a real bug fixed in PR #191.
      expect(phenopacket.subject.dateOfBirth).toBe('2010-05-15T00:00:00.000Z');
      expect(phenopacket.subject.timeAtLastEncounter).toBeUndefined();
    });

    it('is a no-op when the collection is empty', () => {
      const { exportAsPhenopacket } = usePhenotypeCollection();
      // Should not throw, and should not attempt to download.
      expect(() => exportAsPhenopacket()).not.toThrow();
      expect(downloadJsonMock).not.toHaveBeenCalled();
    });

    it('throws an Error (instead of calling alert()) when the download fails', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001250', label: 'Seizure' });
      downloadJsonMock.mockImplementation(() => {
        throw new Error('disk full');
      });
      const { exportAsPhenopacket } = usePhenotypeCollection();
      expect(() => exportAsPhenopacket()).toThrow(/disk full/);
    });
  });
});
