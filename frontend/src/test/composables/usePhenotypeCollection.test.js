import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

// Mock logService before importing composable
vi.mock('../../services/logService', () => ({
  logService: {
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}))

// Mock useFileDownload
vi.mock('../../composables/useFileDownload', () => ({
  useFileDownload: () => ({
    downloadText: vi.fn(),
    downloadJson: vi.fn(),
    downloadBlob: vi.fn(),
  }),
}))

import { usePhenotypeCollection } from '../../composables/usePhenotypeCollection'

describe('usePhenotypeCollection', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('returns all expected properties and functions', () => {
    const result = usePhenotypeCollection()
    expect(result).toHaveProperty('phenopacketSubjectId')
    expect(result).toHaveProperty('phenopacketSex')
    expect(result).toHaveProperty('phenopacketDateOfBirth')
    expect(result).toHaveProperty('addPhenotype')
    expect(result).toHaveProperty('removePhenotype')
    expect(result).toHaveProperty('toggleAssertionStatus')
    expect(result).toHaveProperty('clearCollection')
    expect(result).toHaveProperty('toggleCollectionPanel')
    expect(result).toHaveProperty('exportCollectionAsText')
    expect(result).toHaveProperty('exportAsPhenopacket')
  })

  it('initializes subject fields as empty', () => {
    const { phenopacketSubjectId, phenopacketSex, phenopacketDateOfBirth } =
      usePhenotypeCollection()
    expect(phenopacketSubjectId.value).toBe('')
    expect(phenopacketSex.value).toBeNull()
    expect(phenopacketDateOfBirth.value).toBeNull()
  })

  it('addPhenotype delegates to conversation store', () => {
    const { addPhenotype } = usePhenotypeCollection()
    addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' })
    // The function should work without errors (delegates to store)
  })

  it('clearCollection resets subject info and store', () => {
    const { phenopacketSubjectId, phenopacketSex, phenopacketDateOfBirth, clearCollection } =
      usePhenotypeCollection()
    phenopacketSubjectId.value = 'patient-1'
    phenopacketSex.value = 1
    phenopacketDateOfBirth.value = '2000-01-01'
    clearCollection()
    expect(phenopacketSubjectId.value).toBe('')
    expect(phenopacketSex.value).toBeNull()
    expect(phenopacketDateOfBirth.value).toBeNull()
  })
})
