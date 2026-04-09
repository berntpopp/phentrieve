import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import { createI18n } from 'vue-i18n'
import en from '../../locales/en.json'

// Mock the API service class — methods are queryHpo() and processText()
// (NOT query() — the real class in services/PhentrieveService.js uses queryHpo)
vi.mock('../../services/PhentrieveService', () => {
  const MockService = {
    queryHpo: vi.fn().mockResolvedValue({ results: [] }),
    processText: vi.fn().mockResolvedValue({ processed_chunks: [], aggregated_hpo_terms: [] }),
    getConfigInfo: vi.fn().mockResolvedValue({
      available_embedding_models: [{ name: 'test-model', id: 'test-model', loaded: true }],
      default_embedding_model: 'test-model',
    }),
  }
  return { default: MockService }
})

// Mock logService to avoid side effects
vi.mock('../../services/logService', () => ({
  logService: {
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}))

const vuetify = createVuetify({ components, directives })
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  silentTranslationWarn: true,
  silentFallbackWarn: true,
  missing: () => '',
})

async function mountQueryInterface() {
  // QueryInterface uses Options API with setup() hook, has no props.
  // It renders a search container with query input.
  const pinia = createPinia()
  setActivePinia(pinia)
  const component = (await import('../../components/QueryInterface.vue')).default
  return mount(component, {
    global: {
      plugins: [pinia, vuetify, i18n],
      stubs: {
        ResultsDisplay: true,
        ConversationSkeleton: true,
        'v-navigation-drawer': true,
      },
      mocks: {
        $route: { query: {} },
        $router: { replace: vi.fn().mockResolvedValue(undefined) },
      },
    },
  })
}

describe('QueryInterface (characterization)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('mounts without errors', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.exists()).toBe(true)
  })

  it('renders the search container', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.find('.search-container').exists()).toBe(true)
  })

  it('initializes with empty query text', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.vm.queryText).toBe('')
  })

  it('initializes with loading false', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.vm.isLoading).toBe(false)
  })

  it('has advanced options hidden by default', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.vm.showAdvancedOptions).toBe(false)
  })

  it('has default threshold values', async () => {
    const wrapper = await mountQueryInterface()
    expect(wrapper.vm.similarityThreshold).toBe(0.5)
    expect(wrapper.vm.numResults).toBe(10)
    expect(wrapper.vm.chunkRetrievalThreshold).toBe(0.7)
    expect(wrapper.vm.aggregatedTermConfidence).toBe(0.75)
  })
})
