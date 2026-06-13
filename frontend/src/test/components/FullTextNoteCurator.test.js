import { setActivePinia, createPinia } from 'pinia';
import { beforeEach, describe, it, expect, vi } from 'vitest';
import { mount, flushPromises } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';

vi.mock('../../services/PhentrieveService', () => ({
  default: {
    queryHpo: vi.fn(async () => ({
      query_assertion_status: 'affirmed',
      results: [{ hpo_id: 'HP:9', label: 'Z', similarity: 0.8 }],
    })),
  },
}));
import PhentrieveService from '../../services/PhentrieveService';
import { useFullTextCurationStore } from '../../stores/fullTextCuration';
import FullTextNoteCurator from '../../components/FullTextNoteCurator.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en }, missing: () => '' });

const PopoverStub = {
  name: 'AnnotationActionPopover',
  props: ['visible', 'target', 'mode', 'canRevert', 'selectedText'],
  emits: ['update:visible', 'close', 'change-term', 'remove-annotation', 'add-to-collection', 'annotate-selection', 'revert'],
  template: '<div class="popover-stub" />',
};
const DialogStub = {
  name: 'HpoTermPickerDialog',
  props: ['modelValue', 'mode', 'spanText', 'candidates', 'loading', 'assertion'],
  emits: ['update:modelValue', 'requery', 'submit', 'cancel'],
  template: '<div class="dialog-stub" />',
};

const noteText = 'She has microcephaly.';
const item = {
  id: 't1',
  response: {
    processed_chunks: [{ chunk_id: 1, text: 'She has microcephaly.' }],
    aggregated_hpo_terms: [
      {
        hpo_id: 'HP:0000252',
        name: 'Microcephaly',
        status: 'present',
        confidence: 1,
        text_attributions: [
          { chunk_id: 1, start_char: 8, end_char: 20, matched_text_in_chunk: 'microcephaly' },
        ],
      },
    ],
  },
};

function mountCurator(props = {}) {
  return mount(FullTextNoteCurator, {
    props: {
      item,
      noteText,
      expanded: true,
      activePhenotypeId: null,
      queryOptions: { model_name: 'm', language: 'en', num_results: 8, similarity_threshold: 0.1 },
      ...props,
    },
    global: {
      plugins: [vuetify, i18n],
      stubs: { AnnotationActionPopover: PopoverStub, HpoTermPickerDialog: DialogStub },
    },
  });
}

const ws = (wrapper) => wrapper.findComponent({ name: 'FullTextWorkspace' });
const popover = (wrapper) => wrapper.findComponent({ name: 'AnnotationActionPopover' });
const dialog = (wrapper) => wrapper.findComponent({ name: 'HpoTermPickerDialog' });

beforeEach(() => {
  setActivePinia(createPinia());
  PhentrieveService.queryHpo.mockClear();
});

describe('FullTextNoteCurator', () => {
  it('seeds the turn and renders the auto annotation as a highlighted span', () => {
    const wrapper = mountCurator();
    expect(wrapper.findAll('[data-testid="annotated-note-span"]')).toHaveLength(1);
    expect(useFullTextCurationStore().annotationsForTurn('t1')).toHaveLength(1);
  });

  it('change-term re-queries and replaces the term on submit', async () => {
    const wrapper = mountCurator();
    const store = useFullTextCurationStore();
    const annId = store.annotationsForTurn('t1')[0].id;

    await ws(wrapper).vm.$emit('span-activate', {
      annotationIds: [annId],
      termIds: ['HP:0000252'],
      rect: { left: 0, width: 10, top: 0 },
      text: 'microcephaly',
    });
    await popover(wrapper).vm.$emit('change-term');
    await flushPromises();

    expect(PhentrieveService.queryHpo).toHaveBeenCalledWith(
      expect.objectContaining({ text: 'microcephaly' })
    );

    await dialog(wrapper).vm.$emit('submit', { term: { hpo_id: 'HP:9', label: 'Z' }, assertion: 'negated' });

    const ann = store.annotationsForTurn('t1')[0];
    expect(ann.hpoId).toBe('HP:9');
    expect(ann.origin).toBe('manual');
    expect(ann.replacedFrom).toMatchObject({ hpoId: 'HP:0000252' });
  });

  it('remove drops the annotation and opens the undo snackbar', async () => {
    const wrapper = mount(FullTextNoteCurator, {
      attachTo: document.body,
      props: {
        item,
        noteText,
        expanded: true,
        activePhenotypeId: null,
        queryOptions: { model_name: 'm', language: 'en', num_results: 8, similarity_threshold: 0.1 },
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: { AnnotationActionPopover: PopoverStub, HpoTermPickerDialog: DialogStub },
      },
    });
    const store = useFullTextCurationStore();
    const annId = store.annotationsForTurn('t1')[0].id;

    await ws(wrapper).vm.$emit('span-activate', {
      annotationIds: [annId],
      termIds: ['HP:0000252'],
      rect: null,
      text: 'microcephaly',
    });
    await popover(wrapper).vm.$emit('remove-annotation');

    expect(store.annotationsForTurn('t1')).toHaveLength(0);
    expect(wrapper.findComponent({ name: 'VSnackbar' }).props('modelValue')).toBe(true);
    expect(store.canUndo('t1')).toBe(true);

    // Undo restores it via the snackbar action (teleported to body)
    await flushPromises();
    const undoBtn = document.querySelector('[data-testid="curation-undo-action"]');
    expect(undoBtn).not.toBeNull();
    undoBtn.click();
    await flushPromises();
    expect(store.annotationsForTurn('t1')).toHaveLength(1);

    wrapper.unmount();
    document.body.innerHTML = '';
  });

  it('annotate-selection adds a manual annotation on submit', async () => {
    const wrapper = mountCurator();
    const store = useFullTextCurationStore();

    await ws(wrapper).vm.$emit('text-select', { text: 'She', start: 0, end: 3, rect: null });
    await popover(wrapper).vm.$emit('annotate-selection');
    await flushPromises();

    await dialog(wrapper).vm.$emit('submit', { term: { hpo_id: 'HP:7', label: 'M' }, assertion: 'affirmed' });

    const manual = store.annotationsForTurn('t1').find((a) => a.hpoId === 'HP:7');
    expect(manual).toBeTruthy();
    expect(manual.origin).toBe('manual');
    expect(manual.spans[0]).toMatchObject({ start: 0, end: 3 });
  });

  it('add-to-collection emits the active term', async () => {
    const wrapper = mountCurator();
    const store = useFullTextCurationStore();
    const annId = store.annotationsForTurn('t1')[0].id;

    await ws(wrapper).vm.$emit('span-activate', {
      annotationIds: [annId],
      termIds: ['HP:0000252'],
      rect: null,
      text: 'microcephaly',
    });
    await popover(wrapper).vm.$emit('add-to-collection');

    expect(wrapper.emitted('add-to-collection')[0][0]).toMatchObject({
      hpo_id: 'HP:0000252',
      label: 'Microcephaly',
      assertion_status: 'affirmed',
    });
  });
});
