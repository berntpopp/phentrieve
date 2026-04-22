import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';

let originalCSS;
let originalHighlight;
let originalRange;

function installCustomHighlightSupport(rects = []) {
  const set = vi.fn();
  const deleteFn = vi.fn();
  const rectQueue = [...rects];
  originalCSS = globalThis.CSS;
  originalHighlight = globalThis.Highlight;
  originalRange = globalThis.Range;

  Object.defineProperty(globalThis, 'Highlight', {
    configurable: true,
    writable: true,
    value: class Highlight {
      constructor(...ranges) {
        this.ranges = ranges;
      }
    },
  });
  Object.defineProperty(globalThis, 'CSS', {
    configurable: true,
    writable: true,
    value: {
      highlights: {
        set,
        delete: deleteFn,
      },
    },
  });
  Object.defineProperty(globalThis, 'Range', {
    configurable: true,
    writable: true,
    value: class Range {
      constructor() {
        this.rect = rectQueue.shift() || {
          left: 10,
          top: 20,
          right: 40,
          bottom: 28,
          width: 30,
          height: 8,
        };
      }

      setStart(node, offset) {
        this.start = { node, offset };
      }

      setEnd(node, offset) {
        this.end = { node, offset };
      }

      getClientRects() {
        return [this.rect];
      }

      getBoundingClientRect() {
        return this.rect;
      }
    },
  });

  return { set, deleteFn };
}

function restoreHighlightGlobals() {
  if (originalCSS === undefined) {
    delete globalThis.CSS;
  } else {
    Object.defineProperty(globalThis, 'CSS', {
      configurable: true,
      writable: true,
      value: originalCSS,
    });
  }

  if (originalHighlight === undefined) {
    delete globalThis.Highlight;
  } else {
    Object.defineProperty(globalThis, 'Highlight', {
      configurable: true,
      writable: true,
      value: originalHighlight,
    });
  }

  if (originalRange === undefined) {
    delete globalThis.Range;
  } else {
    Object.defineProperty(globalThis, 'Range', {
      configurable: true,
      writable: true,
      value: originalRange,
    });
  }

  originalCSS = undefined;
  originalHighlight = undefined;
  originalRange = undefined;
}

async function loadComponent() {
  vi.resetModules();
  return (await import('../../components/AnnotatedDocumentPane.vue')).default;
}

function popoverStub() {
  return {
    AnnotationActionPopover: {
      props: ['visible', 'target', 'annotationId', 'selectedText'],
      template:
        '<div class="popover-probe" :data-visible="visible" :data-target-left="target?.x ?? \'\'" :data-annotation-id="annotationId ?? \'\'" :data-selected-text="selectedText ?? \'\'" />',
    },
  };
}

async function waitForHighlightSync(wrapper) {
  await wrapper.vm.$nextTick();
  await wrapper.vm.$nextTick();
  await new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
  restoreHighlightGlobals();
  document.head.querySelectorAll('[data-annotated-document-highlight-style]').forEach((node) => {
    node.remove();
  });
});

describe('AnnotatedDocumentPane', () => {
  it('renders chunk-only evidence with gutter tint and span evidence with marks', async () => {
    const component = await loadComponent();
    const wrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 1,
            text: 'Patient had recurrent seizures.',
            status: 'affirmed',
            evidence_mode: 'chunk',
          },
          {
            chunk_id: 2,
            text: 'Developmental delay was present.',
            status: 'affirmed',
            evidence_mode: 'span',
            annotations: [
              { id: 'ann-1', start_char: 0, end_char: 19, matched_text_in_chunk: 'Developmental delay' },
            ],
          },
        ],
      },
      global: {
        stubs: popoverStub(),
      },
    });

    expect(wrapper.find('[data-chunk-evidence-mode="chunk"]').exists()).toBe(true);
    expect(wrapper.find('mark[data-annotation-id="ann-1"]').exists()).toBe(true);
  });

  it('registers styled custom highlights only for span evidence and resyncs selected annotations', async () => {
    const { set, deleteFn } = installCustomHighlightSupport();
    const component = await loadComponent();
    const wrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 1,
            text: 'Chunk-only evidence should not get offset highlights.',
            evidence_mode: 'chunk',
            annotations: [{ id: 'chunk-ann', start_char: 0, end_char: 10 }],
          },
          {
            chunk_id: 2,
            text: 'Developmental delay with seizures.',
            evidence_mode: 'span',
            annotations: [
              { id: 'ann-1', start_char: 0, end_char: 19 },
              { id: 'ann-2', start_char: 25, end_char: 33 },
            ],
          },
        ],
        selectedAnnotationIds: ['ann-2'],
      },
      global: {
        stubs: popoverStub(),
      },
    });

    await waitForHighlightSync(wrapper);

    expect(wrapper.find('mark[data-annotation-id="ann-1"]').exists()).toBe(false);
    expect(set.mock.calls.some(([name]) => name.endsWith('annotation-ann-1'))).toBe(true);
    expect(set.mock.calls.some(([name]) => name.endsWith('annotation-selected-ann-2'))).toBe(true);
    expect(set.mock.calls.some(([name]) => name.includes('chunk-ann'))).toBe(false);
    expect(document.head.innerHTML).toContain('annotation-ann-1');
    expect(document.head.innerHTML).toContain('annotation-selected-ann-2');

    set.mockClear();
    deleteFn.mockClear();

    await wrapper.setProps({ selectedAnnotationIds: ['ann-1'] });
    await waitForHighlightSync(wrapper);

    expect(deleteFn.mock.calls.some(([name]) => name.endsWith('annotation-ann-1'))).toBe(true);
    expect(deleteFn.mock.calls.some(([name]) => name.endsWith('annotation-selected-ann-2'))).toBe(true);
    expect(set.mock.calls.some(([name]) => name.endsWith('annotation-selected-ann-1'))).toBe(true);
    expect(set.mock.calls.some(([name]) => name.endsWith('annotation-ann-2'))).toBe(true);
    expect(document.head.innerHTML).toContain('annotation-selected-ann-1');
  });

  it('keeps custom highlight registrations isolated across pane instances', async () => {
    const { set, deleteFn } = installCustomHighlightSupport([
      { left: 10, top: 20, right: 40, bottom: 28, width: 30, height: 8 },
      { left: 60, top: 20, right: 90, bottom: 28, width: 30, height: 8 },
      { left: 10, top: 20, right: 40, bottom: 28, width: 30, height: 8 },
      { left: 60, top: 20, right: 90, bottom: 28, width: 30, height: 8 },
    ]);
    const component = await loadComponent();
    const firstWrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 11,
            text: 'First pane annotation.',
            evidence_mode: 'span',
            annotations: [{ id: 'shared-ann', start_char: 0, end_char: 5 }],
          },
        ],
      },
      global: {
        stubs: popoverStub(),
      },
    });
    const secondWrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 12,
            text: 'Second pane annotation.',
            evidence_mode: 'span',
            annotations: [{ id: 'shared-ann', start_char: 0, end_char: 6 }],
          },
        ],
      },
      global: {
        stubs: popoverStub(),
      },
    });

    await waitForHighlightSync(firstWrapper);
    await waitForHighlightSync(secondWrapper);

    const highlightNames = set.mock.calls
      .map(([name]) => name)
      .filter((name) => name.includes('shared-ann'));
    const uniqueHighlightNames = [...new Set(highlightNames)];

    expect(uniqueHighlightNames).toHaveLength(2);
    expect(document.head.querySelectorAll('[data-annotated-document-highlight-style]')).toHaveLength(2);

    firstWrapper.unmount();

    const deletedNames = deleteFn.mock.calls.map(([name]) => name);

    expect(uniqueHighlightNames.filter((name) => deletedNames.includes(name))).toHaveLength(1);
    expect(document.head.querySelectorAll('[data-annotated-document-highlight-style]')).toHaveLength(1);

    secondWrapper.unmount();
  });

  it('keeps repeated linked ranges highlighted together for the same annotation id', async () => {
    const { set, deleteFn } = installCustomHighlightSupport([
      { left: 10, top: 20, right: 40, bottom: 28, width: 30, height: 8 },
      { left: 60, top: 20, right: 90, bottom: 28, width: 30, height: 8 },
      { left: 10, top: 20, right: 40, bottom: 28, width: 30, height: 8 },
      { left: 60, top: 20, right: 90, bottom: 28, width: 30, height: 8 },
    ]);
    const component = await loadComponent();
    const wrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 2,
            text: 'Seizures and more seizures were reported.',
            evidence_mode: 'span',
            annotations: [
              { id: 'shared-ann', start_char: 0, end_char: 8, matched_text_in_chunk: 'Seizures' },
              { id: 'shared-ann', start_char: 18, end_char: 26, matched_text_in_chunk: 'seizures' },
            ],
          },
        ],
        selectedAnnotationIds: ['shared-ann'],
      },
      global: {
        stubs: popoverStub(),
      },
    });

    await waitForHighlightSync(wrapper);

    const selectedSharedCall = set.mock.calls.find(([name]) =>
      name.endsWith('annotation-selected-shared-ann')
    );

    expect(selectedSharedCall).toBeDefined();
    expect(selectedSharedCall[1].ranges).toHaveLength(2);

    set.mockClear();
    deleteFn.mockClear();

    await wrapper.setProps({ selectedAnnotationIds: [] });
    await waitForHighlightSync(wrapper);

    const unselectedSharedCall = set.mock.calls.find(([name]) =>
      name.endsWith('annotation-shared-ann')
    );

    expect(deleteFn.mock.calls.some(([name]) => name.endsWith('annotation-selected-shared-ann'))).toBe(
      true
    );
    expect(unselectedSharedCall).toBeDefined();
    expect(unselectedSharedCall[1].ranges).toHaveLength(2);
  });

  it('renders overlapping fallback spans without clipping nested evidence', async () => {
    const component = await loadComponent();
    const wrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 6,
            text: 'ABCDEFGH',
            evidence_mode: 'span',
            annotations: [
              { id: 'ann-1', start_char: 0, end_char: 4, matched_text_in_chunk: 'ABCD' },
              { id: 'ann-2', start_char: 2, end_char: 6, matched_text_in_chunk: 'CDEF' },
            ],
          },
        ],
      },
      global: {
        stubs: popoverStub(),
      },
    });

    expect(wrapper.text()).toContain('ABCDEFGH');
    expect(wrapper.find('mark[data-annotation-id="ann-1"]').exists()).toBe(true);
    expect(wrapper.find('mark[data-annotation-id="ann-2"]').exists()).toBe(true);
    expect(wrapper.find('mark[data-annotation-id="ann-1"] mark[data-annotation-id="ann-2"]').exists()).toBe(true);
    expect(
      wrapper.findAll('mark[data-annotation-id="ann-1"]').some((node) => node.text().includes('AB'))
    ).toBe(true);
    expect(
      wrapper.findAll('mark[data-annotation-id="ann-2"]').some((node) => node.text().includes('EF'))
    ).toBe(true);
  });

  it('keeps text selection working in the fallback mark path while still opening annotation actions on collapsed click', async () => {
    const component = await loadComponent();
    const wrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 4,
            text: 'Developmental delay was present.',
            evidence_mode: 'span',
            annotations: [
              { id: 'ann-1', start_char: 0, end_char: 19, matched_text_in_chunk: 'Developmental delay' },
            ],
          },
        ],
      },
      global: {
        stubs: popoverStub(),
      },
    });

    const textNode = wrapper.find('.chunk-text').element.firstChild;
    const removeAllRanges = vi.fn();
    vi.spyOn(window, 'getSelection').mockReturnValueOnce({
      isCollapsed: false,
      rangeCount: 1,
      anchorNode: textNode,
      focusNode: textNode,
      toString: () => 'Developmental delay',
      getRangeAt: () => ({
        getBoundingClientRect: () => ({
          left: 12,
          top: 24,
          right: 56,
          bottom: 40,
          width: 44,
          height: 16,
        }),
      }),
      removeAllRanges,
    });

    await wrapper.find('.chunk-text').trigger('mouseup');

    expect(wrapper.find('.popover-probe').attributes('data-selected-text')).toBe(
      'Developmental delay'
    );
    expect(wrapper.find('.popover-probe').attributes('data-annotation-id')).toBe('');

    vi.spyOn(window, 'getSelection').mockReturnValueOnce({
      isCollapsed: false,
      rangeCount: 1,
      toString: () => 'Developmental delay',
      removeAllRanges,
    });

    await wrapper.find('mark[data-annotation-id="ann-1"]').trigger('click');

    expect(wrapper.find('.popover-probe').attributes('data-selected-text')).toBe(
      'Developmental delay'
    );
    expect(wrapper.find('.popover-probe').attributes('data-annotation-id')).toBe('');
    expect(removeAllRanges).not.toHaveBeenCalled();

    vi.spyOn(window, 'getSelection').mockReturnValueOnce({
      isCollapsed: true,
      rangeCount: 0,
      toString: () => '',
      removeAllRanges,
    });

    await wrapper.find('mark[data-annotation-id="ann-1"]').trigger('click');

    expect(wrapper.find('.popover-probe').attributes('data-annotation-id')).toBe('ann-1');
    expect(wrapper.find('.popover-probe').attributes('data-selected-text')).toBe(
      'Developmental delay'
    );
    expect(removeAllRanges).toHaveBeenCalled();
  });

  it('keeps text selection working in the custom path while still opening annotation actions on click', async () => {
    installCustomHighlightSupport([
      { left: 30, top: 40, right: 70, bottom: 52, width: 40, height: 12 },
      { left: 30, top: 40, right: 70, bottom: 52, width: 40, height: 12 },
      { left: 30, top: 40, right: 70, bottom: 52, width: 40, height: 12 },
      { left: 30, top: 40, right: 70, bottom: 52, width: 40, height: 12 },
    ]);
    const component = await loadComponent();
    const wrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 5,
            text: 'Developmental delay was present.',
            evidence_mode: 'span',
            annotations: [
              { id: 'ann-1', start_char: 0, end_char: 19, matched_text_in_chunk: 'Developmental delay' },
            ],
          },
        ],
      },
      global: {
        stubs: popoverStub(),
      },
    });

    await waitForHighlightSync(wrapper);

    const textNode = wrapper.find('.chunk-text').element.firstChild;
    vi.spyOn(window, 'getSelection').mockReturnValueOnce({
      isCollapsed: false,
      rangeCount: 1,
      anchorNode: textNode,
      focusNode: textNode,
      toString: () => 'Developmental delay',
      getRangeAt: () => ({
        getBoundingClientRect: () => ({
          left: 30,
          top: 40,
          right: 70,
          bottom: 52,
          width: 40,
          height: 12,
        }),
      }),
    });

    await wrapper.find('.chunk-text').trigger('mouseup');

    expect(wrapper.find('.popover-probe').attributes('data-visible')).toBe('true');
    expect(wrapper.find('.popover-probe').attributes('data-selected-text')).toBe(
      'Developmental delay'
    );
    expect(wrapper.find('.popover-probe').attributes('data-target-left')).toBe('50');

    vi.spyOn(window, 'getSelection').mockReturnValueOnce({
      isCollapsed: true,
      rangeCount: 0,
      toString: () => '',
    });

    await wrapper.find('.chunk-text').trigger('click', { clientX: 35, clientY: 45 });

    expect(wrapper.find('.popover-probe').attributes('data-visible')).toBe('true');
    expect(wrapper.find('.popover-probe').attributes('data-annotation-id')).toBe('ann-1');
    expect(wrapper.find('.popover-probe').attributes('data-selected-text')).toBe(
      'Developmental delay'
    );
    expect(wrapper.find('.popover-probe').attributes('data-target-left')).toBe('50');
  });

  it('resolves overlapping custom highlight clicks to the innermost annotation', async () => {
    installCustomHighlightSupport([
      { left: 30, top: 40, right: 90, bottom: 52, width: 60, height: 12 },
      { left: 45, top: 40, right: 75, bottom: 52, width: 30, height: 12 },
      { left: 30, top: 40, right: 90, bottom: 52, width: 60, height: 12 },
      { left: 45, top: 40, right: 75, bottom: 52, width: 30, height: 12 },
    ]);
    const component = await loadComponent();
    const wrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 13,
            text: 'ABCDEFGH',
            evidence_mode: 'span',
            annotations: [
              { id: 'outer-ann', start_char: 0, end_char: 6, matched_text_in_chunk: 'ABCDEF' },
              { id: 'inner-ann', start_char: 2, end_char: 5, matched_text_in_chunk: 'CDE' },
            ],
          },
        ],
      },
      global: {
        stubs: popoverStub(),
      },
    });

    await waitForHighlightSync(wrapper);

    vi.spyOn(window, 'getSelection').mockReturnValueOnce({
      isCollapsed: true,
      rangeCount: 0,
      toString: () => '',
    });

    await wrapper.find('.chunk-text').trigger('click', { clientX: 50, clientY: 45 });

    expect(wrapper.find('.popover-probe').attributes('data-annotation-id')).toBe('inner-ann');
    expect(wrapper.find('.popover-probe').attributes('data-selected-text')).toBe('CDE');
    expect(wrapper.find('.popover-probe').attributes('data-target-left')).toBe('60');
  });
});
