import { afterEach, describe, expect, it, vi } from 'vitest';
import { defineComponent, h, ref } from 'vue';
import { enableAutoUnmount, mount } from '@vue/test-utils';
import { useCustomHighlightOverlay } from '../../composables/useCustomHighlightOverlay';

enableAutoUnmount(afterEach);

let originalCSS;
let originalHighlight;
let originalRange;
let originalResizeObserver;
let originalFonts;

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

function installResizeObserverMock() {
  const instances = [];

  originalResizeObserver = globalThis.ResizeObserver;

  Object.defineProperty(globalThis, 'ResizeObserver', {
    configurable: true,
    writable: true,
    value: class ResizeObserver {
      constructor(callback) {
        this.callback = callback;
        this.observe = vi.fn();
        this.unobserve = vi.fn();
        this.disconnect = vi.fn();
        instances.push(this);
      }
    },
  });

  return instances;
}

function installDocumentFontsMock() {
  const addEventListener = vi.fn();
  const removeEventListener = vi.fn();

  originalFonts = document.fonts;

  Object.defineProperty(document, 'fonts', {
    configurable: true,
    writable: true,
    value: {
      addEventListener,
      removeEventListener,
    },
  });

  return { addEventListener, removeEventListener };
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

  if (originalResizeObserver === undefined) {
    delete globalThis.ResizeObserver;
  } else {
    Object.defineProperty(globalThis, 'ResizeObserver', {
      configurable: true,
      writable: true,
      value: originalResizeObserver,
    });
  }

  if (originalFonts === undefined) {
    delete document.fonts;
  } else {
    Object.defineProperty(document, 'fonts', {
      configurable: true,
      writable: true,
      value: originalFonts,
    });
  }

  originalCSS = undefined;
  originalHighlight = undefined;
  originalRange = undefined;
  originalResizeObserver = undefined;
  originalFonts = undefined;
}

function renderChunkText(chunk) {
  if (Array.isArray(chunk.renderParts)) {
    return chunk.renderParts.map((part, index) => h('span', { key: index }, part));
  }

  return chunk.text || '';
}

function mountOverlayHarness({
  chunks: initialChunks = [],
  selectedAnnotationIds: initialSelectedAnnotationIds = [],
  onLayoutRefresh = vi.fn(),
} = {}) {
  const chunks = ref(initialChunks);
  const selectedAnnotationIds = ref(initialSelectedAnnotationIds);
  const rootElement = ref(null);
  let overlay;

  const Harness = defineComponent({
    name: 'OverlayHarness',
    setup() {
      overlay = useCustomHighlightOverlay({
        chunks,
        selectedAnnotationIds,
        rootElement,
        onLayoutRefresh,
      });

      return () =>
        h(
          'section',
          { ref: rootElement },
          chunks.value.map((chunk) =>
            h(
              'p',
              {
                key: chunk.chunk_id,
                'data-chunk-text-id': chunk.chunk_id,
              },
              renderChunkText(chunk)
            )
          )
        );
    },
  });

  const wrapper = mount(Harness);

  return {
    wrapper,
    chunks,
    selectedAnnotationIds,
    rootElement,
    overlay,
    onLayoutRefresh,
  };
}

async function waitForOverlaySync(wrapper) {
  await wrapper.vm.$nextTick();
  await wrapper.vm.$nextTick();
  await new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  restoreHighlightGlobals();
  document.head.querySelectorAll('[data-annotated-document-highlight-style]').forEach((node) => {
    node.remove();
  });
});

describe('useCustomHighlightOverlay', () => {
  it('returns no-op highlight state when CSS highlights are unavailable', () => {
    const { overlay } = mountOverlayHarness();

    expect(overlay.supportsCustomHighlight).toBe(false);
    expect(overlay.customHighlightHitboxes.value).toEqual([]);
    expect(
      overlay.getAnchorTarget({
        type: 'custom-hitbox',
        key: 'missing',
        annotationId: 'missing',
      })
    ).toBeNull();
  });

  it('registers styled custom highlights, refreshes hitbox anchors, and cleans up lifecycle state', async () => {
    const { set, deleteFn } = installCustomHighlightSupport([
      { left: 20, top: 30, right: 40, bottom: 40, width: 20, height: 10 },
      { left: 20, top: 30, right: 40, bottom: 40, width: 20, height: 10 },
      { left: 60, top: 30, right: 80, bottom: 40, width: 20, height: 10 },
      { left: 60, top: 30, right: 80, bottom: 40, width: 20, height: 10 },
    ]);
    const resizeObservers = installResizeObserverMock();
    const fonts = installDocumentFontsMock();
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
      callback();
      return 1;
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});

    const onLayoutRefresh = vi.fn();
    const { wrapper, overlay, selectedAnnotationIds } = mountOverlayHarness({
      chunks: [
        {
          chunk_id: 21,
          text: 'Developmental delay was present.',
          evidence_mode: 'span',
          annotations: [
            {
              id: 'ann-1',
              start_char: 0,
              end_char: 19,
              matched_text_in_chunk: 'Developmental delay',
            },
          ],
        },
      ],
      selectedAnnotationIds: ['ann-1'],
      onLayoutRefresh,
    });

    await waitForOverlaySync(wrapper);

    expect(overlay.supportsCustomHighlight).toBe(true);
    expect(set.mock.calls.some(([name]) => name.endsWith('annotation-selected-ann-1'))).toBe(true);
    expect(document.head.innerHTML).toContain('annotation-ann-1');
    expect(document.head.innerHTML).toContain('annotation-selected-ann-1');
    expect(resizeObservers).toHaveLength(1);
    expect(resizeObservers[0].observe).toHaveBeenCalledWith(wrapper.element);
    expect(fonts.addEventListener).toHaveBeenCalledWith('loadingdone', expect.any(Function));

    const initialHitbox = overlay.customHighlightHitboxes.value[0];

    expect(initialHitbox.target.x).toBe(30);

    set.mockClear();
    deleteFn.mockClear();
    onLayoutRefresh.mockClear();

    selectedAnnotationIds.value = [];
    await waitForOverlaySync(wrapper);

    expect(deleteFn.mock.calls.some(([name]) => name.endsWith('annotation-selected-ann-1'))).toBe(
      true
    );
    expect(set.mock.calls.some(([name]) => name.endsWith('annotation-ann-1'))).toBe(true);
    expect(onLayoutRefresh).toHaveBeenCalledTimes(1);

    window.dispatchEvent(new Event('scroll'));
    await waitForOverlaySync(wrapper);

    expect(onLayoutRefresh).toHaveBeenCalledTimes(2);
    expect(
      overlay.getAnchorTarget(
        {
          type: 'custom-hitbox',
          key: initialHitbox.key,
          annotationId: initialHitbox.annotationId,
          target: initialHitbox.target,
        },
        initialHitbox.target
      )
    ).toEqual({
      x: 70,
      y: 30,
    });

    wrapper.unmount();

    expect(deleteFn.mock.calls.some(([name]) => name.endsWith('annotation-ann-1'))).toBe(true);
    expect(resizeObservers[0].disconnect).toHaveBeenCalled();
    expect(fonts.removeEventListener).toHaveBeenCalledWith('loadingdone', expect.any(Function));
    expect(
      document.head.querySelectorAll('[data-annotated-document-highlight-style]')
    ).toHaveLength(0);
  });

  it('maps ranges across text nodes and resolves overlapping hitboxes to the most specific match', async () => {
    const { set } = installCustomHighlightSupport([
      { left: 30, top: 40, right: 90, bottom: 52, width: 60, height: 12 },
      { left: 45, top: 40, right: 75, bottom: 52, width: 30, height: 12 },
    ]);
    const { wrapper, overlay } = mountOverlayHarness({
      chunks: [
        {
          chunk_id: 13,
          text: 'ABCDEFGH',
          renderParts: ['AB', 'CD', 'EFGH'],
          evidence_mode: 'span',
          annotations: [
            { id: 'outer-ann', start_char: 0, end_char: 6, matched_text_in_chunk: 'ABCDEF' },
            { id: 'inner-ann', start_char: 2, end_char: 5, matched_text_in_chunk: 'CDE' },
          ],
        },
      ],
    });

    await waitForOverlaySync(wrapper);

    const innerCall = set.mock.calls.find(([name]) => name.endsWith('annotation-inner-ann'));

    expect(innerCall).toBeDefined();
    expect(innerCall[1].ranges[0].start.node.textContent).toBe('AB');
    expect(innerCall[1].ranges[0].start.offset).toBe(2);
    expect(innerCall[1].ranges[0].end.node.textContent).toBe('EFGH');
    expect(innerCall[1].ranges[0].end.offset).toBe(1);

    expect(
      overlay.findHitboxForEvent(13, {
        clientX: 50,
        clientY: 45,
      })
    ).toEqual(
      expect.objectContaining({
        annotationId: 'inner-ann',
        selectedText: 'CDE',
        target: { x: 60, y: 40 },
      })
    );
  });
});
