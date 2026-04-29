import {
  computed,
  getCurrentInstance,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  unref,
  watch,
} from 'vue';
import { getSpanAnnotations } from './useDocumentAnnotations';

function supportsCustomHighlightApi() {
  return (
    typeof globalThis.CSS !== 'undefined' &&
    typeof globalThis.Highlight !== 'undefined' &&
    typeof globalThis.CSS.highlights !== 'undefined'
  );
}

function rectToTarget(rect) {
  if (!rect) {
    return null;
  }

  return {
    x: rect.left + rect.width / 2,
    y: rect.top,
  };
}

function collectTextNodes(element) {
  if (!element || typeof document.createTreeWalker !== 'function') {
    return [];
  }

  const walker = document.createTreeWalker(element, globalThis.NodeFilter?.SHOW_TEXT ?? 4);
  const textNodes = [];
  let currentNode = walker.nextNode();

  while (currentNode) {
    if ((currentNode.textContent?.length || 0) > 0) {
      textNodes.push(currentNode);
    }
    currentNode = walker.nextNode();
  }

  return textNodes;
}

function resolveTextPosition(textNodes, offset) {
  if (textNodes.length === 0) {
    return null;
  }

  if (offset <= 0) {
    return { node: textNodes[0], offset: 0 };
  }

  let consumedLength = 0;

  for (const textNode of textNodes) {
    const nodeLength = textNode.textContent?.length || 0;
    const nextLength = consumedLength + nodeLength;

    if (offset <= nextLength) {
      return {
        node: textNode,
        offset: offset - consumedLength,
      };
    }

    consumedLength = nextLength;
  }

  const lastNode = textNodes[textNodes.length - 1];
  return {
    node: lastNode,
    offset: lastNode.textContent?.length || 0,
  };
}

function hitboxContainsPoint(hitbox, event) {
  if (!hitbox?.rect) {
    return false;
  }

  return (
    event.clientX >= hitbox.rect.left &&
    event.clientX <= hitbox.rect.right &&
    event.clientY >= hitbox.rect.top &&
    event.clientY <= hitbox.rect.bottom
  );
}

function compareHitboxesBySpecificity(left, right) {
  const leftWidth = left.endChar - left.startChar;
  const rightWidth = right.endChar - right.startChar;

  if (leftWidth !== rightWidth) {
    return leftWidth - rightWidth;
  }

  if (left.startChar !== right.startChar) {
    return right.startChar - left.startChar;
  }

  if (left.endChar !== right.endChar) {
    return left.endChar - right.endChar;
  }

  return right.annotationIndex - left.annotationIndex;
}

export function useCustomHighlightOverlay({
  chunks,
  selectedAnnotationIds,
  rootElement,
  onLayoutRefresh = () => {},
}) {
  const supportsCustomHighlight = supportsCustomHighlightApi();
  const paneInstanceId = `pane-${getCurrentInstance()?.uid ?? 'unknown'}`;
  const customHighlightNames = new Set();
  const customHighlightHitboxes = ref([]);
  const selectedAnnotationSet = computed(
    () => new Set(Array.isArray(unref(selectedAnnotationIds)) ? unref(selectedAnnotationIds) : [])
  );

  let customHighlightStyleElement = null;
  let customHighlightResizeObserver = null;
  let customHighlightFontListenersAttached = false;
  let layoutRefreshFrame = null;

  function getChunks() {
    return Array.isArray(unref(chunks)) ? unref(chunks) : [];
  }

  function findChunkTextElement(chunkId) {
    return rootElement.value?.querySelector(`[data-chunk-text-id="${chunkId}"]`) || null;
  }

  function getHighlightName(annotationId, selected) {
    return selected
      ? `${paneInstanceId}-annotation-selected-${annotationId}`
      : `${paneInstanceId}-annotation-${annotationId}`;
  }

  function ensureCustomHighlightStyleElement() {
    if (!supportsCustomHighlight) {
      return null;
    }

    if (!customHighlightStyleElement) {
      customHighlightStyleElement = document.createElement('style');
      customHighlightStyleElement.setAttribute('data-annotated-document-highlight-style', 'true');
      customHighlightStyleElement.setAttribute('data-highlight-owner', paneInstanceId);
      document.head.appendChild(customHighlightStyleElement);
    }

    return customHighlightStyleElement;
  }

  function syncCustomHighlightStyles() {
    if (!supportsCustomHighlight) {
      return;
    }

    const styleElement = ensureCustomHighlightStyleElement();
    const rules = [];
    const seenAnnotationIds = new Set();

    getChunks().forEach((chunk) => {
      getSpanAnnotations(chunk).forEach((annotation) => {
        if (seenAnnotationIds.has(annotation.id)) {
          return;
        }

        seenAnnotationIds.add(annotation.id);

        const baseName = getHighlightName(annotation.id, false);
        const selectedName = getHighlightName(annotation.id, true);

        rules.push(
          `::highlight(${baseName}) { background-color: rgba(var(--v-theme-warning), 0.24); text-decoration: underline; text-decoration-color: rgba(var(--v-theme-warning), 0.72); text-decoration-thickness: 0.12em; text-underline-offset: 0.16em; }`
        );
        rules.push(
          `::highlight(${selectedName}) { background-color: rgba(var(--v-theme-error), 0.22); text-decoration: underline; text-decoration-color: rgba(var(--v-theme-error), 0.8); text-decoration-thickness: 0.14em; text-underline-offset: 0.16em; }`
        );
      });
    });

    styleElement.textContent = rules.join('\n');
  }

  function clearCustomHighlights() {
    if (!supportsCustomHighlight || !globalThis.CSS?.highlights) {
      customHighlightNames.clear();
      customHighlightHitboxes.value = [];
      return;
    }

    customHighlightNames.forEach((name) => {
      globalThis.CSS.highlights.delete(name);
    });
    customHighlightNames.clear();
    customHighlightHitboxes.value = [];
  }

  function buildCustomHighlightRange(element, annotation, chunkText) {
    if (!supportsCustomHighlight || !element) {
      return null;
    }

    const textNodes = collectTextNodes(element);
    if (textNodes.length === 0) {
      return null;
    }

    const totalLength = textNodes.reduce(
      (sum, textNode) => sum + (textNode.textContent?.length || 0),
      0
    );
    const fallbackLength = chunkText.length || 0;
    const safeLength = Math.max(totalLength, fallbackLength);

    if (safeLength === 0) {
      return null;
    }

    const start = Math.max(0, Math.min(annotation.start_char, safeLength));
    const end = Math.max(start, Math.min(annotation.end_char, safeLength));

    if (end <= start) {
      return null;
    }

    const range = new globalThis.Range();
    const startPosition = resolveTextPosition(textNodes, start);
    const endPosition = resolveTextPosition(textNodes, end);

    if (!startPosition || !endPosition) {
      return null;
    }

    range.setStart(startPosition.node, startPosition.offset);
    range.setEnd(endPosition.node, endPosition.offset);

    return range;
  }

  function buildHitboxesForRange(range, annotation, chunkId, chunkIndex, annotationIndex) {
    const rects = typeof range.getClientRects === 'function' ? [...range.getClientRects()] : [];
    const usableRects =
      rects.length > 0 ? rects : [range.getBoundingClientRect?.()].filter(Boolean);

    return usableRects.map((rect, rectIndex) => ({
      key: `custom-hitbox-${annotation.id}-${chunkIndex}-${annotationIndex}-${rectIndex}`,
      chunkId,
      annotationId: annotation.id,
      detailText: annotation.matched_text_in_chunk || '',
      selectedText: annotation.matched_text_in_chunk || '',
      startChar: annotation.start_char,
      endChar: annotation.end_char,
      annotationIndex,
      rect,
      target: rectToTarget(rect),
    }));
  }

  function refreshCustomHighlightGeometry() {
    if (!supportsCustomHighlight) {
      customHighlightHitboxes.value = [];
      return;
    }

    const hitboxes = [];

    getChunks().forEach((chunk, chunkIndex) => {
      const element = findChunkTextElement(chunk.chunk_id);

      getSpanAnnotations(chunk).forEach((annotation, annotationIndex) => {
        const range = buildCustomHighlightRange(element, annotation, chunk.text || '');
        if (!range) {
          return;
        }

        hitboxes.push(
          ...buildHitboxesForRange(range, annotation, chunk.chunk_id, chunkIndex, annotationIndex)
        );
      });
    });

    customHighlightHitboxes.value = hitboxes;
  }

  function syncCustomHighlights() {
    if (!supportsCustomHighlight) {
      return;
    }

    clearCustomHighlights();
    syncCustomHighlightStyles();

    const groupedRanges = new Map();
    const hitboxes = [];

    getChunks().forEach((chunk, chunkIndex) => {
      const element = findChunkTextElement(chunk.chunk_id);

      getSpanAnnotations(chunk).forEach((annotation, annotationIndex) => {
        const range = buildCustomHighlightRange(element, annotation, chunk.text || '');
        if (!range) {
          return;
        }

        const highlightName = getHighlightName(
          annotation.id,
          selectedAnnotationSet.value.has(annotation.id)
        );

        if (!groupedRanges.has(highlightName)) {
          groupedRanges.set(highlightName, []);
        }

        groupedRanges.get(highlightName).push(range);
        hitboxes.push(
          ...buildHitboxesForRange(range, annotation, chunk.chunk_id, chunkIndex, annotationIndex)
        );
      });
    });

    groupedRanges.forEach((ranges, highlightName) => {
      globalThis.CSS.highlights.set(highlightName, new globalThis.Highlight(...ranges));
      customHighlightNames.add(highlightName);
    });

    customHighlightHitboxes.value = hitboxes;
    onLayoutRefresh();
  }

  function getAnchorTarget(anchor, referenceTarget = null) {
    if (!anchor) {
      return null;
    }

    if (anchor.type === 'selection') {
      return rectToTarget(anchor.range?.getBoundingClientRect?.());
    }

    if (anchor.type === 'mark-element') {
      if (anchor.element?.isConnected !== false) {
        const rect = anchor.element?.getBoundingClientRect?.();

        if (rect) {
          return rectToTarget(rect);
        }
      }

      if (anchor.annotationId) {
        const markElement = rootElement.value?.querySelector(
          `[data-annotation-id="${anchor.annotationId}"]`
        );

        return rectToTarget(markElement?.getBoundingClientRect?.());
      }
    }

    if (anchor.type === 'custom-hitbox') {
      const exactHitbox = customHighlightHitboxes.value.find((hitbox) => hitbox.key === anchor.key);

      if (exactHitbox) {
        return exactHitbox.target;
      }

      if (anchor.annotationId) {
        const matchingHitboxes = customHighlightHitboxes.value.filter(
          (hitbox) => hitbox.annotationId === anchor.annotationId
        );

        if (matchingHitboxes.length > 0) {
          const fallbackTarget =
            referenceTarget || anchor.target || customHighlightHitboxes.value[0]?.target || null;

          if (!fallbackTarget) {
            return matchingHitboxes[0].target;
          }

          return matchingHitboxes.slice().sort((left, right) => {
            const leftDistance =
              ((left.target?.x ?? 0) - fallbackTarget.x) ** 2 +
              ((left.target?.y ?? 0) - fallbackTarget.y) ** 2;
            const rightDistance =
              ((right.target?.x ?? 0) - fallbackTarget.x) ** 2 +
              ((right.target?.y ?? 0) - fallbackTarget.y) ** 2;

            return leftDistance - rightDistance;
          })[0]?.target;
        }
      }
    }

    return null;
  }

  function refreshLayoutState() {
    refreshCustomHighlightGeometry();
    onLayoutRefresh();
  }

  function scheduleLayoutRefresh() {
    if (!supportsCustomHighlight || layoutRefreshFrame != null) {
      return;
    }

    if (typeof window.requestAnimationFrame !== 'function') {
      refreshLayoutState();
      return;
    }

    layoutRefreshFrame = window.requestAnimationFrame(() => {
      layoutRefreshFrame = null;
      refreshLayoutState();
    });
  }

  function handleRootResize(_entries, _observer) {
    scheduleLayoutRefresh();
  }

  function findHitboxForEvent(chunkId, event) {
    if (!supportsCustomHighlight) {
      return null;
    }

    return (
      customHighlightHitboxes.value
        .filter((hitbox) => hitbox.chunkId === chunkId && hitboxContainsPoint(hitbox, event))
        .sort(compareHitboxesBySpecificity)[0] || null
    );
  }

  onMounted(() => {
    if (!supportsCustomHighlight) {
      return;
    }

    window.addEventListener('scroll', scheduleLayoutRefresh, true);
    window.addEventListener('resize', scheduleLayoutRefresh);

    if (typeof ResizeObserver !== 'undefined' && rootElement.value) {
      customHighlightResizeObserver = new ResizeObserver(handleRootResize);
      customHighlightResizeObserver.observe(rootElement.value);
    }

    if (document.fonts?.addEventListener) {
      document.fonts.addEventListener('loadingdone', scheduleLayoutRefresh);
      document.fonts.addEventListener('loadingerror', scheduleLayoutRefresh);
      customHighlightFontListenersAttached = true;
    }
  });

  watch(
    () => [unref(chunks), unref(selectedAnnotationIds)],
    async () => {
      if (!supportsCustomHighlight) {
        customHighlightHitboxes.value = [];
        return;
      }

      await nextTick();
      syncCustomHighlights();
    },
    { deep: true, immediate: true, flush: 'post' }
  );

  onBeforeUnmount(() => {
    if (supportsCustomHighlight) {
      window.removeEventListener('scroll', scheduleLayoutRefresh, true);
      window.removeEventListener('resize', scheduleLayoutRefresh);
      customHighlightResizeObserver?.disconnect();
      customHighlightResizeObserver = null;

      if (customHighlightFontListenersAttached) {
        document.fonts?.removeEventListener('loadingdone', scheduleLayoutRefresh);
        document.fonts?.removeEventListener('loadingerror', scheduleLayoutRefresh);
        customHighlightFontListenersAttached = false;
      }

      if (layoutRefreshFrame != null && typeof window.cancelAnimationFrame === 'function') {
        window.cancelAnimationFrame(layoutRefreshFrame);
        layoutRefreshFrame = null;
      }
    }

    clearCustomHighlights();
    customHighlightStyleElement?.remove();
    customHighlightStyleElement = null;
  });

  return {
    supportsCustomHighlight,
    customHighlightHitboxes,
    syncCustomHighlights,
    clearCustomHighlights,
    getAnchorTarget,
    refreshCustomHighlightGeometry,
    findHitboxForEvent,
  };
}
