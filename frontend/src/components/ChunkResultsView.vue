<template>
  <div>
    <h3 class="text-h6 my-4">
      {{ $t('resultsDisplay.textProcess.chunksTitle', 'Processed Chunks & Per-Chunk HPO Terms') }}
    </h3>
    <v-expansion-panels v-if="chunks && chunks.length > 0" v-model="openChunkPanels">
      <v-expansion-panel
        v-for="chunk in chunks"
        :key="chunk.chunk_id"
        :ref="
          (el) => {
            if (el) chunkPanelRefs[chunk.chunk_id] = el;
          }
        "
      >
        <v-expansion-panel-title>
          <div class="d-flex align-center">
            <span class="text-truncate">
              {{ $t('resultsDisplay.textProcess.chunkLabel', 'Chunk') }} {{ chunk.chunk_id }}:
              {{ chunk.text.substring(0, 50) }}...
            </span>
            <v-chip
              size="small"
              :color="
                chunk.status === 'negated'
                  ? 'error'
                  : chunk.status === 'affirmed'
                    ? 'success'
                    : 'grey'
              "
              class="ml-2"
            >
              {{ chunk.status || 'unknown' }}
            </v-chip>
          </div>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <p
            :ref="
              (el) => {
                if (el) chunkTextRefs.set(chunk.chunk_id, el);
                else chunkTextRefs.delete(chunk.chunk_id);
              }
            "
            class="font-italic mb-2 chunk-text-displayable"
          >
            <span
              v-for="(segment, segIdx) in getHighlightedChunkSegments(chunk)"
              :key="segIdx"
              :class="{ 'highlighted-text-span': segment.isHighlighted }"
            >
              {{ segment.text }}
            </span>
          </p>
          <div v-if="chunk.assertion_details && chunk.assertion_details.final_status">
            <small
              >({{ $t('resultsDisplay.textProcess.assertionDetail', 'Assertion Method:') }}
              {{ chunk.assertion_details.combination_strategy }},
              {{ $t('resultsDisplay.textProcess.finalStatus', 'Final Status:') }}
              {{ chunk.assertion_details.final_status }})</small
            >
          </div>

          <!-- Per-chunk HPO terms display -->
          <div
            v-if="chunk.hpo_matches && chunk.hpo_matches.length > 0"
            class="mt-3 per-chunk-matches"
          >
            <h4 class="text-subtitle-2 mb-1">
              {{
                $t('resultsDisplay.textProcess.hpoInChunkTitle', 'HPO Terms found in this Chunk:')
              }}
            </h4>
            <v-list density="compact" class="pa-0" style="background-color: transparent">
              <v-list-item
                v-for="(match, matchIndex) in chunk.hpo_matches"
                :key="`chunk-${chunk.chunk_id}-match-${matchIndex}`"
                class="mb-1 pa-1"
                variant="tonal"
                density="compact"
                rounded="sm"
                color="blue-grey-lighten-5"
              >
                <div class="d-flex justify-space-between align-center w-100">
                  <div class="text-caption">
                    <a
                      :href="hpoTermUrl(match.hpo_id)"
                      target="_blank"
                      rel="noopener noreferrer"
                      class="hpo-link"
                    >
                      <strong class="mr-1">{{ match.hpo_id }}</strong
                      >{{ match.name }}
                      <v-icon size="x-small" class="ml-1" color="primary">mdi-open-in-new</v-icon>
                    </a>
                  </div>
                  <SimilarityScore
                    :score="match.score"
                    type="similarity"
                    :decimals="2"
                    :show-animation="false"
                    class="ml-2"
                  />
                </div>
              </v-list-item>
            </v-list>
          </div>
          <div v-else class="text-caption text-medium-emphasis mt-2">
            {{
              $t(
                'resultsDisplay.textProcess.noChunkHPOTermsMatched',
                'No HPO terms met the retrieval threshold for this specific chunk.'
              )
            }}
          </div>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
    <v-alert v-else type="info">
      {{ $t('resultsDisplay.textProcess.noChunksProcessed', 'No text chunks were processed.') }}
    </v-alert>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue';
import SimilarityScore from './SimilarityScore.vue';
import { HPO_TERM_URL } from '../constants/urls';

const props = defineProps({
  chunks: { type: Array, default: () => [] },
  highlightedAttributions: { type: Array, default: () => [] },
});

const openChunkPanels = ref([]);
const chunkPanelRefs = reactive({});
const chunkTextRefs = new Map();

function hpoTermUrl(hpoId) {
  return HPO_TERM_URL(hpoId);
}

function getHighlightedChunkSegments(chunk) {
  const chunkAttributions = (props.highlightedAttributions || []).filter(
    (attr) => attr.chunkId === chunk.chunk_id
  );

  if (chunkAttributions.length === 0) {
    return [{ text: `"${chunk.text}"`, isHighlighted: false }];
  }

  chunkAttributions.sort((a, b) => a.start - b.start);

  const segments = [];
  let lastEnd = 0;

  for (const attr of chunkAttributions) {
    if (attr.start > lastEnd) {
      segments.push({ text: chunk.text.substring(lastEnd, attr.start), isHighlighted: false });
    }
    segments.push({ text: chunk.text.substring(attr.start, attr.end), isHighlighted: true });
    lastEnd = attr.end;
  }

  if (lastEnd < chunk.text.length) {
    segments.push({ text: chunk.text.substring(lastEnd), isHighlighted: false });
  }

  return segments;
}

function flashChunkText(chunkId) {
  const chunkTextElement = chunkTextRefs.get(chunkId);
  if (!chunkTextElement) {
    return false;
  }

  chunkTextElement.classList.add('flash-highlight');
  setTimeout(() => {
    chunkTextElement.classList.remove('flash-highlight');
  }, 1500);

  return true;
}

// Expose for parent scrollToChunk
defineExpose({ chunkPanelRefs, flashChunkText, openChunkPanels });
</script>

<style scoped>
.hpo-link {
  text-decoration: none;
  color: inherit;
  display: inline-flex;
  align-items: center;
  transition: color 0.2s;
}

.hpo-link:hover {
  color: var(--v-theme-primary);
}

.highlighted-text-span {
  background-color: rgba(255, 236, 179, 0.8);
  border-radius: 3px;
  padding: 0.5px 2px;
  box-shadow: 0 0 3px rgba(255, 210, 50, 0.5);
}

.flash-highlight {
  animation: flashHighlightAnimation 0.75s 2 ease-in-out;
}

@keyframes flashHighlightAnimation {
  0%,
  100% {
    background-color: transparent;
  }
  50% {
    background-color: rgba(var(--v-theme-primary), 0.15);
  }
}
</style>
