<template>
  <v-expand-transition>
    <v-sheet
      v-if="visible && !disabled"
      id="advanced-options-panel"
      rounded="lg"
      elevation="1"
      class="mt-2 pa-3"
      role="region"
      aria-label="Advanced search options"
      color="white"
      style="font-size: 0.8rem"
    >
      <div class="text-subtitle-2 mb-2 px-1 font-weight-medium">
        {{ $t('queryInterface.advancedOptions.title') }}
      </div>

      <v-row dense>
        <v-col cols="12" md="6" class="pa-1">
          <v-tooltip
            location="bottom"
            :text="$t('queryInterface.tooltips.embeddingModel')"
            role="tooltip"
          >
            <template #activator="{ props }">
              <v-select
                v-bind="props"
                :model-value="selectedModel"
                :items="availableModels"
                item-title="text"
                item-value="value"
                :disabled="disabled"
                variant="outlined"
                density="compact"
                aria-label="Select embedding model"
                :aria-description="
                  'Choose the model to use for text embedding. Currently selected: ' + selectedModel
                "
                bg-color="white"
                color="primary"
                hide-details
                @update:model-value="$emit('update:selectedModel', $event)"
              >
                <template #label>
                  <span class="text-caption">{{
                    $t('queryInterface.advancedOptions.embeddingModel')
                  }}</span>
                </template>
              </v-select>
            </template>
          </v-tooltip>
        </v-col>

        <v-col cols="12" md="6" class="pa-1">
          <v-tooltip
            location="bottom"
            :text="$t('queryInterface.tooltips.similarityThreshold')"
            role="tooltip"
          >
            <template #activator="{ props }">
              <div>
                <label
                  :for="'similarity-slider'"
                  class="text-caption mb-0 d-block"
                  style="font-size: 0.7rem; padding-left: 4px"
                  >{{ $t('queryInterface.advancedOptions.similarityThreshold') }}:
                  {{ similarityThreshold.toFixed(2) }}</label
                >
                <v-slider
                  v-bind="props"
                  id="similarity-slider"
                  :model-value="similarityThreshold"
                  :disabled="disabled"
                  class="mt-0 mb-1"
                  density="compact"
                  min="0"
                  max="1"
                  step="0.01"
                  color="primary"
                  track-color="grey-lighten-2"
                  thumb-label
                  hide-details
                  aria-label="Similarity threshold slider"
                  :aria-description="
                    'Adjust minimum similarity threshold. Current value: ' +
                    similarityThreshold.toFixed(2)
                  "
                  @update:model-value="$emit('update:similarityThreshold', $event)"
                />
              </div>
            </template>
          </v-tooltip>
        </v-col>
      </v-row>

      <v-row dense>
        <v-col cols="12" md="6" class="pa-1">
          <v-tooltip
            location="bottom"
            :text="$t('queryInterface.tooltips.language')"
            role="tooltip"
          >
            <template #activator="{ props }">
              <v-select
                v-bind="props"
                :model-value="selectedLanguage"
                :items="availableLanguages"
                item-title="text"
                item-value="value"
                :disabled="disabled"
                variant="outlined"
                density="compact"
                aria-label="Select query language"
                :aria-description="
                  'Choose the language for query processing. Currently selected: ' +
                  selectedLanguage
                "
                bg-color="white"
                color="primary"
                hide-details
                @update:model-value="$emit('update:selectedLanguage', $event)"
              >
                <template #label>
                  <span class="text-caption">{{
                    $t('queryInterface.advancedOptions.language')
                  }}</span>
                </template>
              </v-select>
            </template>
          </v-tooltip>
        </v-col>

        <v-col cols="12" md="6" class="pa-1 d-flex align-center">
          <v-tooltip
            location="bottom"
            :text="$t('queryInterface.tooltips.includeDetails')"
            role="tooltip"
          >
            <template #activator="{ props }">
              <v-switch
                v-bind="props"
                :model-value="includeDetails"
                :disabled="disabled"
                :label="$t('queryInterface.advancedOptions.includeDetails')"
                color="primary"
                inset
                density="compact"
                hide-details
                class="mt-0 pt-0"
                aria-label="Include HPO term definitions and synonyms"
                @update:model-value="$emit('update:includeDetails', $event)"
              />
            </template>
          </v-tooltip>
        </v-col>
      </v-row>

      <v-divider class="my-2" />
      <div class="text-subtitle-2 mb-1 px-1 font-weight-medium">
        {{ $t('queryInterface.advancedOptions.processingModeTitle') }}
      </div>

      <v-row dense>
        <v-col cols="12" class="pa-1">
          <v-select
            :model-value="forceEndpointMode"
            :items="[
              { title: $t('queryInterface.advancedOptions.modeAutomatic'), value: null },
              { title: $t('queryInterface.advancedOptions.modeQuery'), value: 'query' },
              {
                title: $t('queryInterface.advancedOptions.modeTextProcess'),
                value: 'textProcess',
              },
            ]"
            item-title="title"
            item-value="value"
            variant="outlined"
            density="compact"
            :disabled="disabled"
            bg-color="white"
            color="primary"
            hide-details
            @update:model-value="$emit('update:forceEndpointMode', $event)"
          >
            <template #label>
              <span class="text-caption">{{
                $t('queryInterface.advancedOptions.processingModeLabel')
              }}</span>
            </template>
          </v-select>
        </v-col>
      </v-row>

      <div v-if="isTextProcessModeActive">
        <v-divider class="my-2" />
        <div class="text-subtitle-2 mb-1 px-1 font-weight-medium">
          {{ $t('queryInterface.advancedOptions.textProcessingTitle') }}
        </div>

        <v-row dense>
          <v-col cols="12" md="6" class="pa-1">
            <v-select
              :model-value="chunkingStrategy"
              :items="[
                'simple',
                'semantic',
                'detailed',
                'sliding_window',
                'sliding_window_cleaned',
                'sliding_window_punct_cleaned',
                'sliding_window_punct_conj_cleaned',
              ]"
              variant="outlined"
              density="compact"
              bg-color="white"
              color="primary"
              hide-details
              @update:model-value="$emit('update:chunkingStrategy', $event)"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.chunkingStrategy')
                }}</span>
              </template>
            </v-select>
          </v-col>
          <v-col cols="12" md="6" class="pa-1">
            <v-text-field
              :model-value="windowSize"
              type="number"
              min="1"
              variant="outlined"
              density="compact"
              bg-color="white"
              color="primary"
              hide-details
              @update:model-value="$emit('update:windowSize', Number($event))"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.windowSize')
                }}</span>
              </template>
            </v-text-field>
          </v-col>
        </v-row>

        <v-row dense>
          <v-col cols="12" md="6" class="pa-1">
            <v-text-field
              :model-value="stepSize"
              type="number"
              min="1"
              variant="outlined"
              density="compact"
              bg-color="white"
              color="primary"
              hide-details
              @update:model-value="$emit('update:stepSize', Number($event))"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.stepSize')
                }}</span>
              </template>
            </v-text-field>
          </v-col>
          <v-col cols="12" md="6" class="pa-1">
            <v-text-field
              :model-value="chunkRetrievalThreshold"
              type="number"
              step="0.01"
              min="0"
              max="1"
              variant="outlined"
              density="compact"
              bg-color="white"
              color="primary"
              hide-details
              @update:model-value="$emit('update:chunkRetrievalThreshold', Number($event))"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.chunkThreshold')
                }}</span>
              </template>
            </v-text-field>
          </v-col>
        </v-row>

        <v-row dense>
          <v-col cols="12" md="6" class="pa-1">
            <v-text-field
              :model-value="aggregatedTermConfidence"
              type="number"
              step="0.01"
              min="0"
              max="1"
              variant="outlined"
              density="compact"
              bg-color="white"
              color="primary"
              hide-details
              @update:model-value="$emit('update:aggregatedTermConfidence', Number($event))"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.aggConfidence')
                }}</span>
              </template>
            </v-text-field>
          </v-col>
          <v-col cols="12" md="6" class="pa-1 d-flex align-center">
            <v-switch
              :model-value="noAssertionDetectionForTextProcess"
              color="primary"
              hide-details
              density="compact"
              inset
              :true-value="false"
              :false-value="true"
              @update:model-value="$emit('update:noAssertionDetectionForTextProcess', $event)"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.detectAssertions')
                }}</span>
              </template>
            </v-switch>
          </v-col>
        </v-row>

        <v-row dense>
          <v-col cols="12" md="6" class="pa-1">
            <v-text-field
              :model-value="splitThreshold"
              type="number"
              step="0.01"
              min="0"
              max="1"
              variant="outlined"
              density="compact"
              bg-color="white"
              color="primary"
              hide-details
              @update:model-value="$emit('update:splitThreshold', Number($event))"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.splitThreshold')
                }}</span>
              </template>
            </v-text-field>
          </v-col>
          <v-col cols="12" md="6" class="pa-1">
            <v-text-field
              :model-value="minSegmentLength"
              type="number"
              min="1"
              variant="outlined"
              density="compact"
              bg-color="white"
              color="primary"
              hide-details
              @update:model-value="$emit('update:minSegmentLength', Number($event))"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.minSegmentLength')
                }}</span>
              </template>
            </v-text-field>
          </v-col>
        </v-row>

        <v-row dense>
          <v-col cols="12" md="6" class="pa-1">
            <v-text-field
              :model-value="numResultsPerChunk"
              type="number"
              min="1"
              variant="outlined"
              density="compact"
              hide-details
              class="mb-0"
              @update:model-value="$emit('update:numResultsPerChunk', Number($event))"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.numResultsPerChunk')
                }}</span>
              </template>
            </v-text-field>
          </v-col>
          <v-col cols="12" md="6" class="pa-1 d-flex align-center">
            <v-switch
              :model-value="topTermPerChunkForAggregation"
              color="primary"
              hide-details
              density="compact"
              inset
              @update:model-value="$emit('update:topTermPerChunkForAggregation', $event)"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.advancedOptions.topTermPerChunk')
                }}</span>
              </template>
            </v-switch>
          </v-col>
        </v-row>
      </div>
    </v-sheet>
  </v-expand-transition>
</template>

<script setup>
/**
 * AdvancedOptionsPanel - Sub-component for advanced query options.
 * Extracted from QueryInterface.vue to reduce component complexity.
 */
defineProps({
  visible: { type: Boolean, default: false },
  disabled: { type: Boolean, default: false },
  selectedModel: { type: [String, null], default: null },
  availableModels: { type: Array, default: () => [] },
  selectedLanguage: { type: [String, null], default: null },
  availableLanguages: { type: Array, default: () => [] },
  includeDetails: { type: Boolean, default: false },
  similarityThreshold: { type: Number, default: 0.5 },
  forceEndpointMode: { type: [String, null], default: null },
  isTextProcessModeActive: { type: Boolean, default: false },
  chunkingStrategy: { type: String, default: 'sliding_window_punct_conj_cleaned' },
  windowSize: { type: Number, default: 3 },
  stepSize: { type: Number, default: 1 },
  chunkRetrievalThreshold: { type: Number, default: 0.7 },
  aggregatedTermConfidence: { type: Number, default: 0.75 },
  noAssertionDetectionForTextProcess: { type: Boolean, default: false },
  splitThreshold: { type: Number, default: 0.5 },
  minSegmentLength: { type: Number, default: 2 },
  numResultsPerChunk: { type: Number, default: 3 },
  topTermPerChunkForAggregation: { type: Boolean, default: false },
});

defineEmits([
  'update:selectedModel',
  'update:selectedLanguage',
  'update:includeDetails',
  'update:similarityThreshold',
  'update:forceEndpointMode',
  'update:chunkingStrategy',
  'update:windowSize',
  'update:stepSize',
  'update:chunkRetrievalThreshold',
  'update:aggregatedTermConfidence',
  'update:noAssertionDetectionForTextProcess',
  'update:splitThreshold',
  'update:minSegmentLength',
  'update:numResultsPerChunk',
  'update:topTermPerChunkForAggregation',
]);
</script>
