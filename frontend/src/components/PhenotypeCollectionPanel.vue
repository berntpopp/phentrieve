<template>
  <div>
    <!-- Floating action button for collection panel -->
    <v-tooltip
      location="left"
      :text="$t('queryInterface.tooltips.phenotypeCollection')"
      :content-props="{ 'aria-label': $t('queryInterface.tooltips.phenotypeCollection') }"
    >
      <template #activator="{ props }">
        <v-btn
          v-bind="props"
          class="collection-fab collection-fab-position"
          color="secondary"
          icon
          position="fixed"
          location="bottom right"
          size="large"
          elevation="3"
          :aria-label="$t('queryInterface.phenotypeCollection.aria.openPanel')"
          data-tutorial-step="collection-fab"
          @click="$emit('toggle-panel')"
        >
          <v-badge :content="phenotypes.length" :model-value="phenotypes.length > 0" color="error">
            <v-icon>mdi-format-list-checks</v-icon>
          </v-badge>
        </v-btn>
      </template>
    </v-tooltip>

    <!-- Collection Panel -->
    <v-navigation-drawer
      :model-value="panelOpen"
      location="right"
      width="400"
      temporary
      style="z-index: 1500"
      :aria-label="$t('queryInterface.phenotypeCollection.aria.panel')"
      @update:model-value="$emit('update:panelOpen', $event)"
    >
      <v-list-item class="pl-2 pr-1">
        <v-list-item-title class="text-h6">Case Workspace</v-list-item-title>
        <template #append>
          <v-btn
            icon
            :aria-label="$t('queryInterface.phenotypeCollection.close')"
            variant="text"
            density="compact"
            @click="$emit('toggle-panel')"
          >
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </template>
      </v-list-item>

      <v-divider />

      <v-list v-if="phenotypes.length > 0" class="pt-0">
        <v-list-subheader>
          {{
            $t('queryInterface.phenotypeCollection.count', {
              count: phenotypes.length,
            })
          }}
        </v-list-subheader>

        <v-list-item
          v-for="(phenotype, index) in phenotypes"
          :key="phenotype.hpo_id + '-' + index"
          density="compact"
          class="py-1"
        >
          <v-list-item-title>
            <strong>{{ phenotype.hpo_id }}</strong>
            <v-chip
              size="x-small"
              class="ml-2"
              :color="
                phenotype.assertion_status === 'negated' ? 'pink-lighten-1' : 'green-lighten-1'
              "
              label
              variant="flat"
            >
              {{
                $t(
                  `queryInterface.phenotypeCollection.assertionStatus.${phenotype.assertion_status || 'affirmed'}`
                )
              }}
            </v-chip>
          </v-list-item-title>
          <v-list-item-subtitle class="wrap-text">
            {{ phenotype.label }}
          </v-list-item-subtitle>

          <template #append>
            <v-tooltip
              :text="$t('queryInterface.phenotypeCollection.assertionToggle')"
              location="start"
              :content-props="{
                'aria-label': $t('queryInterface.phenotypeCollection.assertionToggle'),
              }"
            >
              <template #activator="{ props }">
                <v-btn
                  v-bind="props"
                  :icon="
                    phenotype.assertion_status === 'negated'
                      ? 'mdi-check-circle-outline'
                      : 'mdi-close-circle-outline'
                  "
                  variant="text"
                  density="compact"
                  :color="phenotype.assertion_status === 'negated' ? 'success' : 'error'"
                  class="mr-0"
                  :aria-label="
                    $t('queryInterface.phenotypeCollection.aria.toggleAssertion', {
                      label: phenotype.label,
                      id: phenotype.hpo_id,
                    })
                  "
                  @click="$emit('toggle-assertion', index)"
                />
              </template>
            </v-tooltip>
            <v-btn
              icon="mdi-delete-outline"
              variant="text"
              density="compact"
              color="grey-darken-1"
              :aria-label="
                $t('queryInterface.phenotypeCollection.aria.removeItem', {
                  label: phenotype.label,
                  id: phenotype.hpo_id,
                })
              "
              @click="$emit('remove', index)"
            />
          </template>
        </v-list-item>
      </v-list>

      <v-sheet v-else class="pa-4 text-center">
        <v-icon size="x-large" color="grey-darken-1" class="mb-2"> mdi-tray-plus </v-icon>
        <div class="text-body-1 text-grey-darken-2">
          {{ $t('queryInterface.phenotypeCollection.empty') }}
        </div>
        <div class="text-caption text-grey-darken-3 mt-2">
          {{ $t('queryInterface.phenotypeCollection.instructions') }}
          <v-icon size="small"> mdi-plus-circle-outline </v-icon>
        </div>
      </v-sheet>

      <v-divider class="mt-4" />
      <v-list-subheader>
        {{ $t('queryInterface.phenotypeCollection.subjectInfoHeader') }}
      </v-list-subheader>
      <div class="pa-3">
        <v-tooltip
          location="bottom"
          :text="$t('queryInterface.tooltips.subjectId')"
          :content-props="{ 'aria-label': $t('queryInterface.tooltips.subjectId') }"
        >
          <template #activator="{ props }">
            <v-text-field
              v-bind="props"
              :model-value="subjectId"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              :aria-label="$t('queryInterface.phenotypeCollection.aria.subjectId')"
              bg-color="white"
              color="primary"
              @update:model-value="$emit('update:subjectId', $event)"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.phenotypeCollection.subjectId')
                }}</span>
              </template>
            </v-text-field>
          </template>
        </v-tooltip>

        <v-tooltip
          location="bottom"
          :text="$t('queryInterface.tooltips.sex')"
          :content-props="{ 'aria-label': $t('queryInterface.tooltips.sex') }"
        >
          <template #activator="{ props }">
            <v-select
              v-bind="props"
              :model-value="sex"
              :items="sexOptions"
              item-title="title"
              item-value="value"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              clearable
              :aria-label="$t('queryInterface.phenotypeCollection.aria.sex')"
              bg-color="white"
              color="primary"
              @update:model-value="$emit('update:sex', $event)"
            >
              <template #label>
                <span class="text-caption">{{ $t('queryInterface.phenotypeCollection.sex') }}</span>
              </template>
            </v-select>
          </template>
        </v-tooltip>

        <v-tooltip
          location="bottom"
          :text="$t('queryInterface.tooltips.dateOfBirth')"
          :content-props="{ 'aria-label': $t('queryInterface.tooltips.dateOfBirth') }"
        >
          <template #activator="{ props }">
            <v-text-field
              v-bind="props"
              :model-value="dateOfBirth"
              placeholder="YYYY-MM-DD"
              density="compact"
              variant="outlined"
              hide-details="auto"
              class="mb-3"
              clearable
              type="date"
              :aria-label="$t('queryInterface.phenotypeCollection.aria.dateOfBirth')"
              bg-color="white"
              color="primary"
              @update:model-value="$emit('update:dateOfBirth', $event)"
            >
              <template #label>
                <span class="text-caption">{{
                  $t('queryInterface.phenotypeCollection.dateOfBirth')
                }}</span>
              </template>
            </v-text-field>
          </template>
        </v-tooltip>
      </div>

      <template #append>
        <v-divider />
        <div class="pa-3">
          <v-btn
            block
            color="primary"
            class="mb-2"
            prepend-icon="mdi-download-box-outline"
            :disabled="phenotypes.length === 0"
            :aria-label="$t('queryInterface.phenotypeCollection.aria.exportPhenopacket')"
            @click="$emit('export-json')"
          >
            {{ $t('queryInterface.phenotypeCollection.exportPhenopacket') }}
          </v-btn>
          <v-btn
            block
            variant="outlined"
            color="primary"
            class="mb-2"
            prepend-icon="mdi-text-box-outline"
            :disabled="phenotypes.length === 0"
            @click="$emit('export-text')"
          >
            {{ $t('queryInterface.phenotypeCollection.exportText') }}
          </v-btn>
          <v-btn
            block
            variant="tonal"
            color="error"
            prepend-icon="mdi-delete-sweep-outline"
            :disabled="phenotypes.length === 0"
            @click="$emit('clear')"
          >
            {{ $t('queryInterface.phenotypeCollection.clear') }}
          </v-btn>
        </div>
      </template>
    </v-navigation-drawer>
  </div>
</template>

<script setup>
/**
 * PhenotypeCollectionPanel - Temporary case workspace bridge for the HPO collection panel.
 * Extracted from QueryInterface.vue to reduce component complexity.
 */
defineProps({
  phenotypes: { type: Array, default: () => [] },
  panelOpen: { type: Boolean, default: false },
  subjectId: { type: String, default: '' },
  sex: { type: Number, default: null },
  dateOfBirth: { type: String, default: null },
  sexOptions: { type: Array, default: () => [] },
});

defineEmits([
  'toggle-panel',
  'update:panelOpen',
  'remove',
  'toggle-assertion',
  'export-text',
  'export-json',
  'clear',
  'update:subjectId',
  'update:sex',
  'update:dateOfBirth',
]);
</script>

<style scoped>
.collection-fab-position {
  margin: 16px;
  bottom: 72px !important;
  right: 16px !important;
  z-index: 1050;
}

.v-navigation-drawer .v-list-item-title {
  font-weight: 500;
}
.v-navigation-drawer .v-list-item-subtitle.wrap-text {
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  line-height: 1.3em;
  max-height: 3.9em;
}
.v-navigation-drawer .v-btn {
  text-transform: none;
}
</style>
