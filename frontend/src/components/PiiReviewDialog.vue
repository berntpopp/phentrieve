<template>
  <v-dialog
    v-model="dialogVisible"
    max-width="560"
    persistent
    :content-props="{ role: 'alertdialog', 'aria-labelledby': 'pii-review-title' }"
  >
    <v-card>
      <v-card-title id="pii-review-title" class="d-flex align-center ga-2">
        <v-icon icon="mdi-shield-alert-outline" color="warning" />
        <span>{{ $t('queryInterface.piiReview.title') }}</span>
      </v-card-title>

      <v-card-text>
        <p class="mb-4">{{ $t('queryInterface.piiReview.description') }}</p>

        <v-alert type="warning" variant="tonal" class="mb-4">
          {{ $t('queryInterface.piiReview.overrideNotice') }}
        </v-alert>

        <div v-if="hasHighFindings" class="mb-4">
          <div class="text-subtitle-2 mb-2">
            {{ $t('queryInterface.piiReview.willRedact') }}
          </div>
          <div class="d-flex flex-wrap ga-2">
            <v-chip
              v-for="item in highCategoryItems"
              :key="item.category"
              color="warning"
              variant="tonal"
            >
              {{ item.label }} ({{ item.count }})
            </v-chip>
          </div>
        </div>

        <div v-if="hasReviewFindings" class="mb-4">
          <div class="text-subtitle-2 mb-2">
            {{ $t('queryInterface.piiReview.needsReview') }}
          </div>
          <div class="d-flex flex-wrap ga-2">
            <v-chip
              v-for="item in reviewCategoryItems"
              :key="item.category"
              color="info"
              variant="tonal"
            >
              {{ item.label }} ({{ item.count }})
            </v-chip>
          </div>
        </div>

        <p class="text-caption mb-0">
          {{ $t('queryInterface.piiReview.safetyAid') }}
        </p>
      </v-card-text>

      <v-card-actions class="flex-wrap ga-2">
        <v-btn data-testid="pii-cancel" variant="text" @click="$emit('cancel')">
          {{ $t('queryInterface.piiReview.cancel') }}
        </v-btn>
        <v-spacer />
        <v-btn data-testid="pii-redact" variant="tonal" @click="$emit('redact')">
          {{ $t('queryInterface.piiReview.redact') }}
        </v-btn>
        <v-btn data-testid="pii-continue" color="primary" @click="$emit('continue')">
          {{ $t('queryInterface.piiReview.continue') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script>
export default {
  name: 'PiiReviewDialog',
  props: {
    modelValue: {
      type: Boolean,
      default: false,
    },
    summary: {
      type: Object,
      default: () => ({ high: {}, review: {} }),
    },
  },
  emits: ['update:modelValue', 'cancel', 'redact', 'continue'],
  computed: {
    dialogVisible: {
      get() {
        return this.modelValue;
      },
      set(value) {
        this.$emit('update:modelValue', value);
      },
    },
    highCategoryItems() {
      return this.categoryItems(this.summary.high);
    },
    reviewCategoryItems() {
      return this.categoryItems(this.summary.review);
    },
    hasHighFindings() {
      return this.highCategoryItems.length > 0;
    },
    hasReviewFindings() {
      return this.reviewCategoryItems.length > 0;
    },
  },
  methods: {
    categoryItems(categories = {}) {
      return Object.entries(categories)
        .filter(([, count]) => count > 0)
        .map(([category, count]) => ({
          category,
          count,
          label: this.getCategoryLabel(category),
        }));
    },
    getCategoryLabel(category) {
      return this.$t(`queryInterface.piiReview.categories.${category}`);
    },
  },
};
</script>
