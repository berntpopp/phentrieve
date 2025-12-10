<template>
  <div class="fill-height">
    <div class="content-wrapper">
      <!-- Back Button -->
      <v-btn
        variant="text"
        color="primary"
        class="mb-4 align-self-start"
        :to="{ name: 'home' }"
        prepend-icon="mdi-arrow-left"
        aria-label="Navigate back to home page"
      >
        {{ t('faq.backToHome') }}
      </v-btn>

      <!-- Main Content -->
      <div style="max-width: 600px; width: 100%">
        <!-- Header -->
        <div class="d-flex align-center mb-6">
          <img
            src="/hpo-logo.svg"
            alt="HPO Logo"
            width="40"
            height="40"
            class="mr-3"
            loading="lazy"
          />
          <h1 class="text-h4 font-weight-light">
            {{ t('faq.pageTitle') }}
          </h1>
        </div>

        <!-- Search Bar -->
        <v-card class="mb-6" variant="outlined" role="search">
          <v-card-text class="pa-2">
            <v-text-field
              v-model="searchQuery"
              prepend-icon="mdi-magnify"
              label="Search FAQs"
              clearable
              density="compact"
              variant="outlined"
              hide-details
              aria-label="Search frequently asked questions"
              :placeholder="t('faq.searchPlaceholder', { count: totalQuestions })"
            />
          </v-card-text>
        </v-card>

        <!-- FAQ Categories -->
        <v-card variant="outlined" role="region" aria-label="FAQ Categories">
          <v-expansion-panels variant="accordion" role="tablist">
            <v-expansion-panel
              v-for="category in filteredCategories"
              :key="category.id"
              class="faq-panel"
              role="tab"
              :aria-label="category.title + ' category'"
            >
              <v-expansion-panel-title class="text-subtitle-1 text-primary-darken-1">
                {{ t(`faq.categories.${category.id}.title`) }}
              </v-expansion-panel-title>
              <v-expansion-panel-text role="tabpanel">
                <v-expansion-panels variant="accordion">
                  <v-expansion-panel
                    v-for="(qa, index) in category.questions"
                    :key="index"
                    role="listitem"
                    :aria-label="'Question: ' + qa.question"
                  >
                    <v-expansion-panel-title class="text-body-1">
                      {{ t(`faq.categories.${category.id}.questions.${qa.id}.question`) }}
                    </v-expansion-panel-title>
                    <v-expansion-panel-text>
                      <div
                        class="answer-content text-body-1"
                        role="region"
                        :aria-label="'Answer to: ' + qa.question"
                        :style="{
                          color: 'rgba(0, 0, 0, 0.87)',
                          backgroundColor: 'rgba(0, 0, 0, 0.02)',
                          padding: '16px 20px',
                          lineHeight: '1.6',
                          borderRadius: '4px',
                        }"
                        v-html="t(`faq.categories.${category.id}.questions.${qa.id}.answer`)"
                      />
                    </v-expansion-panel-text>
                  </v-expansion-panel>
                </v-expansion-panels>
              </v-expansion-panel-text>
            </v-expansion-panel>
          </v-expansion-panels>
        </v-card>
      </div>
    </div>
  </div>
</template>

<script>
import { useI18n } from 'vue-i18n';
import faqData from '@/config/faqConfig.json';

export default {
  name: 'FAQView',
  setup() {
    const { t } = useI18n();
    return { t };
  },
  data() {
    return {
      faqData,
      searchQuery: '',
      categoriesMap: {
        general: {
          id: 'general',
          questions: [{ id: 'whatIsPhentrieve' }, { id: 'howItWorks' }, { id: 'whoCanUse' }],
        },
        technical: {
          id: 'technical',
          questions: [{ id: 'performance' }, { id: 'similarityScore' }],
        },
      },
    };
  },
  computed: {
    filteredCategories() {
      const categories = Object.values(this.categoriesMap);

      if (!this.searchQuery) {
        return categories;
      }

      const query = this.searchQuery.toLowerCase();
      return categories
        .map((category) => {
          const filteredQuestions = category.questions.filter((qa) => {
            const question = this.t(
              `faq.categories.${category.id}.questions.${qa.id}.question`
            ).toLowerCase();
            const answer = this.t(
              `faq.categories.${category.id}.questions.${qa.id}.answer`
            ).toLowerCase();
            return question.includes(query) || answer.includes(query);
          });

          return filteredQuestions.length > 0
            ? { ...category, questions: filteredQuestions }
            : null;
        })
        .filter(Boolean);
    },
    totalQuestions() {
      return Object.values(this.categoriesMap).reduce((total, category) => {
        return total + category.questions.length;
      }, 0);
    },
  },
};
</script>

<style scoped>
/* FAQ Page Styling - Modern UX with WCAG AAA Compliance */

.fill-height {
  min-height: 100vh;
  background-color: rgb(var(--v-theme-background));
  position: relative;
}

.content-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 1rem;
}

/* Deep selectors for answer content - styled via inline :style for better HMR */
.answer-content :deep(p) {
  margin-bottom: 1em;
}

.answer-content :deep(p:last-child) {
  margin-bottom: 0;
}

.answer-content :deep(h4) {
  margin-top: 1em;
  margin-bottom: 0.5em;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.9);
}

.answer-content :deep(ul),
.answer-content :deep(ol) {
  padding-left: 1.5em;
  margin-bottom: 1em;
}

.answer-content :deep(li) {
  margin-bottom: 0.5em;
}

.answer-content :deep(li:last-child) {
  margin-bottom: 0;
}

.answer-content :deep(code) {
  background-color: rgba(0, 0, 0, 0.05);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: monospace;
  font-size: 0.9em;
}

.answer-content :deep(a) {
  color: rgb(var(--v-theme-primary));
  text-decoration: none;
  transition: all 0.2s ease;
}

.answer-content :deep(a:hover) {
  text-decoration: underline;
}

.answer-content :deep(a:focus-visible) {
  outline: 2px solid rgb(var(--v-theme-primary));
  outline-offset: 2px;
}

.answer-content :deep(strong) {
  font-weight: 600;
  color: rgba(0, 0, 0, 0.95);
}

/* FAQ panel styling */
.faq-panel {
  border-bottom: thin solid rgba(var(--v-border-color), var(--v-border-opacity));
}

/* Expansion panel improvements */
:deep(.v-expansion-panel-title) {
  padding: 16px;
  font-weight: 500;
  transition: background-color 0.2s ease;
}

:deep(.v-expansion-panel-title:hover) {
  background-color: rgba(0, 0, 0, 0.02);
}

:deep(.v-expansion-panel-text__wrapper) {
  padding: 0;
}

:deep(.v-expansion-panel) {
  margin-bottom: 2px;
}

:deep(.v-expansion-panel:last-child) {
  margin-bottom: 0;
}
</style>
