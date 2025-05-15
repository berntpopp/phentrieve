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
      >
        Back to Home
      </v-btn>

      <!-- Main Content -->
      <div style="max-width: 600px; width: 100%;">
          <!-- Header -->
          <div class="d-flex align-center mb-6">
            <img src="/hpo-logo.svg" alt="HPO Logo" height="40" class="mr-3">
            <h1 class="text-h4 font-weight-light">{{ faqData.pageTitle }}</h1>
          </div>

          <!-- Search Bar -->
          <v-card class="mb-6" variant="outlined">
            <v-card-text class="pa-2">
              <v-text-field
                v-model="searchQuery"
                prepend-icon="mdi-magnify"
                label="Search FAQs"
                clearable
                density="compact"
                variant="outlined"
                hide-details
              />
            </v-card-text>
          </v-card>

          <!-- FAQ Categories -->
          <v-card variant="outlined">
            <v-expansion-panels variant="accordion">
              <v-expansion-panel
                v-for="category in filteredCategories"
                :key="category.id"
                class="faq-panel"
              >
                <v-expansion-panel-title>
                  <v-row no-gutters>
                    <v-col cols="12">
                      <h2 class="text-h6 font-weight-medium">{{ category.title }}</h2>
                    </v-col>
                  </v-row>
                </v-expansion-panel-title>
                <v-expansion-panel-text>
                  <v-expansion-panels variant="accordion">
                    <v-expansion-panel
                      v-for="(qa, index) in category.questions"
                      :key="index"
                      class="qa-panel"
                    >
                      <v-expansion-panel-title>
                        <v-row no-gutters>
                          <v-col cols="12">
                            <span class="text-subtitle-1 font-weight-regular">{{ qa.question }}</span>
                          </v-col>
                        </v-row>
                      </v-expansion-panel-title>
                      <v-expansion-panel-text>
                        <div v-html="qa.answer" class="answer-content text-body-1"></div>
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
import faqData from '@/config/faqConfig.json'

export default {
  name: 'FAQView',
  data() {
    return {
      faqData,
      searchQuery: ''
    }
  },
  computed: {
    filteredCategories() {
      if (!this.searchQuery) {
        return this.faqData.categories
      }

      const query = this.searchQuery.toLowerCase()
      return this.faqData.categories.map(category => {
        const filteredQuestions = category.questions.filter(qa =>
          qa.question.toLowerCase().includes(query) ||
          qa.answer.toLowerCase().includes(query)
        )
        
        return filteredQuestions.length > 0
          ? { ...category, questions: filteredQuestions }
          : null
      }).filter(Boolean)
    }
  }
}
</script>

<style scoped>
.fill-height {
  min-height: 100vh;
  background-color: rgb(var(--v-theme-background));
}

.content-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 1rem;
  position: relative;
}

.answer-content {
  padding: 16px;
  color: rgb(var(--v-theme-on-surface));
  line-height: 1.6;
}

.answer-content :deep(p) {
  margin-bottom: 1em;
}

.answer-content :deep(ul),
.answer-content :deep(ol) {
  padding-left: 1.5em;
  margin-bottom: 1em;
}

.answer-content :deep(li) {
  margin-bottom: 0.5em;
}

.answer-content :deep(a) {
  color: rgb(var(--v-theme-primary));
  text-decoration: none;
  font-weight: 500;
}

.answer-content :deep(a:hover) {
  text-decoration: underline;
}

.answer-content :deep(strong) {
  font-weight: 600;
  color: rgb(var(--v-theme-on-surface));
}

.faq-panel {
  border-bottom: thin solid rgba(var(--v-border-color), var(--v-border-opacity));
}

.qa-panel {
  background-color: rgb(var(--v-theme-surface));
}

:deep(.v-expansion-panel-title) {
  padding: 16px;
}

:deep(.v-expansion-panel-text__wrapper) {
  padding: 0;
}
</style>
