<template>
  <v-app theme="light">
    <v-main class="bg-grey-lighten-4">
      <router-view v-slot="{ Component }">
        <component :is="Component" />
      </router-view>
      <LogViewer />
    </v-main>
    
    <v-footer app class="d-flex justify-space-between pa-2" style="z-index: 1;" role="contentinfo">
      <div class="d-flex align-center">
        <v-tooltip location="top" :text="$t('app.footer.disclaimerTooltip')" role="tooltip" aria-label="Disclaimer information">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              variant="text"
              density="compact"
              class="text-body-2 text-primary mr-2"
              prepend-icon="mdi-alert-circle-outline"
              @click="showDisclaimerDialog"
              aria-label="View disclaimer information"
            >
          {{ $t('app.footer.disclaimer') }}
          <template v-if="disclaimerStore.isAcknowledged" #append>
            <v-tooltip location="top" role="tooltip" aria-label="Disclaimer acknowledgment date">
              <template v-slot:activator="{ props }">
                <v-icon
                  v-bind="props"
                  size="small"
                  color="success"
                  class="ml-1"
                  aria-label="Disclaimer acknowledged"
                >
                  mdi-check-circle
                </v-icon>
              </template>
              <span>{{ $t('app.footer.disclaimerAcknowledgedTooltip', { date: disclaimerStore.formattedAcknowledgmentDate }) }}</span>
            </v-tooltip>
          </template>
            </v-btn>
          </template>
        </v-tooltip>
      </div>
      
      <div class="d-flex align-center">
        <LanguageSwitcher class="mr-2" />
        <v-tooltip location="top" :text="$t('app.footer.faqTooltip')" role="tooltip" aria-label="FAQ information">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              variant="text"
              density="compact"
              class="text-body-2 text-high-emphasis mr-2"
              prepend-icon="mdi-help-circle-outline"
              :to="{ name: 'faq' }"
              aria-label="View frequently asked questions"
              color="primary"
            >
          {{ $t('app.footer.faq') }}
            </v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="top" :text="$t('app.footer.logsTooltip')" role="tooltip" aria-label="Log viewer information">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              variant="text"
              density="compact"
              class="text-body-2 text-high-emphasis mr-2"
              prepend-icon="mdi-text-box-search-outline"
              @click="logStore.toggleViewer"
              aria-label="Toggle log viewer"
              color="primary"
            >
          {{ $t('app.footer.logs') }}
            </v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="top" :text="$t('app.footer.githubTooltip')" role="tooltip" aria-label="GitHub repository link">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon="mdi-github"
              size="small"
              href="https://github.com/berntpopp/phentrieve"
              target="_blank"
              rel="noopener noreferrer"
              variant="text"
              color="primary"
              aria-label="View Phentrieve project on GitHub"
              class="mr-1"
            ></v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="top" :text="$t('app.footer.docsTooltip')" role="tooltip" aria-label="Documentation link">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon="mdi-book-open-page-variant-outline"
              size="small"
              href="https://berntpopp.github.io/phentrieve/"
              target="_blank"
              rel="noopener noreferrer"
              variant="text"
              color="primary"
              aria-label="View project documentation"
              class="mr-1"
            ></v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="top" :text="$t('app.footer.licenseTooltip')" role="tooltip" aria-label="License link">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon="mdi-license"
              size="small"
              href="https://github.com/berntpopp/phentrieve/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              variant="text"
              color="primary"
              aria-label="View project license"
            ></v-btn>
          </template>
        </v-tooltip>
      </div>
    </v-footer>
    
    <!-- Disclaimer Dialog -->
    <DisclaimerDialog
      v-model="disclaimerDialogVisible"
      @acknowledged="handleDisclaimerAcknowledged"
    />


  </v-app>
</template>

<script>
import { useDisclaimerStore } from './stores/disclaimer'
import { useLogStore } from './stores/log'
import { logService } from './services/logService'
import DisclaimerDialog from './components/DisclaimerDialog.vue'
import LogViewer from './components/LogViewer.vue'
import LanguageSwitcher from './components/LanguageSwitcher.vue'

export default {
  name: 'App',
  components: {
    DisclaimerDialog,
    LogViewer,
    LanguageSwitcher
  },
  data() {
    return {
      disclaimerDialogVisible: false
    }
  },
  computed: {
    disclaimerStore() {
      return useDisclaimerStore()
    },
    logStore() {
      return useLogStore()
    }
  },
  created() {
    logService.info('App component created')
    
    // Initialize the store
    this.disclaimerStore.initialize()
    logService.debug('Disclaimer store initialized', {
      isAcknowledged: this.disclaimerStore.isAcknowledged,
      timestamp: this.disclaimerStore.acknowledgmentTimestamp
    })
    
    // Show the disclaimer dialog if it has not been acknowledged
    if (!this.disclaimerStore.isAcknowledged) {
      this.disclaimerDialogVisible = true
      logService.info('Showing initial disclaimer dialog')
    }

    // Log application initialization
    logService.info('Application initialized')
  },
  methods: {
    showDisclaimerDialog() {
      logService.debug('Manual disclaimer dialog trigger')
      this.disclaimerDialogVisible = true
    },
    handleDisclaimerAcknowledged() {
      logService.info('User acknowledged disclaimer')
      // Save the acknowledgment to the store
      this.disclaimerStore.saveAcknowledgment()
    }
  }
}
</script>

<style>
/* Global styles */
</style>
