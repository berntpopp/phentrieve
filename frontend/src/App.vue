<template>
  <v-app theme="light">
    <v-main class="bg-grey-lighten-4">
      <HomeView />
    </v-main>
    
    <v-footer app class="d-flex justify-space-between pa-2" style="z-index: 1;">
      <div class="d-flex align-center">
        <v-btn
          variant="text"
          density="compact"
          class="text-body-2 text-primary mr-2"
          prepend-icon="mdi-alert-circle-outline"
          @click="showDisclaimerDialog"
        >
          Disclaimer
          <template v-if="disclaimerStore.isAcknowledged" #append>
            <v-tooltip location="top">
              <template v-slot:activator="{ props }">
                <v-icon
                  v-bind="props"
                  size="small"
                  color="success"
                  class="ml-1"
                >
                  mdi-check-circle
                </v-icon>
              </template>
              <span>Acknowledged on {{ disclaimerStore.formattedAcknowledgmentDate }}</span>
            </v-tooltip>
          </template>
        </v-btn>
      </div>
      
      <div class="d-flex align-center">
        <v-btn
          variant="text"
          density="compact"
          class="text-body-2 mr-2"
          prepend-icon="mdi-text-box-search-outline"
          @click="logStore.toggleViewer"
        >
          Logs
        </v-btn>
        <div class="text-body-2 text-grey-darken-3 mr-2">&copy; {{ new Date().getFullYear() }} Phentrieve</div>
        <v-tooltip location="top" text="View source code on GitHub">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon="mdi-github"
              size="small"
              href="https://github.com/berntpopp/rag-hpo-testing"
              target="_blank"
              rel="noopener noreferrer"
              variant="text"
              color="grey"
              aria-label="View Phentrieve project on GitHub"
              class="mr-1"
            ></v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="top" text="View project documentation">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon="mdi-book-open-page-variant-outline"
              size="small"
              href="https://github.com/berntpopp/rag-hpo-testing/wiki"
              target="_blank"
              rel="noopener noreferrer"
              variant="text"
              color="grey"
              aria-label="View project documentation"
              class="mr-1"
            ></v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="top" text="View project license">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon="mdi-license"
              size="small"
              href="https://github.com/berntpopp/rag-hpo-testing/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              variant="text"
              color="grey"
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

    <!-- Log Viewer -->
    <LogViewer />
  </v-app>
</template>

<script>
import HomeView from './views/HomeView.vue'
import DisclaimerDialog from './components/DisclaimerDialog.vue'
import LogViewer from './components/LogViewer.vue'
import { useDisclaimerStore } from './stores/disclaimer'
import { useLogStore } from './stores/log'
import { logService } from './services/logService'

export default {
  name: 'App',
  components: {
    HomeView,
    DisclaimerDialog,
    LogViewer
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
