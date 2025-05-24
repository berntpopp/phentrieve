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
            <div class="d-flex align-center">
              <v-btn
                v-bind="props"
                icon
                variant="text"
                color="primary"
                size="small"
                class="mr-1"
                @click="showDisclaimerDialog"
                aria-label="View disclaimer information"
              >
                <v-icon>mdi-scale-balance</v-icon>
              </v-btn>
              <v-icon
                v-if="disclaimerStore.isAcknowledged"
                size="small"
                color="success"
                class="mr-2"
                aria-label="Disclaimer acknowledged"
                v-tooltip="$t('app.footer.disclaimerAcknowledgedTooltip', { date: disclaimerStore.formattedAcknowledgmentDate })"
              >
                mdi-check-circle
              </v-icon>
            </div>
          </template>
        </v-tooltip>
      </div>
      
      <div class="d-flex align-center">
        <LanguageSwitcher class="mr-2" />
        <v-tooltip location="top" :text="$t('app.footer.tutorialTooltip')" role="tooltip" aria-label="Tutorial information">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon
              variant="text"
              color="primary"
              size="small"
              class="mr-2"
              @click="startTutorial"
              aria-label="Start guided tutorial"
            >
              <v-icon>mdi-hand-pointing-up</v-icon>
            </v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="top" :text="$t('app.footer.faqTooltip')" role="tooltip" aria-label="FAQ information">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon
              variant="text"
              color="primary"
              size="small"
              class="mr-2"
              :to="{ name: 'faq' }"
              aria-label="View frequently asked questions"
            >
              <v-icon>mdi-help-circle-outline</v-icon>
            </v-btn>
          </template>
        </v-tooltip>
        <v-tooltip location="top" :text="$t('app.footer.logsTooltip')" role="tooltip" aria-label="Log viewer information">
          <template v-slot:activator="{ props }">
            <v-btn
              v-bind="props"
              icon
              variant="text"
              color="primary"
              size="small"
              class="mr-2"
              @click="logStore.toggleViewer"
              aria-label="Toggle log viewer"
            >
              <v-icon>mdi-text-box-search-outline</v-icon>
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

    <!-- Tutorial Overlay -->
    <TutorialOverlay :visible="tutorialVisible" @update:visible="tutorialVisible = $event" />

  </v-app>
</template>

<script>
import { useDisclaimerStore } from './stores/disclaimer'
import { useLogStore } from './stores/log'
import { logService } from './services/logService'
import { tutorialService } from './services/tutorialService'
import DisclaimerDialog from './components/DisclaimerDialog.vue'
import LogViewer from './components/LogViewer.vue'
import LanguageSwitcher from './components/LanguageSwitcher.vue'
import TutorialOverlay from './components/TutorialOverlay.vue'

export default {
  name: 'App',
  components: {
    DisclaimerDialog,
    LogViewer,
    LanguageSwitcher,
    TutorialOverlay
  },
  data() {
    return {
      disclaimerDialogVisible: false,
      tutorialVisible: false
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

    // Initialize tutorial steps
    this.initializeTutorialSteps()

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
    },
    startTutorial() {
      logService.info('Starting tutorial')
      
      // Set up handlers for tutorial completion
      tutorialService.onComplete(() => {
        // Add a slight delay before closing panels to ensure they're properly detected
        setTimeout(() => {
          this.closeAnyOpenPanels();
          this.tutorialVisible = false;
          logService.info('Tutorial completed and panels closed');
        }, 300);
      });
      
      tutorialService.onSkip(() => {
        // Add a slight delay before closing panels to ensure they're properly detected
        setTimeout(() => {
          this.closeAnyOpenPanels();
          this.tutorialVisible = false;
          logService.info('Tutorial skipped and panels closed');
        }, 300);
      });
      
      tutorialService.start()
      this.tutorialVisible = true
    },
    
    closeAnyOpenPanels() {
      try {
        // Close the navigation drawer if it's open
        const navDrawer = document.querySelector('.v-navigation-drawer');
        if (navDrawer) {
          // Find any button inside and click it
          const buttons = navDrawer.querySelectorAll('.v-btn');
          for (const btn of buttons) {
            if (btn.offsetParent !== null) { // Check if button is visible
              btn.click();
              logService.debug('Clicked button to close collection panel');
              break;
            }
          }
        }
        
        // Close advanced options panel if it's open
        // First check if there's any button with aria-expanded="true"
        const expandedButtons = document.querySelectorAll('.v-btn[aria-expanded="true"]');
        for (const btn of expandedButtons) {
          if (btn.offsetParent !== null) { // Check if button is visible
            btn.click();
            logService.debug('Closed expanded panel');
          }
        }
        
        // Alternative approach for advanced options
        const tuneIcon = document.querySelector('.v-icon:contains("mdi-tune"), .v-icon:contains("mdi-cog")');
        if (tuneIcon && tuneIcon.closest('.v-btn')) {
          const advancedPanel = document.querySelector('#advanced-options-panel');
          if (advancedPanel && advancedPanel.offsetParent !== null) {
            tuneIcon.closest('.v-btn').click();
            logService.debug('Closed advanced options using icon button');
          }
        }
      } catch (err) {
        logService.error('Error closing panels', err);
      }
    },
    
    initializeTutorialSteps() {
      const steps = [
        {
          element: '.search-container',  // For the first step, target the search container
          titleKey: 'tutorial.step1.title',
          contentKey: 'tutorial.step1.content',
          position: 'bottom'
        },
        {
          element: '.search-input',  // Target the search input field (more general selector)
          titleKey: 'tutorial.step2.title',
          contentKey: 'tutorial.step2.content',
          position: 'bottom'
        },
        {
          element: '.v-icon:contains("mdi-magnify")',  // Search magnify icon
          titleKey: 'tutorial.step3.title',
          contentKey: 'tutorial.step3.content',
          position: 'bottom'
        },
        {
          element: '.conversation-container',  // Results area
          titleKey: 'tutorial.step4.title',
          contentKey: 'tutorial.step4.content',
          position: 'top'
        },
        {
          element: '.v-icon:contains("mdi-tune"), .v-icon:contains("mdi-cog")',  // Advanced options icon
          titleKey: 'tutorial.step5.title',
          contentKey: 'tutorial.step5.content',
          position: 'bottom',
          preAction: () => {
            // Make sure advanced options panel is closed first
            const expanded = document.querySelector('.v-btn[aria-expanded="true"]');
            if (expanded) {
              expanded.click(); // Close it first
              setTimeout(() => {}, 100); // Give it a moment
            }
          }
        },
        {
          element: '.collection-fab-position, .collection-fab',  // Collection button (more general)
          titleKey: 'tutorial.step6.title',
          contentKey: 'tutorial.step6.content',
          position: 'left',
          preAction: () => {
            // Force close any open panels first
            this.closeAnyOpenPanels();
          }
        },
        {
          element: '.v-navigation-drawer',  // Collection panel
          titleKey: 'tutorial.step7.title',
          contentKey: 'tutorial.step7.content',
          position: 'left',
          preAction: () => {
            // Close any open panels first
            this.closeAnyOpenPanels();
            // Then open collection panel
            setTimeout(() => {
              const collectionButton = document.querySelector('.collection-fab, .v-btn:has(.v-badge)');
              if (collectionButton) {
                collectionButton.click();
                logService.debug('Opened collection panel for tutorial');
              }
            }, 200);
          }
        },
        {
          element: '.v-footer',  // Footer area
          titleKey: 'tutorial.step8.title',
          contentKey: 'tutorial.step8.content',
          position: 'top',
          preAction: () => {
            // Make sure to close any open panels before showing the footer
            this.closeAnyOpenPanels();
          }
        }
      ]
      
      tutorialService.initializeSteps(steps)
      logService.debug('Tutorial steps initialized', { stepCount: steps.length })
    }
  }
}
</script>

<style>
/* Global styles */
</style>
