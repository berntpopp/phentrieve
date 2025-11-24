<template>
  <v-app theme="light" class="app-container">
    <v-main class="bg-grey-lighten-4 main-container">
      <router-view v-slot="{ Component }">
        <component :is="Component" />
      </router-view>
      <LogViewer />
    </v-main>

    <v-footer app class="d-flex justify-space-between pa-2" style="z-index: 1" role="contentinfo">
      <div class="d-flex align-center">
        <!-- Conversation Settings Button -->
        <ConversationSettings class="mr-1" />

        <!-- Home/Reset Button -->
        <v-tooltip
          location="top"
          :text="$t('app.footer.homeTooltip', 'Go to home / Reset conversation')"
          role="tooltip"
        >
          <template #activator="{ props }">
            <v-btn
              v-bind="props"
              icon
              variant="text"
              color="primary"
              size="small"
              class="mr-1"
              aria-label="Go to home"
              :to="{ name: 'home' }"
            >
              <v-badge
                v-if="conversationStore.hasConversation"
                color="info"
                dot
                offset-x="-2"
                offset-y="-2"
              >
                <v-icon>mdi-home</v-icon>
              </v-badge>
              <v-icon v-else>mdi-home</v-icon>
            </v-btn>
          </template>
        </v-tooltip>

        <!-- Disclaimer Button -->
        <v-tooltip
          location="top"
          :text="$t('app.footer.disclaimerTooltip')"
          role="tooltip"
          aria-label="Disclaimer information"
        >
          <template #activator="{ props }">
            <div class="d-flex align-center">
              <v-btn
                v-bind="props"
                icon
                variant="text"
                color="primary"
                size="small"
                class="mr-1"
                aria-label="View disclaimer information"
                @click="showDisclaimerDialog"
              >
                <v-icon>mdi-scale-balance</v-icon>
              </v-btn>
              <v-icon
                v-if="disclaimerStore.isAcknowledged"
                v-tooltip="
                  $t('app.footer.disclaimerAcknowledgedTooltip', {
                    date: disclaimerStore.formattedAcknowledgmentDate,
                  })
                "
                size="small"
                color="success"
                class="mr-2"
                aria-label="Disclaimer acknowledged"
              >
                mdi-check-circle
              </v-icon>
            </div>
          </template>
        </v-tooltip>
      </div>

      <div class="d-flex align-center">
        <LanguageSwitcher class="mr-2" />
        <v-tooltip
          location="top"
          :text="$t('app.footer.tutorialTooltip')"
          role="tooltip"
          aria-label="Tutorial information"
        >
          <template #activator="{ props }">
            <v-btn
              v-bind="props"
              icon
              variant="text"
              color="primary"
              size="small"
              class="mr-2"
              aria-label="Start guided tutorial"
              @click="startTutorial"
            >
              <v-icon>mdi-hand-pointing-up</v-icon>
            </v-btn>
          </template>
        </v-tooltip>
        <v-tooltip
          location="top"
          :text="$t('app.footer.faqTooltip')"
          role="tooltip"
          aria-label="FAQ information"
        >
          <template #activator="{ props }">
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
        <v-tooltip
          location="top"
          :text="$t('app.footer.logsTooltip')"
          role="tooltip"
          aria-label="Log viewer information"
        >
          <template #activator="{ props }">
            <v-btn
              v-bind="props"
              icon
              variant="text"
              color="primary"
              size="small"
              class="mr-2"
              aria-label="Toggle log viewer"
              @click="logStore.toggleViewer"
            >
              <v-icon>mdi-text-box-search-outline</v-icon>
            </v-btn>
          </template>
        </v-tooltip>
        <v-tooltip
          location="top"
          :text="$t('app.footer.githubTooltip')"
          role="tooltip"
          aria-label="GitHub repository link"
        >
          <template #activator="{ props }">
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
            />
          </template>
        </v-tooltip>
        <v-tooltip
          location="top"
          :text="$t('app.footer.docsTooltip')"
          role="tooltip"
          aria-label="Documentation link"
        >
          <template #activator="{ props }">
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
            />
          </template>
        </v-tooltip>
        <v-tooltip
          location="top"
          :text="$t('app.footer.licenseTooltip')"
          role="tooltip"
          aria-label="License link"
        >
          <template #activator="{ props }">
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
              class="mr-1"
            />
          </template>
        </v-tooltip>
        <v-tooltip
          location="top"
          text="Version & API Status"
          role="tooltip"
          aria-label="Version and connection information"
        >
          <template #activator="{ props }">
            <v-btn
              v-bind="props"
              icon
              variant="text"
              size="small"
              :color="apiConnected ? 'primary' : 'error'"
              aria-label="View version and connection status"
              @click="showVersionDialog = true"
            >
              <v-badge :color="apiConnected ? 'success' : 'error'" dot offset-x="-2" offset-y="-2">
                <v-icon>mdi-information-outline</v-icon>
              </v-badge>
            </v-btn>
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

    <!-- Version & Connection Status Dialog -->
    <v-dialog v-model="showVersionDialog" max-width="400">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon class="mr-2">mdi-package-variant</v-icon>
          Version Information
          <v-spacer />
          <v-btn
            icon="mdi-refresh"
            variant="text"
            size="small"
            :loading="loadingVersions"
            @click="refreshVersions"
          />
        </v-card-title>

        <v-divider />

        <v-card-text>
          <!-- Connection Status -->
          <div class="mb-3">
            <v-chip :color="apiConnected ? 'success' : 'error'" size="small" label>
              <v-icon size="small" start>
                {{ apiConnected ? 'mdi-lan-connect' : 'mdi-lan-disconnect' }}
              </v-icon>
              API {{ apiConnected ? 'Online' : 'Offline' }}
              <span v-if="apiConnected && responseTime" class="ml-1"> ({{ responseTime }}ms) </span>
            </v-chip>
          </div>

          <!-- Frontend Version -->
          <v-list-item class="px-0">
            <template #prepend>
              <v-icon color="success">mdi-vuejs</v-icon>
            </template>
            <v-list-item-title>Frontend</v-list-item-title>
            <v-list-item-subtitle>{{ frontendVersion }} (Vue.js)</v-list-item-subtitle>
          </v-list-item>

          <!-- API Version -->
          <v-list-item class="px-0">
            <template #prepend>
              <v-icon color="primary">mdi-api</v-icon>
            </template>
            <v-list-item-title>API</v-list-item-title>
            <v-list-item-subtitle>{{ apiVersion }} (FastAPI)</v-list-item-subtitle>
          </v-list-item>

          <!-- CLI Version -->
          <v-list-item class="px-0">
            <template #prepend>
              <v-icon color="info">mdi-console</v-icon>
            </template>
            <v-list-item-title>CLI</v-list-item-title>
            <v-list-item-subtitle>{{ cliVersion }} (Python)</v-list-item-subtitle>
          </v-list-item>

          <v-divider class="my-2" />

          <!-- Environment -->
          <div class="d-flex align-center justify-space-between">
            <span class="text-caption">Environment:</span>
            <v-chip :color="getEnvironmentColor(environment)" size="small" label>
              {{ environment }}
            </v-chip>
          </div>
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn text @click="showVersionDialog = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-app>
</template>

<script>
import { useDisclaimerStore } from './stores/disclaimer';
import { useLogStore } from './stores/log';
import { useConversationStore } from './stores/conversation';
import { logService } from './services/logService';
import { tutorialService } from './services/tutorialService';
import { getAllVersions } from './utils/version';
import { useApiHealth } from './services/api-health';
import DisclaimerDialog from './components/DisclaimerDialog.vue';
import LogViewer from './components/LogViewer.vue';
import LanguageSwitcher from './components/LanguageSwitcher.vue';
import TutorialOverlay from './components/TutorialOverlay.vue';
import ConversationSettings from './components/ConversationSettings.vue';

export default {
  name: 'App',
  components: {
    DisclaimerDialog,
    LogViewer,
    LanguageSwitcher,
    TutorialOverlay,
    ConversationSettings,
  },
  data() {
    return {
      disclaimerDialogVisible: false,
      tutorialVisible: false,
      showVersionDialog: false,
      loadingVersions: false,
      frontendVersion: 'Loading...',
      apiVersion: 'Loading...',
      cliVersion: 'Loading...',
      environment: 'unknown',
    };
  },
  computed: {
    disclaimerStore() {
      return useDisclaimerStore();
    },
    logStore() {
      return useLogStore();
    },
    conversationStore() {
      return useConversationStore();
    },
    apiConnected() {
      const { connected } = useApiHealth();
      return connected.value;
    },
    responseTime() {
      const { responseTime } = useApiHealth();
      return responseTime.value;
    },
  },
  created() {
    logService.info('App component created');

    // Initialize the store
    this.disclaimerStore.initialize();
    logService.debug('Disclaimer store initialized', {
      isAcknowledged: this.disclaimerStore.isAcknowledged,
      timestamp: this.disclaimerStore.acknowledgmentTimestamp,
    });

    // Show the disclaimer dialog if it has not been acknowledged
    if (!this.disclaimerStore.isAcknowledged) {
      this.disclaimerDialogVisible = true;
      logService.info('Showing initial disclaimer dialog');
    }

    // Initialize tutorial steps
    this.initializeTutorialSteps();

    // Log application initialization
    logService.info('Application initialized');
  },
  mounted() {
    // Fetch version information on mount
    this.refreshVersions();

    // Start API health monitoring
    const { startMonitoring } = useApiHealth();
    startMonitoring();
    logService.info('API health monitoring started');
  },
  beforeUnmount() {
    // Stop API health monitoring to prevent memory leaks
    const { stopMonitoring } = useApiHealth();
    stopMonitoring();
    logService.info('API health monitoring stopped');
  },
  methods: {
    async refreshVersions() {
      this.loadingVersions = true;
      try {
        const versions = await getAllVersions();
        this.frontendVersion = versions.frontend?.version || 'unknown';
        this.apiVersion = versions.api?.version || 'unknown';
        this.cliVersion = versions.cli?.version || 'unknown';
        this.environment = versions.environment || 'unknown';
        logService.debug('Versions refreshed', versions);
      } catch (error) {
        logService.error('Failed to refresh versions', error);
      } finally {
        this.loadingVersions = false;
      }
    },
    getEnvironmentColor(env) {
      const colors = {
        production: 'success',
        staging: 'warning',
        development: 'info',
      };
      return colors[env] || 'default';
    },
    showDisclaimerDialog() {
      logService.debug('Manual disclaimer dialog trigger');
      this.disclaimerDialogVisible = true;
    },
    handleDisclaimerAcknowledged() {
      logService.info('User acknowledged disclaimer');
      // Save the acknowledgment to the store
      this.disclaimerStore.saveAcknowledgment();
    },
    startTutorial() {
      logService.info('Starting tutorial');

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

      tutorialService.start();
      this.tutorialVisible = true;
    },

    closeAnyOpenPanels() {
      try {
        // Close the navigation drawer if it's open
        const navDrawer = document.querySelector('.v-navigation-drawer');
        if (navDrawer) {
          // Find any button inside and click it
          const buttons = navDrawer.querySelectorAll('.v-btn');
          for (const btn of buttons) {
            if (btn.offsetParent !== null) {
              // Check if button is visible
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
          if (btn.offsetParent !== null) {
            // Check if button is visible
            btn.click();
            logService.debug('Closed expanded panel');
          }
        }

        // Alternative approach for advanced options
        const tuneIcon = document.querySelector(
          '.v-icon:contains("mdi-tune"), .v-icon:contains("mdi-cog")'
        );
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
          element: '.search-container', // For the first step, target the search container
          titleKey: 'tutorial.step1.title',
          contentKey: 'tutorial.step1.content',
          position: 'bottom',
        },
        {
          element: '.search-input', // Target the search input field (more general selector)
          titleKey: 'tutorial.step2.title',
          contentKey: 'tutorial.step2.content',
          position: 'bottom',
        },
        {
          element: '.v-icon:contains("mdi-magnify")', // Search magnify icon
          titleKey: 'tutorial.step3.title',
          contentKey: 'tutorial.step3.content',
          position: 'bottom',
        },
        {
          element: '.conversation-container', // Results area
          titleKey: 'tutorial.step4.title',
          contentKey: 'tutorial.step4.content',
          position: 'top',
        },
        {
          element: '.v-icon:contains("mdi-tune"), .v-icon:contains("mdi-cog")', // Advanced options icon
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
          },
        },
        {
          element: '.collection-fab-position, .collection-fab', // Collection button (more general)
          titleKey: 'tutorial.step6.title',
          contentKey: 'tutorial.step6.content',
          position: 'left',
          preAction: () => {
            // Force close any open panels first
            this.closeAnyOpenPanels();
          },
        },
        {
          element: '.v-navigation-drawer', // Collection panel
          titleKey: 'tutorial.step7.title',
          contentKey: 'tutorial.step7.content',
          position: 'left',
          preAction: () => {
            // Close any open panels first
            this.closeAnyOpenPanels();
            // Then open collection panel
            setTimeout(() => {
              const collectionButton = document.querySelector(
                '.collection-fab, .v-btn:has(.v-badge)'
              );
              if (collectionButton) {
                collectionButton.click();
                logService.debug('Opened collection panel for tutorial');
              }
            }, 200);
          },
        },
        {
          element: '.v-footer', // Footer area
          titleKey: 'tutorial.step8.title',
          contentKey: 'tutorial.step8.content',
          position: 'top',
          preAction: () => {
            // Make sure to close any open panels before showing the footer
            this.closeAnyOpenPanels();
          },
        },
      ];

      tutorialService.initializeSteps(steps);
      logService.debug('Tutorial steps initialized', { stepCount: steps.length });
    },
  },
};
</script>

<style>
/* Global styles */
html,
body {
  height: 100%;
  margin: 0;
  padding: 0;
}

.app-container {
  min-height: 100vh;
}

.main-container {
  overflow: auto;
  display: flex;
  flex-direction: column;
  flex-grow: 1;
}
</style>

<style>
/* Global styles */
</style>
