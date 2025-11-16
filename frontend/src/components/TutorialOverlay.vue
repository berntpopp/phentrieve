<template>
  <div v-if="isVisible" class="tutorial-overlay">
    <!-- Backdrop -->
    <div class="tutorial-backdrop" @click="skipTutorial" />

    <!-- Highlight area -->
    <div v-if="currentStep && highlightBounds" class="tutorial-highlight" :style="highlightStyle" />

    <!-- Tutorial card -->
    <v-card v-if="currentStep" class="tutorial-card elevation-8" :style="cardStyle" max-width="400">
      <v-card-title class="d-flex align-center justify-space-between">
        <span class="text-h6">{{ $t(currentStep.titleKey) }}</span>
        <v-btn
          icon="mdi-close"
          variant="text"
          size="small"
          aria-label="Close tutorial"
          @click="skipTutorial"
        />
      </v-card-title>

      <v-card-text>
        <p class="text-body-1 mb-3">
          {{ $t(currentStep.contentKey) }}
        </p>

        <!-- Progress indicator -->
        <div class="d-flex align-center mb-3">
          <span class="text-caption text-medium-emphasis mr-2">
            {{ currentStepIndex + 1 }} / {{ totalSteps }}
          </span>
          <v-progress-linear :model-value="progress" color="primary" height="4" rounded />
        </div>
      </v-card-text>

      <v-card-actions class="justify-space-between">
        <v-btn variant="text" color="default" @click="skipTutorial">
          {{ $t('tutorial.navigation.skip') }}
        </v-btn>

        <div class="d-flex gap-2">
          <v-btn
            v-if="!isFirstStep"
            variant="outlined"
            prepend-icon="mdi-chevron-left"
            @click="previousStep"
          >
            {{ $t('tutorial.navigation.previous') }}
          </v-btn>

          <v-btn
            v-if="!isLastStep"
            variant="flat"
            color="primary"
            append-icon="mdi-chevron-right"
            @click="nextStep"
          >
            {{ $t('tutorial.navigation.next') }}
          </v-btn>

          <v-btn
            v-else
            variant="flat"
            color="success"
            prepend-icon="mdi-check"
            @click="finishTutorial"
          >
            {{ $t('tutorial.navigation.finish') }}
          </v-btn>
        </div>
      </v-card-actions>
    </v-card>
  </div>
</template>

<script>
import { tutorialService } from '../services/tutorialService';
import { logService } from '../services/logService';

export default {
  name: 'TutorialOverlay',
  props: {
    visible: {
      type: Boolean,
      default: false,
    },
  },
  emits: ['update:visible'],
  data() {
    return {
      isVisible: false,
      currentStep: null,
      currentStepIndex: 0,
      totalSteps: 0,
      highlightBounds: null,
      cardPosition: { top: 100, left: 100 },
    };
  },
  computed: {
    progress() {
      return this.totalSteps > 0 ? ((this.currentStepIndex + 1) / this.totalSteps) * 100 : 0;
    },
    isFirstStep() {
      return this.currentStepIndex === 0;
    },
    isLastStep() {
      return this.currentStepIndex === this.totalSteps - 1;
    },
    highlightStyle() {
      if (!this.highlightBounds) return {};
      return {
        position: 'fixed',
        top: `${this.highlightBounds.top}px`,
        left: `${this.highlightBounds.left}px`,
        width: `${this.highlightBounds.width}px`,
        height: `${this.highlightBounds.height}px`,
        border: '2px solid #1976D2',
        borderRadius: '4px',
        backgroundColor: 'rgba(25, 118, 210, 0.1)',
        boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.5)',
        zIndex: 2000,
      };
    },
    cardStyle() {
      return {
        position: 'fixed',
        top: `${this.cardPosition.top}px`,
        left: `${this.cardPosition.left}px`,
        zIndex: 2001,
      };
    },
  },
  watch: {
    visible(newVal) {
      this.isVisible = newVal;
      if (newVal) {
        this.updateTutorialState();
      }
    },
  },
  mounted() {
    // Set up tutorial service callbacks
    tutorialService.onStepChange((step, stepIndex) => {
      this.currentStep = step;
      this.currentStepIndex = stepIndex;
      this.totalSteps = tutorialService.getTotalSteps();
      this.updateHighlight();
    });

    tutorialService.onComplete(() => {
      this.hideTutorial();
    });

    tutorialService.onSkip(() => {
      this.hideTutorial();
    });

    // Listen for window resize to update positions
    window.addEventListener('resize', this.updateHighlight);
  },
  beforeUnmount() {
    window.removeEventListener('resize', this.updateHighlight);
  },
  methods: {
    updateTutorialState() {
      if (tutorialService.isRunning()) {
        this.currentStep = tutorialService.getCurrentStep();
        this.currentStepIndex = tutorialService.getCurrentStepIndex();
        this.totalSteps = tutorialService.getTotalSteps();
        this.updateHighlight();
      }
    },
    updateHighlight() {
      if (!this.currentStep || !this.currentStep.element) {
        this.highlightBounds = null;
        this.cardPosition = { top: 100, left: 100 };
        return;
      }

      // Find the target element
      const element = document.querySelector(this.currentStep.element);
      if (!element) {
        logService.warn('Tutorial target element not found', {
          selector: this.currentStep.element,
        });
        this.highlightBounds = null;
        this.cardPosition = { top: 100, left: 100 };
        return;
      }

      // Get element bounds
      const rect = element.getBoundingClientRect();
      this.highlightBounds = {
        top: rect.top - 4,
        left: rect.left - 4,
        width: rect.width + 8,
        height: rect.height + 8,
      };

      // Position tutorial card based on step position preference
      this.positionCard(rect);
    },
    positionCard(targetRect) {
      const cardWidth = 400;
      const cardHeight = 300; // Approximate height
      const margin = 20;
      const position = this.currentStep.position || 'bottom';

      let top, left;

      switch (position) {
        case 'top':
          top = targetRect.top - cardHeight - margin;
          left = targetRect.left + targetRect.width / 2 - cardWidth / 2;
          break;
        case 'bottom':
          top = targetRect.bottom + margin;
          left = targetRect.left + targetRect.width / 2 - cardWidth / 2;
          break;
        case 'left':
          top = targetRect.top + targetRect.height / 2 - cardHeight / 2;
          left = targetRect.left - cardWidth - margin;
          break;
        case 'right':
          top = targetRect.top + targetRect.height / 2 - cardHeight / 2;
          left = targetRect.right + margin;
          break;
        default:
          top = targetRect.bottom + margin;
          left = targetRect.left + targetRect.width / 2 - cardWidth / 2;
      }

      // Ensure card stays within viewport
      const viewport = {
        width: window.innerWidth,
        height: window.innerHeight,
      };

      top = Math.max(margin, Math.min(top, viewport.height - cardHeight - margin));
      left = Math.max(margin, Math.min(left, viewport.width - cardWidth - margin));

      this.cardPosition = { top, left };
    },
    nextStep() {
      tutorialService.next();
    },
    previousStep() {
      tutorialService.previous();
    },
    skipTutorial() {
      tutorialService.skip();
    },
    finishTutorial() {
      tutorialService.complete();
    },
    hideTutorial() {
      this.isVisible = false;
      this.$emit('update:visible', false);
    },
  },
};
</script>

<style scoped>
.tutorial-overlay {
  pointer-events: none;
}

.tutorial-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.5);
  z-index: 1999;
  pointer-events: auto;
}

.tutorial-highlight {
  pointer-events: none;
  transition: all 0.3s ease-in-out;
}

.tutorial-card {
  pointer-events: auto;
  transition: all 0.3s ease-in-out;
}
</style>
