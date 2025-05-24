import { logService } from './logService'

class TutorialService {
  constructor() {
    this.currentStep = 0
    this.isActive = false
    this.steps = []
    this.callbacks = {
      onStepChange: null,
      onComplete: null,
      onSkip: null
    }
  }

  // Initialize tutorial steps
  initializeSteps(steps) {
    logService.debug('Initializing tutorial steps', { stepCount: steps.length })
    this.steps = steps
    this.currentStep = 0
  }

  // Start the tutorial
  start() {
    if (this.steps.length === 0) {
      logService.warn('Cannot start tutorial: no steps defined')
      return false
    }

    logService.info('Starting tutorial')
    this.isActive = true
    this.currentStep = 0
    this.showStep(0)
    return true
  }

  // Show a specific step
  showStep(stepIndex) {
    if (stepIndex < 0 || stepIndex >= this.steps.length) {
      logService.warn('Invalid step index', { stepIndex, totalSteps: this.steps.length })
      return false
    }

    const step = this.steps[stepIndex]
    logService.debug('Showing tutorial step', { stepIndex, step: step.titleKey })

    // Execute pre-action if defined
    if (step.preAction && typeof step.preAction === 'function') {
      try {
        step.preAction()
      } catch (error) {
        logService.error('Error executing step pre-action', { error: error.message, stepIndex })
      }
    }

    this.currentStep = stepIndex

    // Notify listeners
    if (this.callbacks.onStepChange) {
      this.callbacks.onStepChange(step, stepIndex)
    }

    return true
  }

  // Go to next step
  next() {
    if (this.currentStep < this.steps.length - 1) {
      this.showStep(this.currentStep + 1)
      return true
    } else {
      this.complete()
      return false
    }
  }

  // Go to previous step
  previous() {
    if (this.currentStep > 0) {
      this.showStep(this.currentStep - 1)
      return true
    }
    return false
  }

  // Skip tutorial
  skip() {
    logService.info('Tutorial skipped')
    this.isActive = false
    if (this.callbacks.onSkip) {
      this.callbacks.onSkip()
    }
  }

  // Complete tutorial
  complete() {
    logService.info('Tutorial completed')
    this.isActive = false
    if (this.callbacks.onComplete) {
      this.callbacks.onComplete()
    }
  }

  // Register callbacks
  onStepChange(callback) {
    this.callbacks.onStepChange = callback
  }

  onComplete(callback) {
    this.callbacks.onComplete = callback
  }

  onSkip(callback) {
    this.callbacks.onSkip = callback
  }

  // Get current step info
  getCurrentStep() {
    return this.steps[this.currentStep] || null
  }

  getCurrentStepIndex() {
    return this.currentStep
  }

  getTotalSteps() {
    return this.steps.length
  }

  isLastStep() {
    return this.currentStep === this.steps.length - 1
  }

  isFirstStep() {
    return this.currentStep === 0
  }

  // Check if tutorial is active
  isRunning() {
    return this.isActive
  }
}

// Export singleton instance
export const tutorialService = new TutorialService()
