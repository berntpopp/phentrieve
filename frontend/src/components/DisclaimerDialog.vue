<template>
  <v-dialog
    v-model="dialogVisible"
    persistent
    max-width="600"
    class="disclaimer-dialog"
    role="alertdialog"
    aria-labelledby="disclaimer-title"
    aria-describedby="disclaimer-content"
  >
    <v-card>
      <v-card-title id="disclaimer-title" class="text-h5 d-flex align-center text-primary-darken-1">
        <v-icon color="warning" class="mr-2" aria-hidden="true">mdi-alert-circle</v-icon>
        Research Use Disclaimer
      </v-card-title>
      
      <v-card-text id="disclaimer-content" class="pb-0">
        <p class="mb-4 text-h6 text-high-emphasis">
          Phentrieve is intended for research and informational purposes only.
        </p>
        
        <p class="mb-3 text-body-1 text-high-emphasis">
          This tool is designed to map clinical text to standardized Human Phenotype Ontology (HPO) terms to aid in research and knowledge discovery. 
        </p>
        
        <h3 class="text-subtitle-1 font-weight-bold mb-3 text-primary-darken-1">
          Important Limitations:
        </h3>
        
        <ul class="mb-4 text-body-1 text-high-emphasis" role="list">
          <li class="mb-2">Phentrieve is <strong class="error--text">NOT</strong> a clinical diagnostic tool and should not replace professional medical advice, diagnosis, or treatment.</li>
          <li class="mb-2">The AI models and HPO mapping may contain inaccuracies, errors, or limitations.</li>
          <li class="mb-2">Results should be verified by qualified medical professionals before being used in any clinical context.</li>
          <li class="mb-2">The tool does not establish a doctor-patient relationship and is not a substitute for consulting with healthcare providers.</li>
        </ul>
        
        <h3 class="text-subtitle-1 font-weight-bold mb-3 text-primary-darken-1">
          User Responsibility:
        </h3>
        
        <p class="text-body-1 text-high-emphasis">
          By using Phentrieve, you acknowledge that you understand these limitations and that you assume full responsibility for how you interpret and use the information provided by this tool.
        </p>
      </v-card-text>
      
      <v-card-actions class="pt-2 pb-4 px-6">
        <v-spacer></v-spacer>
        <v-btn
          color="primary"
          variant="elevated"
          @click="acknowledgeDisclaimer"
          :loading="loading"
          :disabled="loading"
          aria-label="Accept disclaimer and continue to application"
        >
          I Understand and Agree
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script>
export default {
  name: 'DisclaimerDialog',
  props: {
    modelValue: {
      type: Boolean,
      default: false
    }
  },
  emits: ['update:modelValue', 'acknowledged'],
  data() {
    return {
      loading: false
    }
  },
  computed: {
    dialogVisible: {
      get() {
        return this.modelValue
      },
      set(value) {
        this.$emit('update:modelValue', value)
      }
    }
  },
  methods: {
    acknowledgeDisclaimer() {
      this.loading = true
      
      // Simulate a slight delay for better UX
      setTimeout(() => {
        this.loading = false
        this.$emit('acknowledged')
        this.dialogVisible = false
      }, 300)
    }
  }
}
</script>

<style scoped>
.disclaimer-dialog {
  z-index: 1000;
}
</style>
