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
        {{ $t('disclaimerDialog.title') }}
      </v-card-title>
      
      <v-card-text id="disclaimer-content" class="pb-0">
        <p class="mb-4 text-h6 text-high-emphasis">
          {{ $t('disclaimerDialog.mainHeader') }}
        </p>
        
        <p class="mb-3 text-body-1 text-high-emphasis">
          {{ $t('disclaimerDialog.description') }} 
        </p>
        
        <h3 class="text-subtitle-1 font-weight-bold mb-3 text-primary-darken-1">
          {{ $t('disclaimerDialog.limitationsTitle') }}
        </h3>
        
        <ul class="mb-4 text-body-1 text-high-emphasis" role="list">
          <li class="mb-2" v-html="$t('disclaimerDialog.limitations.item1')"></li>
          <li class="mb-2">{{ $t('disclaimerDialog.limitations.item2') }}</li>
          <li class="mb-2">{{ $t('disclaimerDialog.limitations.item3') }}</li>
          <li class="mb-2">{{ $t('disclaimerDialog.limitations.item4') }}</li>
        </ul>
        
        <h3 class="text-subtitle-1 font-weight-bold mb-3 text-primary-darken-1">
          {{ $t('disclaimerDialog.responsibilityTitle') }}
        </h3>
        
        <p class="text-body-1 text-high-emphasis">
          {{ $t('disclaimerDialog.responsibilityText') }}
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
          {{ $t('disclaimerDialog.acceptButton') }}
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
