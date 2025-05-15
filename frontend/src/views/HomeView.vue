<template>
  <div class="fill-height">
    <div class="content-wrapper">
      <div class="d-flex justify-center align-center pb-2">
        <img src="/hpo-logo.svg" alt="HPO Logo" height="50" class="mr-2">
        <h1 class="text-h5 font-weight-light">Phentrieve</h1>
      </div>
      <div class="text-center px-4 pb-4" style="max-width: 600px;">
        <p class="text-body-2 text-medium-emphasis">
          Phentrieve is an AI-powered tool that accurately maps clinical text to standardized Human Phenotype Ontology (HPO) terms, streamlining phenotype annotation for research and diagnostics.
        </p>
      </div>
      <QueryInterface />
    </div>
  </div>
</template>

<script>
import QueryInterface from '@/components/QueryInterface.vue'

export default {
  name: 'HomeView',
  components: {
    QueryInterface
  }
}
</script>

<style scoped>
.content-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  position: absolute;
  top: 25%; /* Position exactly at 1/4 from the top */
  left: 0;
  right: 0;
}

.fill-height {
  min-height: 100vh;
  position: relative;
}
</style>
