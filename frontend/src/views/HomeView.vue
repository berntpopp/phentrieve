<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <h1 class="text-h4 mb-4">Phentrieve</h1>
        <p class="mb-6">Map clinical text to HPO terms using advanced embedding models</p>
      </v-col>
    </v-row>
    
    <v-row>
      <v-col cols="12">
        <QueryInterface />
      </v-col>
    </v-row>
  </v-container>
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
