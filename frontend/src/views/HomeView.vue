<template>
  <v-container fluid class="pa-0 ma-0 fill-height">
    <v-row no-gutters>
      <v-col cols="12">
        <div class="d-flex justify-center align-center pt-10 pb-2">
          <img src="/hpo-logo.svg" alt="HPO Logo" height="50" class="mr-2">
          <h1 class="text-h5 font-weight-light">Phentrieve</h1>
        </div>
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
