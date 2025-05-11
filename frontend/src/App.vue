<template>
  <v-app theme="light">
    <v-main class="bg-grey-lighten-4">
      <HomeView />
    </v-main>
    
    <v-footer app class="d-flex justify-end pa-2" style="z-index: 1;">
      <div class="text-body-2 text-grey mr-2">&copy; {{ new Date().getFullYear() }} Phentrieve</div>
      <v-btn icon="mdi-github" size="small" href="https://github.com/berntpopp/rag-hpo-testing" target="_blank" variant="text" color="grey"></v-btn>
    </v-footer>
  </v-app>
</template>

<script>
import HomeView from './views/HomeView.vue'

export default {
  name: 'App',
  components: {
    HomeView
  }
}
</script>

<style>
/* Global styles */
</style>
