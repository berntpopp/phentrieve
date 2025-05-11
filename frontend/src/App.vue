<template>
  <v-app>
    <v-app-bar app color="primary" dark>
      <v-app-bar-title>Phentrieve HPO</v-app-bar-title>
      <v-spacer></v-spacer>
      <v-btn icon href="https://github.com/berntpopp/rag-hpo-testing" target="_blank">
        <v-icon>mdi-github</v-icon>
      </v-btn>
    </v-app-bar>

    <v-main>
      <HomeView />
    </v-main>

    <v-footer app>
      <span>&copy; {{ new Date().getFullYear() }} Phentrieve</span>
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
