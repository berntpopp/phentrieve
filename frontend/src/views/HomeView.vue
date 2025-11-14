<template>
  <div class="fill-height">
    <div class="content-wrapper">
      <div class="d-flex justify-center align-center pb-1">
        <img
          src="/favicon.svg"
          alt="Phentrieve Logo"
          width="65"
          height="65"
          class="mr-2"
          loading="lazy"
        />
        <h1 class="text-h5 font-weight-light logo-text">
          <span class="logo-visible">Phen</span><span class="logo-bracket">[</span
          ><span class="logo-hidden">otype re</span><span class="logo-bracket">]</span
          ><span class="logo-visible">trieve</span>
        </h1>
      </div>
      <div class="text-center px-4 py-0" style="max-width: 600px">
        <p class="text-body-2 text-medium-emphasis mb-1">
          {{ t('queryInterface.welcomeText') }}
        </p>
      </div>
      <QueryInterface />
    </div>
  </div>
</template>

<script>
import QueryInterface from '@/components/QueryInterface.vue';
import { useLogStore } from '@/stores/log';
import { useI18n } from 'vue-i18n';

export default {
  name: 'HomeView',
  components: {
    QueryInterface,
  },
  setup() {
    const logStore = useLogStore();
    const { t } = useI18n();
    return { logStore, t };
  },
};
</script>

<style scoped>
.content-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding-top: 10vh; /* Replace absolute positioning with padding */
}

.fill-height {
  min-height: 100vh;
  position: relative;
  display: flex;
  flex-direction: column;
}

.logo-text {
  position: relative;
  cursor: default;
  display: inline-flex;
  overflow: hidden;
  white-space: nowrap;
}

.logo-hidden {
  max-width: 0;
  opacity: 0;
  transition:
    max-width 0.5s ease,
    opacity 0.3s ease;
  display: inline-block;
  overflow: hidden;
  color: #1867c0;
}

.logo-visible {
  display: inline-block;
}

.logo-bracket {
  display: inline-block;
  max-width: 0;
  opacity: 0;
  overflow: hidden;
  transition:
    max-width 0.5s ease,
    opacity 0.3s ease;
  color: #42a5f5;
  font-weight: bold;
}

.logo-text:hover .logo-hidden,
.logo-text:hover .logo-bracket {
  max-width: 150px;
  opacity: 1;
}
</style>
