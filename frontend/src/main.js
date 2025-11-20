import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import i18n from './i18n';

// Import critical CSS first to optimize rendering
import './critical.css';

// Pinia store
import { createPinia } from 'pinia';
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate';
import { useLogStore } from './stores/log';
import { logService } from './services/logService';

// Vuetify (tree-shaken via plugin - only used components imported)
import 'vuetify/styles';
import '@mdi/font/css/materialdesignicons.css';
import vuetify from './plugins/vuetify';

const app = createApp(App);
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate);
app.use(pinia);
app.use(vuetify);
app.use(router);
app.use(i18n);

// Initialize logService with store
const logStore = useLogStore(pinia);
logService.initStore(logStore);

app.mount('#app');
