import { createApp } from 'vue'
import App from './App.vue'

// Pinia store
import { createPinia } from 'pinia'
import { useLogStore } from './stores/log'
import { logService } from './services/logService'

// Vuetify
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import { mdi } from 'vuetify/iconsets/mdi'
import '@mdi/font/css/materialdesignicons.css'

const vuetify = createVuetify({
  components,
  directives,
  icons: {
    defaultSet: 'mdi',
    sets: {
      mdi
    }
  },
  theme: {
    defaultTheme: 'light'
  }
})

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)
app.use(vuetify)

// Initialize logService with store
const logStore = useLogStore(pinia)
logService.initStore(logStore)

app.mount('#app')
