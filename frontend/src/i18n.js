import { createI18n } from 'vue-i18n'

// Import locale files explicitly to ensure they're included in the production build
import en from './locales/en.json'
import fr from './locales/fr.json'
import es from './locales/es.json'
import de from './locales/de.json'
import nl from './locales/nl.json'

// Create messages object directly with imported locales
const messages = {
  en,
  fr,
  es,
  de,
  nl
}

// Determine initial locale: localStorage > browser language > default ('en')
let initialLocale = 'en'; // Default
try {
  const savedLang = localStorage.getItem('phentrieve-lang');
  if (savedLang && messages[savedLang]) {
    initialLocale = savedLang;
  } else {
    const browserLang = navigator.language.split('-')[0];
    if (messages[browserLang]) {
      initialLocale = browserLang;
    }
  }
} catch (e) {
  console.warn('Could not access localStorage for language preference. Defaulting to English.');
}

const i18n = createI18n({
  legacy: false, // Use Composition API features by default
  locale: initialLocale,
  fallbackLocale: 'en',
  messages: messages,
  
  // Critical: Ensure $t is available in Options API components
  globalInjection: true,
  
  // Use runtime-only mode to avoid eval() usage for better CSP compatibility
  runtimeOnly: true,
  // Pre-compile all messages to avoid dynamic compilation
  compileMode: 'hoist',
  
  // Other configuration options
  warnHtmlMessage: false, // Allow HTML in messages (e.g., for FAQ)
  missingWarn: import.meta.env.MODE !== 'production', // Show missing warnings only in dev
  fallbackWarn: import.meta.env.MODE !== 'production', // Show fallback warnings only in dev
  escapeParameter: true, // Escape HTML in parameters for security
  // Silent option to avoid console errors
  silentTranslationWarn: true,
  silentFallbackWarn: true
})

export default i18n
