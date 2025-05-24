import { createI18n } from 'vue-i18n'

// Import locale files explicitly to ensure they're included in the production build
import en from './locales/en.json'
import fr from './locales/fr.json'
import es from './locales/es.json'
import de from './locales/de.json'
import nl from './locales/nl.json'

// Create messages object with imported locales
function loadLocaleMessages() {
  const messages = {
    en,
    fr,
    es,
    de,
    nl
  }
  return messages
}

const messages = loadLocaleMessages()

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

export default createI18n({
  legacy: false, // Important for Composition API
  locale: initialLocale,
  fallbackLocale: 'en',
  messages: messages,
  warnHtmlMessage: false, // Allow HTML in messages (e.g., for FAQ)
  missingWarn: process.env.NODE_ENV !== 'production', // Show missing warnings only in dev
  fallbackWarn: process.env.NODE_ENV !== 'production', // Show fallback warnings only in dev
})
