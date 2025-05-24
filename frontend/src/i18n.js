import { createI18n } from 'vue-i18n'

// Function to load locale messages dynamically
function loadLocaleMessages() {
  const locales = import.meta.glob('./locales/*.json', { eager: true })
  const messages = {}
  for (const path in locales) {
    const matched = path.match(/([A-Za-z0-9-_]+)\.json$/i)
    if (matched && matched.length > 1) {
      const locale = matched[1]
      // Handle cases where JSON might be directly exported or under a 'default' key
      messages[locale] = locales[path].default || locales[path]
    }
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
