import { usedIcons } from './src/plugins/icons';

export default function iconOptimizer() {
  return {
    name: 'icon-optimizer',
    transform(code, id) {
      // Only process Material Design Icons CSS
      if (id.includes('materialdesignicons') && id.endsWith('.css')) {
        // Keep only the icons we use
        const iconPattern = /\.mdi-[^:{\s]+:before\s*{[^}]+}/g;

        const optimizedCode = code.split('\n').filter(line => {
          if (line.match(iconPattern)) {
            return usedIcons.some(icon => line.includes(icon.replace('mdi-', '')));
          }
          return true;
        }).join('\n');
        
        return {
          code: optimizedCode,
          map: null
        };
      }
    }
  };
}
