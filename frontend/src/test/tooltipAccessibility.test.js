import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const tooltipFiles = [
  'src/App.vue',
  'src/components/AdvancedOptionsPanel.vue',
  'src/components/AggregatedTermsView.vue',
  'src/components/ConversationSettings.vue',
  'src/components/PhenotypeCollectionPanel.vue',
  'src/components/QueryInterface.vue',
  'src/components/ResultItem.vue',
  'src/components/SimilarityScore.vue',
];

describe('tooltip accessibility markup', () => {
  it('names every audited tooltip on its role="tooltip" root', () => {
    for (const relativePath of tooltipFiles) {
      const source = readFileSync(resolve(import.meta.dirname, '../../', relativePath), 'utf8');
      const tooltipCount = (source.match(/<v-tooltip\b/g) || []).length;
      // Vuetify puts role="tooltip" on the overlay root; content-props would name the inner div
      const labelCount = (source.match(/<v-tooltip\b[^>]*?\s:?aria-label=/gs) || []).length;

      expect(labelCount, `${relativePath} should give each v-tooltip an aria-label`).toBe(
        tooltipCount
      );
      expect(source).not.toMatch(/content-props="\{\s*'aria-label'/);
    }
  });

  it('does not use the tooltip directive in App footer icons', () => {
    const appSource = readFileSync(resolve(import.meta.dirname, '../../src/App.vue'), 'utf8');

    expect(appSource).not.toContain(' v-tooltip=');
  });
});
