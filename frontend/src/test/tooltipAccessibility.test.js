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
  it('keeps audited tooltip components paired with content-props labels', () => {
    for (const relativePath of tooltipFiles) {
      const source = readFileSync(resolve(import.meta.dirname, '../../', relativePath), 'utf8');
      const tooltipCount = (source.match(/<v-tooltip\b/g) || []).length;
      const contentPropsCount = (source.match(/content-props=/g) || []).length;

      expect(
        contentPropsCount,
        `${relativePath} should provide content-props for each tooltip`
      ).toBe(tooltipCount);
    }
  });

  it('does not use the tooltip directive in App footer icons', () => {
    const appSource = readFileSync(resolve(import.meta.dirname, '../../src/App.vue'), 'utf8');

    expect(appSource).not.toContain(' v-tooltip=');
  });
});
