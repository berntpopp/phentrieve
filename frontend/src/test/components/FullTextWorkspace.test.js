import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import FullTextWorkspace from '../../components/query/FullTextWorkspace.vue';

const vuetify = createVuetify({ components, directives });

const baseSegments = [
  { key: 'plain-0', text: 'Patient had ', highlighted: false },
  {
    key: 'mark-0',
    text: 'seizures',
    highlighted: true,
    termIds: ['HP:0001250'],
    tooltip: 'Seizure (HP:0001250)',
  },
  { key: 'plain-1', text: ' and ', highlighted: false },
  {
    key: 'mark-1',
    text: 'developmental delay',
    highlighted: true,
    termIds: ['HP:0001263'],
    tooltip: 'Developmental delay (HP:0001263)',
  },
];

function mountWorkspace(props = {}) {
  return mount(FullTextWorkspace, {
    props: {
      summary: 'Patient had seizures and developmental delay',
      meta: '6 words',
      expanded: true,
      segments: baseSegments,
      activePhenotypeId: null,
      ...props,
    },
    global: { plugins: [vuetify] },
  });
}

describe('FullTextWorkspace annotated note', () => {
  it('renders every highlighted segment as a mark (no hover collapse)', () => {
    const wrapper = mountWorkspace();
    expect(wrapper.findAll('[data-testid="annotated-note-span"]')).toHaveLength(2);
  });

  it('exposes each annotated span to keyboard and assistive technology', () => {
    const wrapper = mountWorkspace();
    const marks = wrapper.findAll('[data-testid="annotated-note-span"]');
    marks.forEach((mark) => {
      expect(mark.attributes('tabindex')).toBe('0');
      expect(mark.attributes('aria-label')).toBeTruthy();
    });
    expect(marks[0].attributes('aria-label')).toContain('Seizure (HP:0001250)');
  });

  it('keeps all spans highlighted while emphasising only the active phenotype', () => {
    const wrapper = mountWorkspace({ activePhenotypeId: 'HP:0001250' });
    const marks = wrapper.findAll('[data-testid="annotated-note-span"]');
    expect(marks).toHaveLength(2);
    expect(marks[0].classes()).toContain('annotated-note-span--active');
    expect(marks[1].classes()).not.toContain('annotated-note-span--active');
  });

  it('gives the expand/collapse toggle an accessible name', () => {
    const collapsed = mountWorkspace({ expanded: false });
    expect(collapsed.get('[data-testid="user-note-summary-toggle"]').attributes('aria-label')).toBe(
      'Expand clinical note'
    );

    const expanded = mountWorkspace({ expanded: true });
    expect(expanded.get('[data-testid="user-note-summary-toggle"]').attributes('aria-label')).toBe(
      'Collapse clinical note'
    );
  });

  it('emits hover/clear for keyboard focus parity with mouse', async () => {
    const wrapper = mountWorkspace();
    const firstMark = wrapper.findAll('[data-testid="annotated-note-span"]')[0];

    await firstMark.trigger('focus');
    expect(wrapper.emitted('hover')?.[0]?.[0]).toEqual(['HP:0001250']);

    await firstMark.trigger('blur');
    expect(wrapper.emitted('clear-hover')).toBeTruthy();
  });
});
