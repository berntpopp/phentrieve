import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';

describe('AnnotatedDocumentPane', () => {
  it('renders chunk-only evidence with gutter tint and span evidence with marks', async () => {
    const component = (await import('../../components/AnnotatedDocumentPane.vue')).default;
    const wrapper = mount(component, {
      props: {
        chunks: [
          {
            chunk_id: 1,
            text: 'Patient had recurrent seizures.',
            status: 'affirmed',
            evidence_mode: 'chunk',
          },
          {
            chunk_id: 2,
            text: 'Developmental delay was present.',
            status: 'affirmed',
            evidence_mode: 'span',
            annotations: [
              { id: 'ann-1', start_char: 0, end_char: 19, matched_text_in_chunk: 'Developmental delay' },
            ],
          },
        ],
      },
    });

    expect(wrapper.find('[data-chunk-evidence-mode="chunk"]').exists()).toBe(true);
    expect(wrapper.find('mark[data-annotation-id="ann-1"]').exists()).toBe(true);
  });
});
