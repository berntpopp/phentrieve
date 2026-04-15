import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('frontend index.html head tags', () => {
  it('does not preload the favicon asset', () => {
    const indexHtml = readFileSync(resolve(import.meta.dirname, '../../index.html'), 'utf8');

    expect(indexHtml).not.toContain('rel="preload" href="/favicon.svg"');
    expect(indexHtml).toContain('rel="icon" href="/favicon.svg"');
  });
});
