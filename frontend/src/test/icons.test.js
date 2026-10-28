import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { icons } from '../plugins/icons';

const srcDir = resolve(import.meta.dirname, '..');

function sourceFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === 'test' ? [] : sourceFiles(path);
    return /\.(vue|js|json)$/.test(entry.name) && entry.name !== 'icons.js' ? [path] : [];
  });
}

describe('SVG icon map', () => {
  it('has an SVG path for every mdi-* icon name used in the app', () => {
    const used = new Set(
      sourceFiles(srcDir).flatMap(
        (file) => readFileSync(file, 'utf8').match(/mdi-[a-z0-9-]+/g) || []
      )
    );
    // 'mdi-svg' is Vuetify's iconset module path, not an icon
    const missing = [...used].filter((name) => name !== 'mdi-svg' && !(name in icons));

    expect(used.size).toBeGreaterThan(0);
    expect(missing).toEqual([]);
  });
});
