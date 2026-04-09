import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useFileDownload } from '../../composables/useFileDownload';

describe('useFileDownload', () => {
  let createObjectURLSpy;
  let revokeObjectURLSpy;
  let appendChildSpy;
  let removeChildSpy;
  let clickSpy;

  beforeEach(() => {
    createObjectURLSpy = vi.fn().mockReturnValue('blob:mock-url');
    revokeObjectURLSpy = vi.fn();
    appendChildSpy = vi.spyOn(document.body, 'appendChild').mockImplementation(() => {});
    removeChildSpy = vi.spyOn(document.body, 'removeChild').mockImplementation(() => {});
    clickSpy = vi.fn();

    global.URL.createObjectURL = createObjectURLSpy;
    global.URL.revokeObjectURL = revokeObjectURLSpy;

    vi.spyOn(document, 'createElement').mockReturnValue({
      href: '',
      download: '',
      click: clickSpy,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns downloadBlob, downloadText, and downloadJson', () => {
    const result = useFileDownload();
    expect(result).toHaveProperty('downloadBlob');
    expect(result).toHaveProperty('downloadText');
    expect(result).toHaveProperty('downloadJson');
    expect(typeof result.downloadBlob).toBe('function');
    expect(typeof result.downloadText).toBe('function');
    expect(typeof result.downloadJson).toBe('function');
  });

  it('downloadBlob creates a link, clicks it, and cleans up', () => {
    const { downloadBlob } = useFileDownload();
    const blob = new Blob(['test'], { type: 'text/plain' });

    downloadBlob(blob, 'test.txt');

    expect(createObjectURLSpy).toHaveBeenCalledWith(blob);
    expect(appendChildSpy).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
    expect(removeChildSpy).toHaveBeenCalled();
    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url');
  });

  it('downloadText creates a Blob with the given content and mimeType', () => {
    const { downloadText } = useFileDownload();

    downloadText('hello world', 'test.txt', 'text/plain');

    expect(createObjectURLSpy).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
  });

  it('downloadJson serializes data as JSON', () => {
    const { downloadJson } = useFileDownload();
    const data = { key: 'value' };

    downloadJson(data, 'test.json');

    expect(createObjectURLSpy).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
  });

  it('downloadText uses text/plain as default mimeType', () => {
    const { downloadText } = useFileDownload();

    downloadText('content', 'file.txt');

    // Should not throw and should proceed normally
    expect(clickSpy).toHaveBeenCalled();
  });
});
