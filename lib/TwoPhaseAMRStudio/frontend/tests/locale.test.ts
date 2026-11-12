import { describe, expect, it } from 'vitest';
import { repairBrowserLocale, repairedLocale, usableLocale } from '../src/locale';

describe('browser locale repair', () => {
  it('rejects ICU-private and malformed language tags', () => {
    expect(usableLocale('en-US@posix')).toBeNull();
    expect(usableLocale('not a tag')).toBeNull();
    expect(usableLocale('')).toBeNull();
    expect(usableLocale(undefined)).toBeNull();
    expect(usableLocale('en-US')).toBe('en-US');
    expect(usableLocale('de-de')).toBe('de-DE');
  });

  it('keeps a usable tag and strips the suffix of an unusable one', () => {
    expect(repairedLocale({ language: 'en-US', languages: ['en-US', 'de'] })).toBeNull();
    // Chromium reports ICU-private tags under the C/POSIX locale, which uPlot
    // passes straight into Intl.NumberFormat while the module loads.
    expect(repairedLocale({ language: 'en-US@posix', languages: ['en-US@posix'] })).toBe('en-US');
    expect(repairedLocale({ language: 'de_DE@euro', languages: [] })).toBe('de-DE');
    expect(repairedLocale({ language: 'x', languages: ['he', 'x'] })).toBe('he');
    expect(repairedLocale({ language: 'x', languages: ['y'] })).toBe('en-US');
  });

  it('overwrites an unusable host locale and leaves a usable one alone', () => {
    const broken = { language: 'en-US@posix', languages: ['en-US@posix'] };
    expect(repairBrowserLocale(broken)).toBe('en-US');
    expect(broken.language).toBe('en-US');
    expect(broken.languages).toEqual(['en-US']);
    expect(() => new Intl.NumberFormat(broken.language)).not.toThrow();

    const working = { language: 'fr-FR', languages: ['fr-FR'] };
    expect(repairBrowserLocale(working)).toBeNull();
    expect(working.language).toBe('fr-FR');
    expect(working.languages).toEqual(['fr-FR']);
  });
});
