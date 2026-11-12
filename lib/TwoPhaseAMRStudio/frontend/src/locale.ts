/**
 * uPlot builds its tick formatter with `new Intl.NumberFormat(navigator.language)`
 * when the module loads, and `Intl` throws a RangeError for ICU-private language
 * tags. Chromium reports exactly such a tag under the C/POSIX locale (for example
 * `en-US@posix`), and the uncaught error aborts the whole Studio bundle before the
 * first render, leaving the page on "Connecting to TwoPhaseAMR Studio…".
 *
 * Repair the reported locale once at startup so every `Intl` user in the app sees
 * a tag `Intl` accepts. Valid locales are left untouched.
 */

/** Canonical form of `tag`, or null when `Intl` rejects it. */
export function usableLocale(tag: unknown): string | null {
  if (typeof tag !== 'string' || tag.trim() === '') return null;
  try {
    return Intl.getCanonicalLocales(tag)[0] ?? null;
  } catch {
    return null;
  }
}

/** Locale to substitute for a host whose reported language `Intl` rejects. */
export function repairedLocale(host: { language: string; languages: readonly string[] }): string | null {
  if (usableLocale(host.language) !== null) return null;
  const candidates: string[] = [];
  for (const tag of [host.language, ...host.languages]) {
    if (typeof tag !== 'string') continue;
    // `de-DE@euro` and `de_DE.UTF-8` name a language plus ICU-private, charset
    // and POSIX separators: try the BCP 47 spelling of the bare tag as well.
    const bare = tag.split('@')[0].split('.')[0];
    candidates.push(tag, bare, bare.replace(/_/g, '-'));
  }
  for (const candidate of candidates) {
    const locale = usableLocale(candidate);
    if (locale !== null) return locale;
  }
  return 'en-US';
}

/** Overwrite an unusable host locale with a valid one; returns the replacement. */
export function repairBrowserLocale(
  host: { language: string; languages: readonly string[] } = navigator,
): string | null {
  const locale = repairedLocale(host);
  if (locale === null) return null;
  Object.defineProperties(host, {
    language: { configurable: true, enumerable: true, get: () => locale },
    languages: { configurable: true, enumerable: true, get: () => [locale] },
  });
  return locale;
}

repairBrowserLocale();
