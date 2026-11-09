import { describe, expect, it } from 'vitest';
import { formatBlockSize, formatBytes, formatCount, formatDomain, formatNumber, formatRate, formatStatusNumber, formatStore } from '../src/panels';

describe('run metadata formatting', () => {
  it('shows physical domain bounds and block dimensions', () => {
    expect(formatDomain({ origin: [-1, 0.25], extent: [2, 1.75] })).toBe('[-1, 1] × [0.25, 2]');
    expect(formatBlockSize([4, 4])).toBe('4 × 4');
    expect(formatDomain({ origin: [0, 0], extent: [1, 1] })).toBe('[0, 1] × [0, 1]');
    expect(formatDomain({ origin: [0, null], extent: [1, 1] })).toBe('—');
    expect(formatDomain(null)).toBe('—');
    expect(formatBlockSize([4, '4'])).toBe('—');
    expect(formatBlockSize(null)).toBe('—');
  });

  it('rounds solver throughput to three significant figures', () => {
    expect(formatRate(31.12345)).toBe('31.1');
    expect(formatRate(0.00123456)).toBe('0.00123');
    expect(formatRate(12345.6)).toBe('12300');
    expect(formatRate(null)).toBe('—');
  });

  it('formats status times and the last step to four significant figures', () => {
    expect(formatStatusNumber(0.798283)).toBe('0.7983');
    expect(formatStatusNumber(5.451234e-14)).toBe('5.451e-14');
    expect(formatStatusNumber(null)).toBe('—');
  });

  it('formats byte units and actual frame counts without inventing missing values', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1000)).toBe('1 KB');
    expect(formatBytes(412_000_000)).toBe('412 MB');
    expect(formatBytes(2_000_000_000)).toBe('2 GB');
    expect(formatBytes(2_147_483_648)).toBe('2.15 GB');
    expect(formatBytes(-1)).toBe('—');
    expect(formatBytes(null)).toBe('—');
    expect(formatStore(412_000_000, 2_000_000_000, 37)).toBe('412 MB / 2 GB (37 frames)');
    expect(formatStore(null, null, null)).toBe('— / — (— frames)');
    expect(formatStore(1, 2, 1)).toBe('1 B / 2 B (1 frame)');
    expect(formatNumber(NaN)).toBe('—');
    expect(formatCount(1_234_567)).toBe('1234567');
    expect(formatCount(NaN)).toBe('—');
    expect(formatNumber(Infinity)).toBe('—');
  });
});
