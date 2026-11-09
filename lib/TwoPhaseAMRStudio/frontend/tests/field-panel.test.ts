import { describe, expect, it } from 'vitest';
import { combineRanges, defaultFieldColor, resolvePanelFrame, robustValuesRange, sameFieldDataKey, statisticsRange, valuesRange } from '../src/field-panel';
import type { FrameSummary } from '../src/types';

const frame = (stats?: unknown): FrameSummary => ({ id: 4, time: 0.5, step: 12, kind: 'keyframe', topology: 2, stats });

describe('field colour-range data', () => {
  it('defaults automatic fields to robust ranges and respects signed and fixed metadata', () => {
    expect(defaultFieldColor({ name: 'velocity_magnitude', colormap: 'viridis' }))
      .toEqual({ fixedRange: null, colorPolicy: 'robust', symmetric: false });
    expect(defaultFieldColor({ name: 'velocity_vorticity', symmetric: true, colormap: 'viridis' }))
      .toEqual({ fixedRange: null, colorPolicy: 'robust', symmetric: true });
    expect(defaultFieldColor({ name: 'pressure', colormap: 'balance' }))
      .toEqual({ fixedRange: null, colorPolicy: 'robust', symmetric: true });
    expect(defaultFieldColor({ name: 'volume_fraction', colorrange: [0, 1], symmetric: true }))
      .toEqual({ fixedRange: [0, 1], colorPolicy: 'fixed', symmetric: false });
    expect(defaultFieldColor({ name: 'bad_range', colorrange: [4, -4] }))
      .toEqual({ fixedRange: null, colorPolicy: 'robust', symmetric: false });
  });

  it('distinguishes missing, empty, and finite server statistics', () => {
    expect(statisticsRange(frame(), 'velocity')).toBeUndefined();
    expect(statisticsRange(frame({ velocity: { finite: 0, minimum: null, maximum: null } }), 'velocity')).toBeNull();
    expect(statisticsRange(frame({ velocity: { finite: 3, minimum: -2.5, maximum: 4 } }), 'velocity')).toEqual([-2.5, 4]);
    expect(statisticsRange(frame({ velocity: { finite: 3, minimum: 4, maximum: -2 } }), 'velocity')).toBeUndefined();
    expect(statisticsRange(frame({ velocity: { finite: 3, minimum: null, maximum: 4 } }), 'velocity')).toBeUndefined();
  });

  it('falls back to both intrinsic phase arrays and ignores non-finite entries', () => {
    expect(valuesRange(new Float32Array([NaN, -3, 1]), new Float32Array([Infinity, 5, -Infinity]))).toEqual([-3, 5]);
    expect(valuesRange(new Float32Array([NaN, Infinity]))).toBeNull();
    expect(valuesRange(new Float32Array([7, 7]))).toEqual([7, 7]);
  });

  it('clips the 99.5th percentile per frame using both phase arrays', () => {
    const liquid = Float32Array.from(Array.from({ length: 100 }, (_, i) => i));
    const gas = Float32Array.from([...Array.from({ length: 99 }, (_, i) => 100 + i), 10000, NaN, Infinity]);
    expect(robustValuesRange(false, liquid, gas)).toEqual([0, 198]);
    expect(robustValuesRange(true,
      Float32Array.from([...Array(100).fill(-2), NaN]),
      Float32Array.from([...Array(99).fill(3), -1000, Infinity]))).toEqual([-3, 3]);
    expect(robustValuesRange(false, new Float32Array([-5, -2, -3, NaN]))).toEqual([-5, -2]);
    expect(robustValuesRange(false, new Float32Array([NaN, Infinity]))).toBeNull();
    expect(robustValuesRange(true, new Float32Array([0, 0]))).toEqual([-0, 0]);
  });

  it('selects the exact finite percentile without modifying the source values', () => {
    const first = Float32Array.from(Array.from({ length: 1201 }, (_, i) => Math.sin(i * 1.7) * (i % 47)));
    const second = Float32Array.from([...Array.from({ length: 800 }, (_, i) => Math.cos(i * 0.9) * (i % 39)), NaN, Infinity]);
    const originalFirst = first.slice(), originalSecond = second.slice();
    const values = [...first, ...second].filter(Number.isFinite).sort((a, b) => a - b);
    const rank = Math.ceil(values.length * 0.995) - 1;
    expect(robustValuesRange(false, first, second)).toEqual([values[0], values[rank]]);
    const magnitudes = values.map(Math.abs).sort((a, b) => a - b);
    expect(robustValuesRange(true, first, second)).toEqual([-magnitudes[rank], magnitudes[rank]]);
    expect(first).toEqual(originalFirst);
    expect(second).toEqual(originalSecond);
  });

  it('merges stored-frame ranges and can apply a symmetric signed range', () => {
    expect(combineRanges([null, [-3, 1], [-2, 5], undefined])).toEqual([-3, 5]);
    expect(combineRanges([[-3, 1], [-2, 5]], true)).toEqual([-5, 5]);
    expect(combineRanges([null, undefined])).toBeNull();
    expect(combineRanges([[0, 0]], true)).toEqual([-0, 0]);
  });
});

describe('field frame pins', () => {
  it('follows the playhead only when unpinned and retains thinned preview frames', () => {
    const pinned = { ...frame(), id: 1, time: 0.1, kind: 'preview' };
    const playhead = { ...frame(), id: 2, time: 0.2 };
    expect(resolvePanelFrame(null, pinned, [pinned, playhead], playhead)).toBe(playhead);
    expect(resolvePanelFrame(1, pinned, [pinned, playhead], playhead)).toBe(pinned);
    expect(resolvePanelFrame(1, pinned, [playhead], playhead)).toBe(pinned);
    expect(resolvePanelFrame(1, undefined, [playhead], playhead)).toBeUndefined();
    expect(resolvePanelFrame(1, playhead, [playhead], playhead)).toBeUndefined();
    const promoted = { ...pinned, kind: 'keyframe' };
    expect(resolvePanelFrame(1, pinned, [promoted, playhead], playhead)).toBe(promoted);
  });

  it('retains cached data across epochs only for the same frame, topology and field', () => {
    const original = JSON.stringify([1, 4, 2, 'velocity', 1, 'cell']);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 2, 'velocity', 1, 'cell']))).toBe(true);
    expect(sameFieldDataKey(original, JSON.stringify([2, 5, 2, 'velocity', 1, 'cell']))).toBe(false);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 3, 'velocity', 1, 'cell']))).toBe(false);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 2, 'pressure', 1, 'cell']))).toBe(false);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 2, 'velocity', 0, 'cell']))).toBe(false);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 2, 'velocity', 1, 'face_x']))).toBe(false);
  });
});
