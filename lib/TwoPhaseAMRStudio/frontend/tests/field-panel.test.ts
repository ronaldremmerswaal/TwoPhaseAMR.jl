import { describe, expect, it } from 'vitest';
import { combineRanges, statisticsRange, valuesRange } from '../src/field-panel';
import type { FrameSummary } from '../src/types';

const frame = (stats?: unknown): FrameSummary => ({ id: 4, time: 0.5, step: 12, kind: 'keyframe', topology: 2, stats });

describe('field colour-range data', () => {
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

  it('merges stored-frame ranges and can apply a symmetric signed range', () => {
    expect(combineRanges([null, [-3, 1], [-2, 5], undefined])).toEqual([-3, 5]);
    expect(combineRanges([[-3, 1], [-2, 5]], true)).toEqual([-5, 5]);
    expect(combineRanges([null, undefined])).toBeNull();
    expect(combineRanges([[0, 0]], true)).toEqual([-0, 0]);
  });
});
