import { describe, expect, it } from 'vitest';
import {
  frameSelection, formatSimTime, liveSpacing, liveSummary, parseTimeWindow,
  scheduleAttributes, scheduleSummary,
} from '../src/export-plan';
import type { RecordingSchedule } from '../src/types';

const schedule: RecordingSchedule = {
  frames: 41, distinct: 3, repeats: 38, sparse: true, window: [0, 0.04],
  fps: 20, duration: 2, spacing: 0.001, sim_per_second: 0.02, first_frame: 1,
};

describe('export plan', () => {
  it('treats both empty sides as the whole recording and rejects half a window', () => {
    expect(parseTimeWindow('', '')).toEqual({});
    expect(parseTimeWindow('0.1', '').error).toBeTruthy();
    expect(parseTimeWindow('', '0.8').error).toBeTruthy();
    expect(parseTimeWindow('0.1', '0.8')).toEqual({ window: [0.1, 0.8] });
  });

  it('rejects an invalid or reversed window', () => {
    expect(parseTimeWindow('0.8', '0.1').error).toBeTruthy();
    expect(parseTimeWindow('nan', '0.8').error).toBeTruthy();
    expect(parseTimeWindow('0.1', 'Infinity').error).toBeTruthy();
  });

  it('derives the live output spacing from simulation time per video second', () => {
    expect(liveSpacing(0.2, 20)).toBeCloseTo(0.01);
    expect(liveSummary(0.2, 20)).toContain('0.01 simulation s per frame');
    expect(liveSummary(0.2, 20)).toContain('0.2 per video second');
    expect(liveSummary(0.2, 20)).toContain('grows until stopped');
  });

  it('formats small and large simulation times without losing the value', () => {
    expect(formatSimTime(0)).toBe('0');
    expect(formatSimTime(0.00123456)).toBe('0.001235');
    expect(formatSimTime(2e-9)).toMatch(/e-9$/);
    expect(formatSimTime(Number.NaN)).toBe('—');
  });

  it('summarizes the stored schedule and flags a sparse recording', () => {
    const text = scheduleSummary(schedule);
    expect(text).toContain('41 output frames');
    expect(text).toContain('2 s at 20 fps');
    expect(text).toContain('0.001 simulation s per frame');
    expect(text).toContain('3 distinct stored frames');
    expect(text).toContain('sparse recording');
    expect(scheduleSummary({ ...schedule, sparse: false })).not.toContain('sparse');
    expect(scheduleAttributes(schedule)).toMatchObject({
      'data-frames': '41', 'data-distinct': '3', 'data-spacing': '0.001',
      'data-window-start': '0', 'data-window-end': '0.04', 'data-sparse': 'true',
    });
  });

  it('accepts only the two frame candidate sets', () => {
    expect(frameSelection('all')).toBe('all');
    expect(frameSelection('keyframes')).toBe('keyframes');
    expect(frameSelection('every')).toBe('keyframes');
  });
});
