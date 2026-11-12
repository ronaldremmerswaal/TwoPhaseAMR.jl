import { describe, expect, it } from 'vitest';
import {
  durationLimit, frameSelection, formatSimTime, liveSpacing, liveSummary, MAX_EXPORT_FRAMES,
  parseTimeWindow, pendingScheduleText, scheduleAttributes, scheduleKey, scheduleSummary, ScheduleRequests,
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

  it('bounds the duration input by the server frame cap', () => {
    expect(MAX_EXPORT_FRAMES).toBe(100_000);
    expect(durationLimit(20)).toBeCloseTo((MAX_EXPORT_FRAMES - 1) / 20);
    // Invalid frame rates keep a usable bound instead of an infinite input.
    expect(durationLimit(0)).toBe(1);
    expect(durationLimit(Number.NaN)).toBe(1);
  });
});

describe('export schedule requests', () => {
  const command = { field: 'pressure', fps: 10, duration: 1 };
  const frames = [{ id: 1, time: 0 }, { id: 2, time: 0.5 }];

  it('keys a preflight on its command and on the stored frames it saw', () => {
    const key = scheduleKey(command, frames);
    expect(scheduleKey(command, frames)).toBe(key);
    // A newly stored frame makes the previous summary stale.
    expect(scheduleKey(command, [...frames, { id: 3, time: 1 }])).not.toBe(key);
    expect(scheduleKey(command, frames.slice(0, 1))).not.toBe(key);
    expect(scheduleKey({ ...command, fps: 20 }, frames)).not.toBe(key);
  });

  it('asks once per key and applies only the newest answer', () => {
    const requests = new ScheduleRequests();
    expect(requests.needs('a')).toBe(true);
    requests.arm('a');
    expect(requests.needs('a')).toBe(false);
    const first = requests.start('a');
    expect(requests.needs('a')).toBe(false);
    const second = requests.start('b');
    expect(requests.settle(first, 'a')).toBe(false);
    expect(requests.needs('b')).toBe(false);
    expect(requests.settle(second, 'b')).toBe(true);
    expect(requests.needs('b')).toBe(false);
    expect(requests.needs('c')).toBe(true);
  });

  it('frees a discarded response and retries only a changed plan after an error', () => {
    const requests = new ScheduleRequests();
    // A response dropped because the dialog switched to a live source must not
    // leave its key pending, which blocked every later stored preflight.
    requests.cancel(requests.start('stored'));
    expect(requests.needs('stored')).toBe(true);
    expect(requests.fail(requests.start('bad'), 'bad')).toBe(true);
    expect(requests.needs('bad')).toBe(false);
    expect(requests.needs('edited')).toBe(true);
    expect(requests.settle(requests.start('x'), 'x')).toBe(true);
    requests.reset();
    expect(requests.needs('x')).toBe(true);
  });

  it('keeps a preflight failure visible instead of reverting to progress text', () => {
    // `update()` re-renders the summary on every state/frame message, so a
    // failure must outrank the "Computing…" placeholder until the plan changes.
    expect(pendingScheduleText(null)).toContain('Computing');
    expect(pendingScheduleText('fps × duration gives 100001 output frames')).toBe(
      'fps × duration gives 100001 output frames',
    );
  });
});
