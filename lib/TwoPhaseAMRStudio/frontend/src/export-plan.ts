import type { RecordingSchedule } from './types';

/** Time-window form fields: an omitted side means the recording's edge. */
export interface WindowSelection {
  window?: [number, number];
  error?: string;
}

export function parseTimeWindow(from: string, to: string): WindowSelection {
  const hasStart = from.trim() !== '';
  const hasEnd = to.trim() !== '';
  if (!hasStart && !hasEnd) return {};
  if (hasStart !== hasEnd) {
    return { error: 'Enter both window bounds, or leave both empty for the whole recording.' };
  }
  const start = Number(from);
  const end = Number(to);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
    return { error: 'Choose a valid export time window.' };
  }
  return { window: [start, end] };
}

/** Simulation-time spacing of a live export on a uniform output grid. */
export function liveSpacing(simPerSecond: number, fps: number): number {
  return simPerSecond / fps;
}

export function formatSimTime(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e-3 && magnitude < 1e6) return Number(value.toPrecision(4)).toString();
  return value.toExponential(3);
}

/**
 * Summary of a stored export: output frame count, play time, uniform
 * simulation-time spacing, and how many stored frames are distinct. The
 * warning is part of the text so a caller can show it verbatim.
 */
export function scheduleSummary(schedule: RecordingSchedule): string {
  const parts = [
    `${schedule.frames} output frames · ${formatSimTime(schedule.duration)} s at ${schedule.fps} fps`,
    `Δt = ${formatSimTime(schedule.spacing)} simulation s per frame · ${formatSimTime(schedule.sim_per_second)} per video second`,
    `${schedule.distinct} distinct stored frames`,
  ];
  if (schedule.sparse) {
    parts.push('sparse recording: many output frames repeat a stored frame');
  }
  return parts.join(' · ');
}

/** Attributes attached to the summary element for tests and styling. */
export function scheduleAttributes(schedule: RecordingSchedule): Record<string, string> {
  return {
    'data-frames': String(schedule.frames),
    'data-distinct': String(schedule.distinct),
    'data-spacing': String(schedule.spacing),
    'data-sim-per-second': String(schedule.sim_per_second),
    'data-window-start': String(schedule.window[0]),
    'data-window-end': String(schedule.window[1]),
    'data-fps': String(schedule.fps),
    'data-sparse': String(schedule.sparse),
  };
}

/**
 * Summary of a live export. Its duration and frame count are not known until
 * recording stops, but the uniform simulation-time spacing is.
 */
export function liveSummary(simPerSecond: number, fps: number): string {
  return `Δt = ${formatSimTime(liveSpacing(simPerSecond, fps))} simulation s per frame · ` +
    `${formatSimTime(simPerSecond)} per video second · ${fps} fps · the video grows until stopped`;
}

/** Frame-selection values the server accepts for stored exports. */
export type FrameSelection = 'keyframes' | 'all';

export function frameSelection(value: string): FrameSelection {
  return value === 'all' ? 'all' : 'keyframes';
}
