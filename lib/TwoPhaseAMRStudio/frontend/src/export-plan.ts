import type { RecordingSchedule } from './types';

/**
 * Largest output frame count the server plans (`MAX_VIDEO_FRAMES` in the core
 * package). Keep in step with `src/observation/video.jl`.
 */
export const MAX_EXPORT_FRAMES = 100_000;

/** Longest stored export duration whose frame count stays within the cap. */
export function durationLimit(fps: number): number {
  return Number.isFinite(fps) && fps > 0 ? (MAX_EXPORT_FRAMES - 1) / fps : 1;
}

/**
 * Identity of a preflight request: its command plus the stored frames the
 * dialog had when it asked, so newly stored frames make the summary stale.
 */
export function scheduleKey(
  command: object, frames: readonly { id: number | string; time: number }[],
): string {
  const last = frames[frames.length - 1];
  return JSON.stringify([command, frames.length, last ? [last.id, last.time] : null]);
}

/**
 * Bookkeeping for the dialog's schedule preflight. A key is armed while its
 * debounce timer runs, pending while its response is in flight, applied once a
 * response was shown, and failed when the server rejected it; only a key in
 * none of those states is requested again, so new frames or edited fields
 * refresh the summary without re-asking for an unchanged plan.
 */
export class ScheduleRequests {
  private armed = '';
  private pending = '';
  private applied = '';
  private failed = '';
  private generation = 0;

  /** True when no armed, in-flight, applied, or failed request covers `key`. */
  needs(key: string): boolean {
    return key !== this.armed && key !== this.pending &&
      key !== this.applied && key !== this.failed;
  }

  /** Remember the key whose debounce timer is pending. */
  arm(key: string): void { this.armed = key; }

  /** Forget an armed request, e.g. when the form changed again. */
  disarm(): void { this.armed = ''; }

  /** Mark `key` as in flight and return the generation of this request. */
  start(key: string): number {
    this.armed = '';
    this.pending = key;
    return ++this.generation;
  }

  /** Record a shown answer; false when a newer request superseded it. */
  settle(generation: number, key: string): boolean {
    if (generation !== this.generation) return false;
    this.pending = '';
    this.applied = key;
    this.failed = '';
    return true;
  }

  /** Record a rejected answer; false when a newer request superseded it. */
  fail(generation: number, key: string): boolean {
    if (generation !== this.generation) return false;
    this.pending = '';
    this.failed = key;
    return true;
  }

  /**
   * Drop the request in flight so a later pass can ask again. A response that
   * was discarded because the dialog switched to a live source must not leave
   * its key pending forever, which would stop every later stored request.
   */
  cancel(generation: number): void {
    if (generation === this.generation) this.pending = '';
  }

  /** Forget every request, e.g. when the export source changes. */
  reset(): void {
    this.armed = '';
    this.pending = '';
    this.applied = '';
    this.failed = '';
    this.generation += 1;
  }
}

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
