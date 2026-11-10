import type { FrameId, FrameSummary } from './types';

export const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 2, 4, 8] as const;
const KEYFRAME_KINDS: Record<string, true> = { initial: true, keyframe: true, manual: true, pause: true, final: true };

export function isKeyframe(frame: FrameSummary): boolean {
  return KEYFRAME_KINDS[frame.kind] === true;
}

export interface TimelineBounds { start: number; end: number }

/** The axis includes the planned end and the solver head, not just stored frames. */
export function timelineBounds(frames: FrameSummary[], time: number | null, finalTime: number | null, initialTime: number | null): TimelineBounds | null {
  let start = Infinity;
  let end = -Infinity;
  for (const value of [initialTime, finalTime, time]) {
    if (typeof value !== 'number' || !Number.isFinite(value)) continue;
    start = Math.min(start, value);
    end = Math.max(end, value);
  }
  for (const frame of frames) {
    if (!Number.isFinite(frame.time)) continue;
    start = Math.min(start, frame.time);
    end = Math.max(end, frame.time);
  }
  return start === Infinity ? null : { start, end };
}

export function timelinePosition(time: number, bounds: TimelineBounds): number {
  if (!Number.isFinite(time) || bounds.start === bounds.end) return 0;
  return Math.max(0, Math.min(1, (time - bounds.start) / (bounds.end - bounds.start)));
}

export interface TimelineMarker {
  kind: 'keyframe' | 'preview' | 'regrid';
  time: number;
  position: number;
}

/** Mark topology changes at the first frame of the new topology. */
export function timelineMarkers(frames: FrameSummary[], bounds: TimelineBounds): TimelineMarker[] {
  const markers: TimelineMarker[] = [];
  frames.forEach((frame, index) => {
    const position = timelinePosition(frame.time, bounds);
    if (isKeyframe(frame)) markers.push({ kind: 'keyframe', time: frame.time, position });
    else if (frame.kind === 'preview') markers.push({ kind: 'preview', time: frame.time, position });
    if (index > 0 && frame.topology !== frames[index - 1].topology) {
      markers.push({ kind: 'regrid', time: frame.time, position });
    }
  });
  return markers;
}

function ordered(frames: FrameSummary[]): FrameSummary[] {
  return [...frames].sort((a, b) => {
    const timeAndStep = a.time - b.time || a.step - b.step;
    if (timeAndStep) return timeAndStep;
    if (typeof a.id === 'number' && typeof b.id === 'number') return a.id - b.id;
    const first = `${typeof a.id}:${a.id}`;
    const second = `${typeof b.id}:${b.id}`;
    return first < second ? -1 : first > second ? 1 : 0;
  });
}

/** Client-local navigation only. None of these operations sends a recorder command. */
export class Timeline {
  frames: FrameSummary[] = [];
  mode: 'live' | 'history' = 'live';
  selectedId: FrameId | null = null;
  newCount = 0;
  playing = false;
  speed: number = 1;
  loop = false;

  get index(): number { return this.frames.findIndex(frame => frame.id === this.selectedId); }
  get current(): FrameSummary | undefined { return this.frames[this.index]; }

  initialize(frames: FrameSummary[]): void {
    this.mode = 'live';
    this.newCount = 0;
    this.playing = false;
    this.replace(frames, false);
  }

  /** The server's full index is authoritative: entries may change kind or be removed. */
  replace(frames: FrameSummary[], countNew = true): void {
    const oldTime = this.current?.time;
    const known = new Set(this.frames.map(frame => frame.id));
    for (const frame of frames) {
      if (countNew && this.mode === 'history' && !known.has(frame.id)) this.newCount++;
      known.add(frame.id);
    }
    this.frames = ordered(frames);
    if (this.mode === 'live') {
      this.selectedId = this.frames.at(-1)?.id ?? null;
    } else if (this.index < 0) {
      if (this.frames.length === 0) {
        this.selectedId = null;
        this.playing = false;
      } else {
        const nearest = oldTime === undefined ? 0 : this.frames.reduce((best, frame, index) =>
          Math.abs(frame.time - oldTime) < Math.abs(this.frames[best].time - oldTime) ? index : best, 0);
        this.selectedId = this.frames[nearest].id;
      }
    }
  }

  /** Apply an immediate notification; the next full index reconciles retention/promotion. */
  arrive(frame: FrameSummary): void {
    const existing = this.frames.findIndex(item => item.id === frame.id);
    const frames = [...this.frames];
    if (existing < 0) frames.push(frame); else frames[existing] = frame;
    this.replace(frames);
  }

  follow(): void {
    this.mode = 'live';
    this.newCount = 0;
    this.playing = false;
    this.selectedId = this.frames.at(-1)?.id ?? null;
  }

  private history(): void {
    if (this.mode === 'live') {
      this.mode = 'history';
      this.newCount = 0;
    }
  }

  seek(index: number): void {
    this.history();
    if (this.frames.length) {
      const clamped = Math.min(this.frames.length - 1, Math.max(0, Math.round(index)));
      this.selectedId = this.frames[clamped].id;
    }
  }

  seekTime(time: number): void {
    if (!Number.isFinite(time) || !this.frames.length) return;
    let nearest = 0;
    for (let index = 1; index < this.frames.length; index++) {
      if (Math.abs(this.frames[index].time - time) < Math.abs(this.frames[nearest].time - time)) nearest = index;
    }
    this.seek(nearest);
  }

  step(direction: -1 | 1): void {
    this.playing = false;
    this.seek((this.index < 0 ? 0 : this.index) + direction);
  }

  jumpKeyframe(direction: -1 | 1): void {
    this.playing = false;
    this.history();
    for (let i = this.index + direction; i >= 0 && i < this.frames.length; i += direction) {
      if (isKeyframe(this.frames[i])) {
        this.selectedId = this.frames[i].id;
        break;
      }
    }
  }

  edge(edge: 'first' | 'last'): void {
    this.playing = false;
    this.seek(edge === 'first' ? 0 : this.frames.length - 1);
  }

  togglePlay(): void {
    if (!this.frames.length) return;
    if (this.playing) { this.playing = false; return; }
    this.history();
    if (this.frames.length > 1 && this.index === this.frames.length - 1) this.selectedId = this.frames[0].id;
    this.playing = true;
  }

  advance(): void {
    if (!this.playing || !this.frames.length) return;
    if (this.index < this.frames.length - 1) this.seek(this.index + 1);
    else if (this.loop) this.seek(0);
    else this.playing = false;
  }

  changeSpeed(direction: -1 | 1): void {
    const current = PLAYBACK_SPEEDS.indexOf(this.speed as typeof PLAYBACK_SPEEDS[number]);
    this.speed = PLAYBACK_SPEEDS[Math.max(0, Math.min(PLAYBACK_SPEEDS.length - 1, current + direction))];
  }
}
