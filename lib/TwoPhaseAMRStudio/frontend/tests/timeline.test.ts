import { describe, expect, it } from 'vitest';
import { Timeline } from '../src/timeline';
import type { FrameSummary } from '../src/types';

const frame = (id: number, kind = 'preview'): FrameSummary => ({ id, time: id * 0.1, step: id, kind, topology: 1 });

describe('per-client timeline', () => {
  it('follows live, pins history, counts arrivals once, and resumes following', () => {
    const timeline = new Timeline();
    timeline.initialize([frame(1, 'initial'), frame(2), frame(3, 'keyframe')]);
    expect(timeline.current?.id).toBe(3);
    timeline.step(-1);
    expect(timeline.mode).toBe('history');
    expect(timeline.current?.id).toBe(2);
    timeline.arrive(frame(4));
    timeline.arrive(frame(4));
    timeline.replace([frame(1, 'initial'), frame(2), frame(3, 'keyframe'), frame(4), frame(5)]);
    expect(timeline.current?.id).toBe(2);
    expect(timeline.newCount).toBe(2);
    timeline.follow();
    expect(timeline.mode).toBe('live');
    expect(timeline.current?.id).toBe(5);
    expect(timeline.newCount).toBe(0);
    timeline.arrive(frame(6));
    expect(timeline.current?.id).toBe(6);
  });

  it('orders late-committed keyframes by simulation time and stable tie-breakers', () => {
    const timeline = new Timeline();
    timeline.initialize([
      { ...frame(8), time: 0.4, step: 4 },
      { ...frame(12, 'keyframe'), time: 0.2, step: 2 },
      { ...frame(11, 'keyframe'), time: 0.2, step: 2 },
    ]);
    expect(timeline.frames.map(item => item.id)).toEqual([11, 12, 8]);
    expect(timeline.current?.id).toBe(8);
    timeline.step(-1);
    expect(timeline.current?.id).toBe(12);
    timeline.step(-1);
    expect(timeline.current?.id).toBe(11);
  });

  it('reconciles preview removal and keyframe promotion from the authoritative index', () => {
    const timeline = new Timeline();
    timeline.initialize([frame(1, 'initial'), frame(2), frame(3)]);
    timeline.seek(1);
    timeline.replace([frame(1, 'initial'), frame(2, 'keyframe'), frame(4)]);
    expect(timeline.frames.map(item => item.id)).toEqual([1, 2, 4]);
    expect(timeline.current?.kind).toBe('keyframe');
    expect(timeline.newCount).toBe(1);
    timeline.jumpKeyframe(-1);
    expect(timeline.current?.id).toBe(1);
    timeline.jumpKeyframe(1);
    expect(timeline.current?.id).toBe(2);
    timeline.replace([frame(1, 'initial'), frame(4)]);
    expect(timeline.current?.id).toBe(1);
    timeline.replace([]);
    expect(timeline.current).toBeUndefined();
    expect(timeline.playing).toBe(false);
  });

  it('counts arrivals while retaining only the server index', () => {
    const timeline = new Timeline();
    timeline.initialize([frame(1, 'initial'), frame(2)]);
    timeline.seek(0);
    for (let id = 3; id <= 1000; id++) {
      timeline.arrive(frame(id));
      timeline.replace([frame(1, 'initial'), frame(id)]);
    }
    expect(timeline.current?.id).toBe(1);
    expect(timeline.newCount).toBe(998);
    expect(timeline.frames.map(item => item.id)).toEqual([1, 1000]);
  });

  it('navigates frames, boundaries, keyframes, playback speed and looping locally', () => {
    const timeline = new Timeline();
    timeline.initialize([frame(1, 'initial'), frame(2), frame(3, 'manual'), frame(4, 'final')]);
    timeline.jumpKeyframe(-1);
    expect(timeline.current?.id).toBe(3);
    timeline.edge('first');
    timeline.step(1);
    expect(timeline.current?.id).toBe(2);
    timeline.edge('last');
    timeline.togglePlay();
    expect(timeline.mode).toBe('history');
    expect(timeline.current?.id).toBe(1);
    timeline.advance();
    expect(timeline.current?.id).toBe(2);
    timeline.changeSpeed(1);
    expect(timeline.speed).toBe(2);
    timeline.edge('last');
    timeline.loop = true;
    timeline.togglePlay();
    timeline.seek(3);
    timeline.advance();
    expect(timeline.current?.id).toBe(1);
    timeline.loop = false;
    timeline.seek(3);
    timeline.advance();
    expect(timeline.playing).toBe(false);
  });
});
