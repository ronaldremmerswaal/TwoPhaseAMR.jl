import { describe, expect, it } from 'vitest';
import { StudioState } from '../src/state';
import type { FrameSummary, RecordingState, RunSummary } from '../src/types';

const frame = (id: number): FrameSummary => ({ id, time: id, step: id, kind: id === 1 ? 'initial' : 'preview', topology: 1 });
const run: RunSummary = { id: 'run-1', name: 'Run 1', domain: null, block_size: null, t0: null, t_final: null, fields: [], diagnostics: [] };
const recording: RecordingState = { active: true, source: 'live', field: 'pressure', path: 'run.mp4', encoded: 7, dropped: 2, status: 'recording' };

describe('Studio state', () => {
  it('keeps local view operations independent of shared recording', () => {
    const state = new StudioState();
    state.receive({ type: 'hello', protocol: 1, runs: [run], frames: [frame(1), frame(2)], recording, theme: 'dark' });
    expect(state.preferredTheme).toBe('dark');
    state.receive({ type: 'state', run_state: 'running', time: 2, step: 2, dt: 0.1, store_bytes: 100, recording });
    state.navigate(timeline => timeline.step(-1));
    state.receive({ type: 'frame', ...frame(3) });
    expect(state.timeline.mode).toBe('history');
    expect(state.timeline.current?.id).toBe(1);
    expect(state.timeline.newCount).toBe(1);
    expect(state.recording).toEqual(recording);
    state.navigate(timeline => timeline.follow());
    expect(state.recording).toEqual(recording);
    expect(state.timeline.current?.id).toBe(3);
    state.receive({ type: 'state', run_state: 'paused', time: null, step: null, dt: null, store_bytes: 100, recording: { ...recording, active: false, status: 'completed' } });
    expect(state.runState).toBe('paused');
    expect(state.recording.active).toBe(false);
    expect(state.time).toBeNull();
    expect(state.step).toBeNull();
  });

  it('notifies subscribers, reports errors, and stops notifications after disposal', () => {
    const state = new StudioState();
    let changes = 0;
    const unsubscribe = state.subscribe(() => { changes++; });
    state.setConnection('connected');
    state.receive({ type: 'error', message: 'Denied', request: 4 });
    expect(state.error).toBe('Denied');
    state.clearError();
    expect(changes).toBe(3);
    unsubscribe();
    state.setConnection('disconnected');
    expect(changes).toBe(3);
  });
});
