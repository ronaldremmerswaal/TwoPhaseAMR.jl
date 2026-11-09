import { describe, expect, it } from 'vitest';
import { runStatePresentation, solverControlsEnabled, StudioState } from '../src/state';
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

  it('tracks server statistics and retains the target when the run ends', () => {
    const state = new StudioState();
    state.receive({ type: 'hello', protocol: 1, runs: [run], frames: [frame(1)], recording });
    expect(state.finalTime).toBeNull();
    state.receive({
      type: 'state', run_state: 'running', time: 0.5, step: 5, dt: 0.1,
      store_bytes: 1234, store_budget: 8000, frame_count: 3,
      steps_per_second: 12.5, stopped: false, t_final: 2, recording,
    });
    expect([state.storeBytes, state.storeBudget, state.frameCount, state.stepsPerSecond]).toEqual([1234, 8000, 3, 12.5]);
    expect(state.finalTime).toBe(2);
    state.receive({
      type: 'state', run_state: 'completed', time: 0.6, step: 6, dt: 0.1,
      store_bytes: 1500, store_budget: 8000, frame_count: 4,
      steps_per_second: null, stopped: true, t_final: null, recording,
    });
    expect(state.finalTime).toBe(2);
    expect(state.time).toBe(0.6);
    expect(state.stepsPerSecond).toBeNull();
    expect(runStatePresentation(state.runState, state.stopped).label).toBe('STOPPED');
  });

  it('distinguishes stopped, finished, transitional, failed and unknown states', () => {
    expect(runStatePresentation('running', false)).toEqual({ label: 'RUNNING', tone: 'running' });
    expect(runStatePresentation('paused', false)).toEqual({ label: 'PAUSED', tone: 'pending' });
    expect(runStatePresentation('pause_requested', false).label).toBe('PAUSING');
    expect(runStatePresentation('stopping', false).label).toBe('STOPPING');
    expect(runStatePresentation('created', false).label).toBe('CREATED');
    expect(runStatePresentation('failed', false)).toEqual({ label: 'FAILED', tone: 'failed' });
    expect(runStatePresentation('completed', true).label).toBe('STOPPED');
    expect(runStatePresentation('completed', false).label).toBe('FINISHED');
    expect(runStatePresentation('completed', null).label).toBe('COMPLETED');
  });

  it('disables solver controls after finishing, stopping, failure, or disconnection', () => {
    for (const status of ['created', 'running', 'pause_requested', 'paused']) {
      expect(solverControlsEnabled('connected', status)).toBe(true);
      expect(solverControlsEnabled('disconnected', status)).toBe(false);
    }
    for (const status of ['completed', 'stopping', 'failed', 'unknown']) {
      expect(solverControlsEnabled('connected', status)).toBe(false);
    }
  });

  it('uses hello metadata for the target and treats absent additive fields as unknown', () => {
    const state = new StudioState();
    state.receive({ type: 'hello', protocol: 1, runs: [{ ...run, t_final: 5 }], frames: [], recording });
    state.receive({ type: 'state', run_state: 'completed', time: 5, step: 10, dt: 0.5, store_bytes: 0, recording });
    expect(state.finalTime).toBe(5);
    expect([state.storeBudget, state.frameCount, state.stepsPerSecond, state.stopped]).toEqual([null, null, null, null]);
    expect(runStatePresentation(state.runState, state.stopped).label).toBe('COMPLETED');
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
