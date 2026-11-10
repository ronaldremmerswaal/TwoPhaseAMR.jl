import { Timeline } from './timeline';
import { idleRecording, type RecordingState, type RunSummary, type ServerMessage } from './types';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export function runStatePresentation(runState: string, stopped: boolean | null): { label: string; tone: string } {
  switch (runState) {
    case 'running': return { label: 'RUNNING', tone: 'running' };
    case 'pause_requested': return { label: 'PAUSING', tone: 'pending' };
    case 'paused': return { label: 'PAUSED', tone: 'pending' };
    case 'stopping': return { label: 'STOPPING', tone: 'pending' };
    case 'completed': return { label: stopped === null ? 'COMPLETED' : stopped ? 'STOPPED' : 'FINISHED', tone: 'neutral' };
    case 'failed': return { label: 'FAILED', tone: 'failed' };
    case 'created': return { label: 'CREATED', tone: 'neutral' };
    default: return { label: runState.toUpperCase(), tone: 'neutral' };
  }
}

/** Solver commands cannot affect a completed, failed, or stopping run. */
export function solverControlsEnabled(connection: ConnectionStatus, runState: string): boolean {
  return connection === 'connected' && ['created', 'running', 'pause_requested', 'paused'].includes(runState);
}

function finite(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Shared server state plus a separate, strictly client-local timeline. */
export class StudioState {
  readonly timeline = new Timeline();
  connection: ConnectionStatus = 'connecting';
  error: string | null = null;
  runs: RunSummary[] = [];
  runState = 'unknown';
  time: number | null = null;
  step: number | null = null;
  dt: number | null = null;
  storeBytes: number | null = null;
  storeBudget: number | null = null;
  frameCount: number | null = null;
  stepsPerSecond: number | null = null;
  stopped: boolean | null = null;
  finalTime: number | null = null;
  recording: RecordingState = idleRecording();
  preferredTheme: 'light' | 'dark' | 'system' | null = null;
  private listeners = new Set<() => void>();

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  changed(): void { for (const listener of this.listeners) listener(); }

  navigate(action: (timeline: Timeline) => void): void {
    action(this.timeline);
    this.changed();
  }

  receive(message: ServerMessage): void {
    switch (message.type) {
      case 'hello':
        this.runs = message.runs;
        this.timeline.initialize(message.frames);
        this.finalTime = finite(message.runs[0]?.t_final) ?? this.finalTime;
        this.recording = message.recording;
        this.preferredTheme = message.theme ?? null;
        this.connection = 'connected';
        this.error = null;
        break;
      case 'state':
        this.runState = message.run_state;
        this.time = message.time;
        this.step = message.step;
        this.dt = message.dt;
        this.storeBytes = finite(message.store_bytes);
        this.storeBudget = finite(message.store_budget);
        this.frameCount = finite(message.frame_count);
        this.stepsPerSecond = finite(message.steps_per_second);
        this.stopped = typeof message.stopped === 'boolean' ? message.stopped : null;
        this.finalTime = finite(message.t_final) ?? this.finalTime;
        this.recording = message.recording;
        break;
      case 'frame':
        this.timeline.arrive(message);
        break;
      case 'frames':
        this.timeline.replace(message.frames);
        break;
      case 'error':
        this.error = message.message;
        break;
    }
    this.changed();
  }

  setConnection(status: ConnectionStatus): void { this.connection = status; this.changed(); }
  setError(message: string): void { this.error = message; this.changed(); }
  clearError(): void { this.error = null; this.changed(); }
}
