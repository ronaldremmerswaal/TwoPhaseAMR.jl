import { Timeline } from './timeline';
import { idleRecording, type RecordingState, type RunSummary, type ServerMessage } from './types';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

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
  storeBytes = 0;
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
        this.storeBytes = message.store_bytes;
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
