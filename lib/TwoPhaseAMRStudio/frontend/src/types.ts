export type FrameId = number | string;

export interface FrameSummary {
  id: FrameId;
  time: number;
  step: number;
  kind: string;
  topology: number | string;
  run?: number | string;
  stats?: unknown;
}

export interface NamedMetadata {
  name: string;
  label?: string;
  units?: string;
  [key: string]: unknown;
}

export interface RunSummary {
  id: number | string;
  name: string;
  domain: unknown;
  block_size: unknown;
  t0: number | null;
  t_final: number | null;
  fields: NamedMetadata[];
  diagnostics: NamedMetadata[];
}

export interface RecordingState {
  active: boolean;
  source: string | null;
  field: string | null;
  path: string | null;
  encoded: number;
  dropped: number;
  status: string;
  error?: string | null;
}

export const idleRecording = (): RecordingState => ({
  active: false, source: null, field: null, path: null,
  encoded: 0, dropped: 0, status: 'idle',
});
export type DataRequest =
  | { what: 'topology'; topology: number }
  | { what: 'planes'; frame: number }
  | { what: 'diagnostics'; names?: string[] }
  | { what: 'steplog' };

export type DataResponse = {
  type: DataRequest['what'];
  request: number | null;
  [key: string]: unknown;
};

export type ServerMessage =
  | { type: 'hello'; protocol: number; runs: RunSummary[]; frames: FrameSummary[]; recording: RecordingState; theme?: 'light' | 'dark' | 'system' }
  | { type: 'state'; run_state: string; time: number | null; step: number | null; dt: number | null; store_bytes: number; recording: RecordingState;
      store_budget?: number | null; frame_count?: number | null; steps_per_second?: number | null; stopped?: boolean; t_final?: number | null }
  | ({ type: 'frame' } & FrameSummary)
  | { type: 'frames'; frames: FrameSummary[] }
  | DataResponse
  | { type: 'error'; message: string; request?: number | null };

export type ControlAction = 'pause' | 'resume' | 'stop' | 'keyframe';
export type FieldEncoding = 'f32' | 'u16';
export type Phase = 'liquid' | 'gas';
export type RecordingCommand =
  | { type: 'recording'; action: 'start'; source: 'live' | 'stored'; path: string; field: string; diagnostics: string[] }
  | { type: 'recording'; action: 'stop' };
