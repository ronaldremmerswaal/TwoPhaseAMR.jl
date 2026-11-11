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
  progress?: number | null;
  url?: string | null;
  times?: number[] | null;
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

export interface RecordingSchedule {
  frames: number;
  distinct: number;
  repeats: number;
  sparse: boolean;
  window: [number, number];
  fps: number;
  duration: number;
  spacing: number;
  sim_per_second: number;
  first_frame: FrameId;
}

export type ServerMessage =
  | { type: 'hello'; protocol: number; runs: RunSummary[]; frames: FrameSummary[]; recording: RecordingState; theme?: 'light' | 'dark' | 'system' }
  | { type: 'state'; run_state: string; time: number | null; step: number | null; dt: number | null; store_bytes: number; recording: RecordingState;
      store_budget?: number | null; frame_count?: number | null; steps_per_second?: number | null; stopped?: boolean; t_final?: number | null;
      resumable?: boolean; suggested_final_time?: number | null }
  | ({ type: 'frame' } & FrameSummary)
  | { type: 'frames'; frames: FrameSummary[] }
  | DataResponse
  | { type: 'recording_preview'; request: number; url: string }
  | ({ type: 'recording_schedule'; request: number } & RecordingSchedule)
  | { type: 'error'; message: string; request?: number | null };

export type ControlAction = 'pause' | 'resume' | 'stop' | 'keyframe';
export type FieldEncoding = 'f32' | 'u16';
export type Phase = 'liquid' | 'gas';
export interface DiagnosticExportOptions {
  names: string[];
  scale: 'linear' | 'log';
  hidden: { name: string; component: number }[];
}
export interface StoredExportOptions {
  source: 'stored';
  field: string;
  diagnostics: string[];
  frames: 'keyframes' | 'all';
  window?: [number, number];
  camera: 'fit' | [number, number, number, number];
  colorrange: 'global' | [number, number];
  colormap?: string;
  phase_rendering: 'cell' | 'plic';
  size: [number, number];
  fps: number;
  duration: number;
  theme: 'light' | 'dark';
  arrangement: 'right' | 'below';
  interface: boolean;
  geometry: boolean;
  blocks: boolean;
  diagnostic_options?: DiagnosticExportOptions[];
}

export interface LiveExportOptions {
  source: 'live';
  field: string;
  diagnostics: string[];
  camera: 'fit' | [number, number, number, number];
  colorrange?: [number, number];
  colormap?: string;
  phase_rendering: 'cell' | 'plic';
  size: [number, number];
  fps: number;
  sim_per_second: number;
  theme: 'light' | 'dark';
  arrangement: 'right' | 'below';
  interface: boolean;
  geometry: boolean;
  blocks: boolean;
  diagnostic_options?: DiagnosticExportOptions[];
}

export type ScheduleCommand = {
  type: 'recording';
  action: 'schedule';
  source: 'stored';
  field: string;
  diagnostics: string[];
  frames: 'keyframes' | 'all';
  window?: [number, number];
  fps: number;
  duration: number;
};

export type RecordingCommand =
  | ({ type: 'recording'; action: 'export'; path: string } & StoredExportOptions)
  | ({ type: 'recording'; action: 'start'; path: string } & LiveExportOptions)
  | ScheduleCommand
  | { type: 'recording'; action: 'stop' | 'cancel' };

export type PreviewCommand = ({ type: 'recording'; action: 'preview'; frame: FrameId } & StoredExportOptions);
