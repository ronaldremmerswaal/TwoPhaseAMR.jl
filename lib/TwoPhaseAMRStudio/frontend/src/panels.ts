import type { StudioClient } from './client';
import type { StudioState } from './state';

export interface PanelContext {
  state: StudioState;
  client: StudioClient;
  panelId: string;
}

export interface PanelInstance {
  element: HTMLElement;
  dispose?: () => void;
}

/** Register a real panel with a unique component name; instances may have different IDs. */
export interface PanelDefinition {
  id: string;
  title: string;
  create: (context: PanelContext) => PanelInstance;
}

export class PanelRegistry {
  private definitions = new Map<string, PanelDefinition>();
  private listeners = new Set<() => void>();

  register(definition: PanelDefinition): void {
    if (!definition.id || !definition.title || typeof definition.create !== 'function') throw new Error('A panel requires an id, title and create function');
    if (this.definitions.has(definition.id)) throw new Error(`Panel already registered: ${definition.id}`);
    this.definitions.set(definition.id, definition);
    for (const listener of this.listeners) listener();
  }

  get(id: string): PanelDefinition | undefined { return this.definitions.get(id); }
  all(): PanelDefinition[] { return [...this.definitions.values()]; }
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

function addRow(list: HTMLDListElement, name: string, value: string): void {
  const term = document.createElement('dt');
  const description = document.createElement('dd');
  term.textContent = name;
  description.textContent = value;
  list.append(term, description);
}

export function formatNumber(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return Number(value.toPrecision(6)).toString();
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index++; }
  return `${formatNumber(value)} ${units[index]}`;
}

export const runPanel: PanelDefinition = {
  id: 'run', title: 'Run',
  create({ state }) {
    const element = document.createElement('section');
    element.className = 'panel-content';
    element.dataset.testid = 'run-panel';
    element.setAttribute('aria-label', 'Run details');
    const render = () => {
      const heading = document.createElement('h2');
      heading.textContent = state.runs[0]?.name || 'Run';
      const list = document.createElement('dl');
      list.className = 'run-summary';
      const run = state.runs[0];
      addRow(list, 'State', state.runState);
      addRow(list, 'Time', run ? `${formatNumber(state.time)} / ${formatNumber(run.t_final)}` : formatNumber(state.time));
      addRow(list, 'Step', state.step === null ? '—' : String(state.step));
      addRow(list, 'Δt', formatNumber(state.dt));
      addRow(list, 'Store', `${formatBytes(state.storeBytes)} · ${state.timeline.frames.length} frames`);
      if (run) {
        addRow(list, 'Domain', JSON.stringify(run.domain) ?? '—');
        addRow(list, 'Block size', JSON.stringify(run.block_size) ?? '—');
        addRow(list, 'Fields', run.fields.map(field => field.label || field.name).join(', ') || 'None');
        addRow(list, 'Diagnostics', run.diagnostics.map(diag => diag.label || diag.name).join(', ') || 'None');
      }
      const current = state.timeline.current;
      if (current) addRow(list, 'Viewed frame', `#${current.id} · ${current.kind} · t=${formatNumber(current.time)}`);
      element.replaceChildren(heading, list);
    };
    const dispose = state.subscribe(render);
    render();
    return { element, dispose };
  },
};

export const framesPanel: PanelDefinition = {
  id: 'frames', title: 'Frames',
  create({ state }) {
    const element = document.createElement('section');
    element.className = 'panel-content frames-panel';
    element.dataset.testid = 'frames-panel';
    element.setAttribute('aria-label', 'Stored frames');
    let previousSignature = '';
    const render = () => {
      const signature = JSON.stringify([state.timeline.selectedId, state.timeline.frames.map(({ id, time, step, kind, topology }) => [id, time, step, kind, topology])]);
      if (signature === previousSignature) return;
      previousSignature = signature;
      const active = document.activeElement;
      const focusedFrame = active instanceof HTMLButtonElement && element.contains(active) ? active.dataset.frameId : undefined;
      const scrollTop = element.scrollTop;
      const heading = document.createElement('h2');
      heading.textContent = `Stored frames (${state.timeline.frames.length})`;
      const description = document.createElement('p');
      description.textContent = 'Select a frame to inspect its time and metadata. Viewing history does not change recording.';
      const list = document.createElement('div');
      list.className = 'frame-list';
      if (!state.timeline.frames.length) list.textContent = 'No stored frames yet.';
      for (const [index, frame] of state.timeline.frames.entries()) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'frame-row';
        button.dataset.testid = 'frame-row';
        button.dataset.frameId = String(frame.id);
        button.setAttribute('aria-pressed', String(frame.id === state.timeline.selectedId));
        button.textContent = `#${frame.id}  ·  t=${formatNumber(frame.time)}  ·  step ${frame.step}  ·  ${frame.kind}  ·  topology ${frame.topology}`;
        button.addEventListener('click', () => state.navigate(timeline => { timeline.playing = false; timeline.seek(index); }));
        list.append(button);
      }
      element.replaceChildren(heading, description, list);
      element.scrollTop = scrollTop;
      if (focusedFrame !== undefined) {
        [...list.querySelectorAll<HTMLButtonElement>('button')].find(button => button.dataset.frameId === focusedFrame)?.focus({ preventScroll: true });
      }
    };
    const dispose = state.subscribe(render);
    render();
    return { element, dispose };
  },
};
