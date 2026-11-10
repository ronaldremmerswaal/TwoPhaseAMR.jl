import type { StudioClient } from './client';
import type { StudioState } from './state';
import type { FrameId } from './types';
import { runStatePresentation } from './state';

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

export function formatNumber(value: number | null | undefined, digits = 6): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return Number(value.toPrecision(digits)).toString();
}

/** Compact status bar numbers, preserving scientific notation for small steps. */
export function formatStatusNumber(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? Number(value.toPrecision(4)).toString() : '—';
}

/** Display the solver throughput rounded to three significant figures. */
export function formatRate(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? Number(value.toPrecision(3)).toString() : '—';
}

export function formatCount(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? String(value) : '—';
}

export function formatBytes(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let index = 0;
  while (value >= 1000 && index < units.length - 1) { value /= 1000; index++; }
  return `${index === 0 ? formatNumber(value) : Number(value.toPrecision(3))} ${units[index]}`;
}

export function formatStore(bytes: number | null, budget: number | null, count: number | null): string {
  const frames = formatCount(count);
  return `${formatBytes(bytes)} / ${formatBytes(budget)} (${frames} ${count === 1 ? 'frame' : 'frames'})`;
}

export function formatDomain(value: unknown): string {
  if (!value || typeof value !== 'object') return '—';
  const domain = value as { origin?: unknown; extent?: unknown };
  if (!Array.isArray(domain.origin) || !Array.isArray(domain.extent) ||
      domain.origin.length !== 2 || domain.extent.length !== 2 ||
      ![...domain.origin, ...domain.extent].every(item => typeof item === 'number' && Number.isFinite(item))) return '—';
  const [x, y] = domain.origin as number[];
  const [width, height] = domain.extent as number[];
  return `[${formatNumber(x)}, ${formatNumber(x + width)}] × [${formatNumber(y)}, ${formatNumber(y + height)}]`;
}

export function formatBlockSize(value: unknown): string {
  if (!Array.isArray(value) || value.length !== 2 ||
      !value.every(item => typeof item === 'number' && Number.isSafeInteger(item) && item > 0)) return '—';
  return value.join(' × ');
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
      addRow(list, 'State', runStatePresentation(state.runState, state.stopped).label);
      addRow(list, 'Time', `${formatNumber(state.time)} / ${formatNumber(state.finalTime)}`);
      addRow(list, 'Step', formatCount(state.step));
      addRow(list, 'Δt', formatNumber(state.dt));
      addRow(list, 'Speed', `${formatRate(state.stepsPerSecond)} steps/s`);
      addRow(list, 'Store', formatStore(state.storeBytes, state.storeBudget, state.frameCount));
      if (run) {
        addRow(list, 'Domain', formatDomain(run.domain));
        addRow(list, 'Block size', formatBlockSize(run.block_size));
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
    const heading = document.createElement('h2');
    const description = document.createElement('p');
    description.textContent = 'Select a frame to inspect its time and metadata. Viewing history does not change recording.';
    const empty = document.createElement('p');
    empty.textContent = 'No stored frames yet.';
    const list = document.createElement('div');
    list.className = 'frame-list';
    element.append(heading, description, empty, list);
    const buttons = new Map<FrameId, HTMLButtonElement>();
    let previousSignature = '';
    const render = () => {
      const signature = JSON.stringify([state.timeline.selectedId, state.timeline.frames.map(({ id, time, step, kind, topology }) => [id, time, step, kind, topology])]);
      if (signature === previousSignature) return;
      previousSignature = signature;
      const active = document.activeElement;
      const focused = active instanceof HTMLButtonElement && list.contains(active) ? active : null;
      const scrollTop = element.scrollTop;
      heading.textContent = `Stored frames (${state.timeline.frames.length})`;
      empty.hidden = state.timeline.frames.length !== 0;
      const current = new Set(state.timeline.frames.map(frame => frame.id));
      for (const [id, button] of buttons) {
        if (!current.has(id)) { list.removeChild(button); buttons.delete(id); }
      }
      for (const [index, frame] of state.timeline.frames.entries()) {
        let button = buttons.get(frame.id);
        if (!button) {
          button = document.createElement('button');
          button.type = 'button';
          button.className = 'frame-row';
          button.dataset.testid = 'frame-row';
          button.dataset.frameId = String(frame.id);
          button.addEventListener('click', () => {
            const currentIndex = state.timeline.frames.findIndex(item => item.id === frame.id);
            if (currentIndex >= 0) state.navigate(timeline => { timeline.playing = false; timeline.seek(currentIndex); });
          });
          buttons.set(frame.id, button);
        }
        const pressed = String(frame.id === state.timeline.selectedId);
        if (button.getAttribute('aria-pressed') !== pressed) button.setAttribute('aria-pressed', pressed);
        const label = `#${frame.id}  ·  t=${formatNumber(frame.time)}  ·  step ${frame.step}  ·  ${frame.kind}  ·  topology ${frame.topology}`;
        if (button.textContent !== label) button.textContent = label;
        if (list.children[index] !== button) list.insertBefore(button, list.children[index] ?? null);
      }
      if (element.scrollTop !== scrollTop) element.scrollTop = scrollTop;
      if (focused && list.contains(focused) && document.activeElement !== focused) focused.focus({ preventScroll: true });
    };
    const dispose = state.subscribe(render);
    render();
    return { element, dispose };
  },
};
