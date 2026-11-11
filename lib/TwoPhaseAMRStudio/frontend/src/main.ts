import 'dockview/dist/styles/dockview.css';
import './style.css';
import './continue.css';
// Before anything that imports uPlot, which formats ticks with the reported
// browser locale while it loads.
import './locale';
import { createDockview, themeDark, themeLight, type DockviewApi } from 'dockview';
import { StudioClient } from './client';
import { groupDiagnostics, type DiagnosticGroup, type DiagnosticMetadata } from './diagnostic-data';
import { diagnosticPanel, type DiagnosticPanelDefinition } from './diagnostic-panel';
import { installExportDialog } from './export-dialog';
import { createFieldPanel, type FieldViewConfiguration } from './field-panel';
import { SharedRenderer } from './renderer';
import { framesPanel, formatCount, formatNumber, formatRate, formatStatusNumber, formatStore, PanelRegistry, runPanel, type PanelDefinition } from './panels';
import { continuationControlsEnabled, runStatePresentation, solverControlsEnabled, StudioState, validContinuationTarget } from './state';
import { PLAYBACK_SPEEDS, timelineBounds, timelineMarkers, timelinePosition } from './timeline';

const root = document.querySelector<HTMLElement>('#app');
if (!root) throw new Error('Missing Studio root element');
root.innerHTML = `
<div class="studio-shell">
  <header class="studio-header">
    <div class="header-row status-row">
      <h1>TwoPhaseAMR Studio</h1>
      <button type="button" class="run-name" data-testid="run-name" title="Open run summary">—</button>
      <span class="run-status" data-tone="neutral"><strong data-testid="run-state">UNKNOWN</strong></span>
      <span id="run-clock" data-testid="run-clock">t = — / — · step — · last dt — · — steps/s</span>
      <span class="connection" data-testid="connection"><span data-testid="connection-status" role="status">Connecting</span></span>
    </div>
    <div class="header-row action-row">
      <span id="store-size" data-testid="store-size">store — / — (— frames)</span>
      <div class="controls" aria-label="Solver controls">
        <button type="button" class="control-button" data-testid="control-pause" title="Pause or resume solver (P)"><span class="control-icon" aria-hidden="true">Ⅱ</span> <span class="control-label">Pause</span></button>
        <button type="button" class="control-button" data-testid="control-stop" title="Stop after the current step"><span aria-hidden="true">■</span> Stop</button>
        <button type="button" class="control-button" data-testid="control-keyframe" title="Capture keyframe (K)"><span aria-hidden="true">+</span> Keyframe</button>
        <form class="continue-controls" data-testid="continue-controls" aria-label="Continue run" novalidate>
          <label for="continue-target">Target time</label>
          <input id="continue-target" data-testid="continue-target" type="number" step="any" required aria-describedby="continue-reason">
          <span class="continue-button-wrap" data-testid="continue-availability"><button type="submit" class="control-button" data-testid="control-continue" aria-describedby="continue-reason">Continue ▸</button></span>
          <small id="continue-reason" data-testid="continue-reason" role="status"></small>
        </form>
        <button type="button" id="reconnect" hidden>Reconnect</button>
      </div>
      <button type="button" class="control-button" data-testid="open-export">Export…</button>
      <details class="add-panel-menu">
        <summary data-testid="add-panel-menu" title="Open a dockable panel">+ Panel</summary>
        <div id="panel-options" class="panel-options" aria-label="Panel types"></div>
      </details>
      <details class="layout-menu">
        <summary>Layout</summary>
        <div class="layout-options">
          <button type="button" data-testid="layout-auto">Reset automatic layout</button>
          <p>Drag a tab to move or dock. Shift-drag a tab to float it.</p>
        </div>
      </details>
      <label class="theme-control">Theme <select data-testid="theme-select" aria-label="Theme"><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
      <details class="help"><summary title="Keyboard shortcuts (?)">Shortcuts</summary>
        <div>Space: playback · P: pause/resume solver · L: follow live · K: keyframe · ←/→: frame · Shift+←/→: keyframe · Home/End: first/last · [ / ]: speed · D: theme<br>Drag a tab to move or dock; Shift-drag to float.<br>Focused field: F fit · 1 cell zoom · C colour range · I interface · G blocks · E cells · Shift+drag box zoom · Esc cancel box</div>
      </details>
    </div>
    <div id="error" class="error" role="alert" hidden><span id="error-text"></span><button type="button" id="dismiss-error" aria-label="Dismiss error">Dismiss</button></div>
  </header>
  <main id="dock" aria-label="Dockable Studio panels"></main>
  <footer class="timeline" aria-label="Timeline controls">
    <div class="timeline-controls">
      <button type="button" data-testid="timeline-prev-keyframe" aria-label="Previous keyframe" title="Previous keyframe (Shift+Left)">|◀</button>
      <button type="button" data-testid="timeline-prev" aria-label="Step backward" title="Previous frame (Left)">◀</button>
      <button type="button" data-testid="timeline-play" aria-label="Play" title="Play or pause playback (Space)">Play</button>
      <button type="button" data-testid="timeline-next" aria-label="Step forward" title="Next frame (Right)">▶</button>
      <button type="button" data-testid="timeline-next-keyframe" aria-label="Next keyframe" title="Next keyframe (Shift+Right)">▶|</button>
      <label>Speed <select id="playback-speed" aria-label="Playback speed"></select></label>
      <label><input type="checkbox" id="playback-loop"> Loop</label>
      <span data-testid="timeline-frame"><span data-testid="timeline-current">No frame</span></span>
      <strong class="timeline-mode" data-testid="timeline-mode" data-mode="live">LIVE</strong>
      <span data-testid="timeline-new-count" aria-live="polite" hidden></span>
      <button type="button" data-testid="timeline-follow" title="Follow live (L)">Follow →</button>
      <button type="button" id="refresh-frames" title="Refresh the stored frame index">Refresh frames</button>
    </div>
    <div class="timeline-track">
      <div class="timeline-markers" aria-hidden="true" data-testid="timeline-markers"></div>
      <input data-testid="timeline-scrub" aria-label="Timeline" type="range" min="0" max="1" value="0" step="any" disabled>
    </div>
    <div class="timeline-ends"><span id="timeline-start">—</span><span id="timeline-middle">—</span><span id="timeline-end">—</span></div>
    <div class="timeline-legend" aria-hidden="true"><span class="legend-keyframe">keyframe</span><span class="legend-regrid">R regrid</span><span class="legend-playhead">playhead</span><span class="legend-head">live head</span></div>
  </footer>
</div>`;

function find<T extends HTMLElement>(selector: string): T {
  const element = root!.querySelector<T>(selector);
  if (!element) throw new Error(`Missing Studio element: ${selector}`);
  return element;
}

const state = new StudioState();
const client = new StudioClient(state);
const registry = new PanelRegistry();
registry.register(runPanel);
registry.register(framesPanel);
const dockElement = find<HTMLElement>('#dock');
const renderer = new SharedRenderer();
const initialFields: Record<string, string> = {};
const fieldPanels = createFieldPanel(renderer, {
  initialFields,
  onViewChange(view) {
    const panel = dock.getPanel(view.panelId);
    if (panel && panel.api.title !== view.title) panel.api.setTitle(view.title);
    window.dispatchEvent(new Event('studio-view-change'));
  },
});
registry.register(fieldPanels);
const panelOptions = find<HTMLElement>('#panel-options');
const themePicker = find<HTMLSelectElement>('[data-testid="theme-select"]');
const darkPreference = window.matchMedia('(prefers-color-scheme: dark)');
let initialTheme: string | null = null;
try { initialTheme = localStorage.getItem('two-phase-amr-studio-theme'); } catch { /* Storage is optional. */ }
let themeChosen = initialTheme === 'light' || initialTheme === 'dark' || initialTheme === 'system';
themePicker.value = initialTheme === 'dark' || initialTheme === 'light' ? initialTheme : 'system';

const dock: DockviewApi = createDockview(dockElement, {
  theme: themePicker.value === 'dark' || (themePicker.value === 'system' && darkPreference.matches) ? themeDark : themeLight,
  createComponent({ name, id }) {
    const definition = registry.get(name);
    if (!definition) throw new Error(`Unknown panel: ${name}`);
    const instance = definition.create({ state, client, panelId: id });
    return { element: instance.element, init() {}, dispose() { instance.dispose?.(); } };
  },
  createWatermarkComponent() {
    const element = document.createElement('p');
    element.className = 'dock-empty';
    element.textContent = 'No panels open. Use + Panel in the status bar to open one.';
    return { element, init() {} };
  },
});

export interface OpenPanelOptions {
  id?: string;
  title?: string;
  referencePanel?: string;
  direction?: 'left' | 'right' | 'above' | 'below' | 'within';
}

/**
 * Available as window.twoPhaseAMRStudio after startup. Extensions register a
 * component with { id, title, create({ state, client, panelId }) }, returning
 * { element, dispose? }. Subscribe with state.subscribe() and release the
 * subscription in dispose. client.getField() decodes fields; client.getData()
 * requests topology, phase planes, diagnostics, and the step log.
 * openPanel(component, { id, referencePanel, direction }) permits multiple
 * independently closable instances. The panel picker reopens registered types.
 */
export interface StudioExtensionAPI {
  readonly state: StudioState;
  readonly client: StudioClient;
  registerPanel(definition: PanelDefinition): void;
  openPanel(component: string, options?: OpenPanelOptions): void;
  getFieldViews(): FieldViewConfiguration[];
}

function openPanel(component: string, options: OpenPanelOptions = {}): void {
  const definition = registry.get(component);
  if (!definition) throw new Error(`Unknown panel: ${component}`);
  const id = options.id ?? component;
  const existing = dock.getPanel(id);
  if (existing) { existing.api.setActive(); return; }
  const reference = options.referencePanel ?? dock.panels[0]?.id;
  const position = reference ? { referencePanel: reference, direction: options.direction ?? 'within' as const } : undefined;
  dock.addPanel({ id, component, title: options.title ?? definition.title, ...(position ? { position } : {}) });
}

const extensionAPI: StudioExtensionAPI = {
  state, client,
  registerPanel(definition) { registry.register(definition); },
  openPanel,
  getFieldViews() { return fieldPanels.getViews(); },
};
declare global {
  interface Window {
    twoPhaseAMRStudio: StudioExtensionAPI;
  }
}
window.twoPhaseAMRStudio = extensionAPI;

let panelSequence = 0;
function updatePanelPicker(): void {
  const definitions = registry.all();
  definitions.sort((a, b) => {
    const rank = (id: string) => id === fieldPanels.id ? 0 : id.startsWith('diagnostic-') ? 1 : 2;
    return rank(a.id) - rank(b.id);
  });
  panelOptions.replaceChildren(...definitions.map(definition => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.testid = `panel-option-${definition.id}`;
    button.textContent = definition.title;
    button.addEventListener('click', () => {
      let id: string;
      do { id = `${definition.id}-added-${++panelSequence}`; } while (dock.getPanel(id));
      openPanel(definition.id, { id, referencePanel: dock.activePanel?.id, direction: 'right' });
      find<HTMLDetailsElement>('.add-panel-menu').open = false;
    });
    return button;
  }));
}
registry.subscribe(updatePanelPicker);
updatePanelPicker();
let orientation: 'right' | 'bottom' = window.innerWidth / window.innerHeight > 1.6 ? 'right' : 'bottom';
let automaticLayout = true;
let reflowing = false;
let layoutInitialized = false;
let diagnosticIds: string[] = [];
const diagnosticDefinitions = new Map<string, { id: string; group: DiagnosticGroup; definition: DiagnosticPanelDefinition }>();

function configureLayoutMetadata(): void {
  const run = state.runs[0];
  if (!run) return;
  const fields = run.fields;
  const fraction = fields.find(field => field.name === 'volume_fraction') ??
    fields.find(field => field.name.includes('volume_fraction'));
  const twoPhase = !!fraction || fields.some(field => typeof field.interface === 'number' && field.interface > 0);
  const preferred = twoPhase
    ? [fraction?.name, 'velocity_magnitude']
    : [fields.find(field => field.name === 'velocity_vorticity' || field.name === 'vorticity')?.name, 'pressure'];
  const chosen = preferred.filter((name): name is string => !!name && fields.some(field => field.name === name));
  for (const field of fields) {
    if (chosen.length >= 2) break;
    if (!chosen.includes(field.name)) chosen.push(field.name);
  }
  initialFields['field-primary'] = chosen[0] ?? '';
  initialFields['field-secondary'] = chosen[1] ?? '';
  diagnosticIds = groupDiagnostics(run.diagnostics as DiagnosticMetadata[]).map(group => {
    const existing = diagnosticDefinitions.get(group.key);
    if (existing) {
      existing.group.names.splice(0, existing.group.names.length, ...group.names);
      return existing.id;
    }
    const id = `diagnostic-${diagnosticDefinitions.size}`;
    const definition = diagnosticPanel(group, id);
    diagnosticDefinitions.set(group.key, { id, group, definition });
    registry.register(definition);
    return id;
  });
}

function buildAutomaticLayout(): void {
  reflowing = true;
  try {
    dock.clear();
    orientation = window.innerWidth / window.innerHeight > 1.6 ? 'right' : 'bottom';
    const first = initialFields['field-primary'];
    const second = initialFields['field-secondary'];
    const firstTitle = state.runs[0]?.fields.find(field => field.name === first)?.label || first;
    const secondTitle = state.runs[0]?.fields.find(field => field.name === second)?.label || second;
    const domain = state.runs[0]?.domain;
    const extent = domain && typeof domain === 'object' && 'extent' in domain ? domain.extent : null;
    const validExtent = Array.isArray(extent) && typeof extent[0] === 'number' && typeof extent[1] === 'number' &&
      Number.isFinite(extent[0]) && Number.isFinite(extent[1]) && extent[0] > 0 && extent[1] > 0;
    const tall = validExtent && extent[1] / extent[0] > 1.5;
    const diagnosticShare = tall ? 0.25 : 1 / 3;
    let fieldDirection: 'right' | 'below' = 'right';
    if (validExtent && dockElement.clientWidth > 0 && dockElement.clientHeight > 0) {
      const width = dockElement.clientWidth * (orientation === 'right' && diagnosticIds.length ? 1 - diagnosticShare : 1);
      const height = dockElement.clientHeight * (orientation === 'bottom' && diagnosticIds.length ? 0.5 : 1);
      const sideBySide = Math.min(width / (2 * extent[0]), height / extent[1]);
      const stacked = Math.min(width / extent[0], height / (2 * extent[1]));
      if (stacked > sideBySide) fieldDirection = 'below';
    }
    let reference: string | undefined;
    let previous: string | undefined;
    if (first) {
      openPanel(fieldPanels.id, { id: 'field-primary', title: firstTitle });
      reference = 'field-primary';
      // Create the diagnostics region first when it crosses the field split,
      // so it spans both field panels instead of only the second one.
      if (diagnosticIds.length && ((orientation === 'bottom' && fieldDirection === 'right') ||
          (orientation === 'right' && fieldDirection === 'below'))) {
        openPanel(diagnosticIds[0], { referencePanel: reference, direction: orientation === 'right' ? 'right' : 'below' });
        previous = diagnosticIds[0];
      }
      if (second && second !== first) {
        openPanel(fieldPanels.id, { id: 'field-secondary', title: secondTitle, referencePanel: reference, direction: fieldDirection });
        reference = 'field-secondary';
      }
    }
    diagnosticIds.forEach((id, index) => {
      if (index === 0 && previous) return;
      openPanel(id, { referencePanel: previous ?? reference, direction: previous ? 'below' : orientation === 'right' ? 'right' : 'below' });
      previous = id;
    });
    const anchor = diagnosticIds[0] ?? reference;
    openPanel('run', { referencePanel: anchor, direction: anchor && diagnosticIds.length ? 'within' : orientation === 'right' ? 'right' : 'below' });
    openPanel('frames', { referencePanel: 'run', direction: 'within' });
    dock.getPanel(diagnosticIds[0] ?? 'run')?.api.setActive();
    if (orientation === 'right' && first && second && diagnosticIds.length) {
      dock.getPanel(diagnosticIds[0])?.group.api.setSize({ width: dockElement.clientWidth * diagnosticShare });
      if (fieldDirection === 'right') {
        dock.getPanel('field-primary')?.group.api.setSize({ width: dockElement.clientWidth * (1 - diagnosticShare) / 2 });
      }
    }
    automaticLayout = true;
    layoutInitialized = true;
    renderer.requestRender();
  } finally {
    reflowing = false;
  }
}

openPanel('run');
openPanel('frames', { referencePanel: 'run', direction: orientation === 'right' ? 'right' : 'below' });
dock.onDidDrop(() => { automaticLayout = false; });
dock.onDidAddPanel(() => { if (!reflowing) automaticLayout = false; });
dock.onDidRemovePanel(() => { if (!reflowing) automaticLayout = false; });
dockElement.addEventListener('pointerdown', event => {
  if (event.target instanceof Element && event.target.closest('.dv-sash')) automaticLayout = false;
});
dockElement.addEventListener('pointerover', event => {
  const tab = event.target instanceof Element ? event.target.closest<HTMLElement>('.dv-tab') : null;
  if (tab && tab.title !== 'Drag to move or dock. Shift-drag to float.') tab.title = 'Drag to move or dock. Shift-drag to float.';
});
window.addEventListener('resize', () => {
  const next = window.innerWidth / window.innerHeight > 1.6 ? 'right' : 'bottom';
  if (!automaticLayout || next === orientation) return;
  fieldPanels.restoreViews(fieldPanels.getViews());
  buildAutomaticLayout();
});
find<HTMLButtonElement>('[data-testid="layout-auto"]').addEventListener('click', () => {
  configureLayoutMetadata();
  buildAutomaticLayout();
  find<HTMLDetailsElement>('.layout-menu').open = false;
});
find<HTMLButtonElement>('[data-testid="run-name"]').addEventListener('click', () => {
  const run = dock.panels.find(panel => panel.id === 'run' || panel.id.startsWith('run-added-'));
  if (run) run.api.setActive();
  else openPanel('run', { referencePanel: dock.activePanel?.id, direction: 'within' });
});
installExportDialog(find<HTMLButtonElement>('[data-testid="open-export"]'), state, client, () =>
  fieldPanels.getViews().map(view => ({
    id: view.panelId, title: view.title, field: view.field, frame: view.frame, mode: view.mode,
    camera: view.worldRect, colorrange: view.colorRange, colormap: view.colormap, ready: view.ready, overlays: view.overlays,
  })), () => [...diagnosticDefinitions.values()].flatMap(item => item.definition.getConfigs()));

function applyTheme(): void {
  const dark = themePicker.value === 'dark' || (themePicker.value === 'system' && darkPreference.matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  dock.updateOptions({ theme: dark ? themeDark : themeLight });
  renderer.requestRender();
}
themePicker.addEventListener('change', () => {
  themeChosen = true;
  try { localStorage.setItem('two-phase-amr-studio-theme', themePicker.value); } catch { /* Storage is optional. */ }
  applyTheme();
});
darkPreference.addEventListener('change', applyTheme);
applyTheme();

const pauseButton = find<HTMLButtonElement>('[data-testid="control-pause"]');
const stopButton = find<HTMLButtonElement>('[data-testid="control-stop"]');
const keyframeButton = find<HTMLButtonElement>('[data-testid="control-keyframe"]');
const continueForm = find<HTMLFormElement>('[data-testid="continue-controls"]');
const continueTarget = find<HTMLInputElement>('[data-testid="continue-target"]');
const continueButton = find<HTMLButtonElement>('[data-testid="control-continue"]');
const continueAvailability = find<HTMLElement>('[data-testid="continue-availability"]');
const continueReason = find<HTMLElement>('#continue-reason');
let suggestionKey: string | null = null;
let continuationPending: { previousFinalTime: number | null; previousTime: number | null; previousStep: number | null } | null = null;
const reconnectButton = find<HTMLButtonElement>('#reconnect');
const slider = find<HTMLInputElement>('[data-testid="timeline-scrub"]');
const playButton = find<HTMLButtonElement>('[data-testid="timeline-play"]');
const speedPicker = find<HTMLSelectElement>('#playback-speed');
const loopCheckbox = find<HTMLInputElement>('#playback-loop');
for (const speed of PLAYBACK_SPEEDS) {
  const option = document.createElement('option');
  option.value = String(speed);
  option.textContent = `${speed}×`;
  speedPicker.append(option);
}

pauseButton.addEventListener('click', () => client.control(state.runState === 'paused' || state.runState === 'pause_requested' ? 'resume' : 'pause'));
stopButton.addEventListener('click', () => client.control('stop'));
keyframeButton.addEventListener('click', () => client.control('keyframe'));
continueTarget.addEventListener('input', () => render());
continueForm.addEventListener('submit', event => {
  event.preventDefault();
  const target = continueTarget.valueAsNumber;
  if (!continuationControlsEnabled(state.connection, state.runState, state.resumable) ||
      continuationPending || !validContinuationTarget(target, state.time)) return;
  if (state.error) state.clearError();
  continuationPending = { previousFinalTime: state.finalTime, previousTime: state.time, previousStep: state.step };
  if (!client.continueRun(target)) continuationPending = null;
  render();
});
reconnectButton.addEventListener('click', () => client.connect());
find<HTMLButtonElement>('#dismiss-error').addEventListener('click', () => state.clearError());
find<HTMLButtonElement>('#refresh-frames').addEventListener('click', () => client.getFrames());
find<HTMLButtonElement>('[data-testid="timeline-follow"]').addEventListener('click', () => state.navigate(timeline => timeline.follow()));
find<HTMLButtonElement>('[data-testid="timeline-prev"]').addEventListener('click', () => state.navigate(timeline => timeline.step(-1)));
find<HTMLButtonElement>('[data-testid="timeline-next"]').addEventListener('click', () => state.navigate(timeline => timeline.step(1)));
find<HTMLButtonElement>('[data-testid="timeline-prev-keyframe"]').addEventListener('click', () => state.navigate(timeline => timeline.jumpKeyframe(-1)));
find<HTMLButtonElement>('[data-testid="timeline-next-keyframe"]').addEventListener('click', () => state.navigate(timeline => timeline.jumpKeyframe(1)));
playButton.addEventListener('click', () => state.navigate(timeline => timeline.togglePlay()));
slider.addEventListener('input', () => {
  const bounds = timelineBounds(state.timeline.frames, state.time, state.finalTime, state.runs[0]?.t0 ?? null);
  if (!bounds) return;
  state.navigate(timeline => {
    timeline.playing = false;
    timeline.seekTime(bounds.start + Number(slider.value) * (bounds.end - bounds.start));
  });
});
slider.addEventListener('pointermove', event => {
  const frames = state.timeline.frames;
  const bounds = timelineBounds(frames, state.time, state.finalTime, state.runs[0]?.t0 ?? null);
  const rect = slider.getBoundingClientRect();
  if (!bounds || !frames.length || rect.width <= 0) { slider.title = ''; return; }
  const position = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  const time = bounds.start + position * (bounds.end - bounds.start);
  const frame = frames.reduce((best, item) => Math.abs(item.time - time) < Math.abs(best.time - time) ? item : best);
  slider.title = `Frame #${frame.id} · t=${formatNumber(frame.time)} · ${frame.kind}`;
});
slider.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
    state.navigate(timeline => event.shiftKey ? timeline.jumpKeyframe(-1) : timeline.step(-1));
  } else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
    state.navigate(timeline => event.shiftKey ? timeline.jumpKeyframe(1) : timeline.step(1));
  } else if (event.key === 'Home' || event.key === 'End') {
    state.navigate(timeline => timeline.edge(event.key === 'Home' ? 'first' : 'last'));
  } else return;
  event.preventDefault();
});
speedPicker.addEventListener('change', () => state.navigate(timeline => { timeline.speed = Number(speedPicker.value); }));
loopCheckbox.addEventListener('change', () => state.navigate(timeline => { timeline.loop = loopCheckbox.checked; }));

let metadataSignature = '';
let playbackTimer: number | undefined;
let markerSignature = '';
let timerSpeed = 0;
function render(): void {
  if (!themeChosen && state.preferredTheme) {
    themePicker.value = state.preferredTheme;
    themeChosen = true;
    applyTheme();
  }
  const connected = state.connection === 'connected';
  const run = state.runs[0];
  const presentation = runStatePresentation(state.runState, state.stopped);
  find<HTMLElement>('[data-testid="connection-status"]').textContent = state.connection;
  find<HTMLElement>('[data-testid="run-name"]').textContent = run?.name || '—';
  find<HTMLElement>('[data-testid="run-state"]').textContent = presentation.label;
  find<HTMLElement>('.run-status').dataset.tone = presentation.tone;
  find<HTMLElement>('#run-clock').textContent = `t = ${formatStatusNumber(state.time)} / ${formatStatusNumber(state.finalTime)} · step ${formatCount(state.step)} · last dt ${formatStatusNumber(state.dt)} · ${formatRate(state.stepsPerSecond)} steps/s`;
  find<HTMLElement>('#store-size').textContent = `store ${formatStore(state.storeBytes, state.storeBudget, state.frameCount)}`;
  const resuming = state.runState === 'paused' || state.runState === 'pause_requested';
  pauseButton.querySelector<HTMLElement>('.control-label')!.textContent = resuming ? 'Resume' : 'Pause';
  pauseButton.querySelector<HTMLElement>('.control-icon')!.textContent = resuming ? '▶' : 'Ⅱ';
  const canControl = solverControlsEnabled(state.connection, state.runState);
  pauseButton.disabled = !canControl;
  stopButton.disabled = !canControl;
  keyframeButton.disabled = !canControl;
  if (continuationPending && (!connected || state.error || state.runState !== 'completed' ||
      state.finalTime !== continuationPending.previousFinalTime || state.time !== continuationPending.previousTime ||
      state.step !== continuationPending.previousStep)) continuationPending = null;
  // A stopped run can target the same previous end, so the server suggestion
  // may stay unchanged even though the completed run needs a fresh draft.
  const completedAt = state.runState === 'completed' ? [state.time, state.step] : null;
  const nextSuggestionKey = JSON.stringify([state.finalTime, state.suggestedFinalTime, completedAt]);
  if (nextSuggestionKey !== suggestionKey) {
    suggestionKey = nextSuggestionKey;
    if (state.suggestedFinalTime !== null) continueTarget.value = String(state.suggestedFinalTime);
  }
  const canContinue = continuationControlsEnabled(state.connection, state.runState, state.resumable);
  const validTarget = validContinuationTarget(continueTarget.valueAsNumber, state.time);
  continueTarget.disabled = !canContinue || continuationPending !== null;
  continueButton.disabled = !canContinue || continuationPending !== null || !validTarget;
  continueTarget.setAttribute('aria-invalid', String(canContinue && !validTarget));
  const continuationHelp = !connected ? 'Connect to the run to continue.' :
    !state.resumable ? 'This run has no resumable simulation driver.' :
    state.runState !== 'completed' ? 'Continue is available after the run finishes or stops.' :
    continuationPending ? 'Waiting for the continuation request.' :
    !validTarget ? 'Enter a finite target later than the current simulation time.' : '';
  continueReason.textContent = continuationHelp;
  continueTarget.title = continuationHelp;
  continueAvailability.title = continuationHelp || 'Continue the run to the target time.';
  continueAvailability.tabIndex = continueButton.disabled ? 0 : -1;
  reconnectButton.hidden = connected || state.connection === 'connecting';
  find<HTMLButtonElement>('#refresh-frames').disabled = !connected;
  const errorBox = find<HTMLElement>('#error');
  errorBox.hidden = !state.error;
  find<HTMLElement>('#error-text').textContent = state.error ?? '';

  const fields = run?.fields ?? [];
  const diagnostics = run?.diagnostics ?? [];
  const signature = JSON.stringify([fields, diagnostics]);
  if (signature !== metadataSignature) {
    metadataSignature = signature;
    if (run) {
      configureLayoutMetadata();
      if (!layoutInitialized && automaticLayout) buildAutomaticLayout();
    }
  }

  const timeline = state.timeline;
  const current = timeline.current;
  const bounds = timelineBounds(timeline.frames, state.time, state.finalTime, run?.t0 ?? null);
  const mode = find<HTMLElement>('[data-testid="timeline-mode"]');
  mode.textContent = timeline.mode.toUpperCase();
  mode.dataset.mode = timeline.mode;
  const newCount = find<HTMLElement>('[data-testid="timeline-new-count"]');
  newCount.hidden = timeline.mode === 'live';
  newCount.textContent = `+${timeline.newCount} new`;
  find<HTMLElement>('[data-testid="timeline-current"]').textContent = current ? `#${current.id} · t = ${formatNumber(current.time)} · step ${formatCount(current.step)} · ${current.kind}` : 'No frame';
  slider.value = String(bounds && current ? timelinePosition(current.time, bounds) : 0);
  slider.disabled = !current || !bounds;
  slider.setAttribute('aria-valuetext', current ? `Frame ${current.id}, time ${formatNumber(current.time)}, ${current.kind}` : 'No frame');
  const markers = find<HTMLElement>('[data-testid="timeline-markers"]');
  const markerKey = JSON.stringify([bounds, timeline.frames.map(frame => [frame.id, frame.time, frame.kind, frame.topology])]);
  if (markerKey !== markerSignature) {
    markerSignature = markerKey;
    const fragment = document.createDocumentFragment();
    if (bounds) timelineMarkers(timeline.frames, bounds).forEach(marker => {
      const tick = document.createElement('span');
      tick.className = `timeline-marker ${marker.kind}`;
      tick.style.left = `${100 * marker.position}%`;
      if (marker.kind === 'regrid') tick.textContent = 'R';
      const label = marker.kind === 'regrid' ? 'Topology change' : marker.kind === 'keyframe' ? 'Keyframe' : 'Preview';
      tick.title = `${label} · ${formatNumber(marker.time)}`;
      fragment.append(tick);
    });
    const head = document.createElement('span');
    head.className = 'timeline-marker live-head';
    head.dataset.testid = 'timeline-live-head';
    fragment.append(head);
    const playhead = document.createElement('span');
    playhead.className = 'timeline-marker playhead';
    playhead.dataset.testid = 'timeline-playhead';
    fragment.append(playhead);
    markers.replaceChildren(fragment);
  }
  const head = markers.querySelector<HTMLElement>('.live-head')!;
  head.hidden = !bounds || state.time === null || !Number.isFinite(state.time);
  if (bounds && state.time !== null && Number.isFinite(state.time)) {
    head.style.left = `${100 * timelinePosition(state.time, bounds)}%`;
    head.title = `Live head · ${formatNumber(state.time)}`;
  }
  const playhead = markers.querySelector<HTMLElement>('.playhead')!;
  playhead.hidden = !bounds || !current;
  if (bounds && current) {
    playhead.style.left = `${100 * timelinePosition(current.time, bounds)}%`;
    playhead.title = `Playhead · ${formatNumber(current.time)}`;
  }
  find<HTMLElement>('#timeline-start').textContent = formatNumber(bounds?.start);
  find<HTMLElement>('#timeline-middle').textContent = bounds && bounds.start !== bounds.end ? formatNumber(bounds.start + (bounds.end - bounds.start) / 2) : '';
  find<HTMLElement>('#timeline-end').textContent = formatNumber(bounds?.end);
  for (const button of root!.querySelectorAll<HTMLButtonElement>('[data-testid="timeline-prev"], [data-testid="timeline-next"], [data-testid="timeline-prev-keyframe"], [data-testid="timeline-next-keyframe"], [data-testid="timeline-play"], [data-testid="timeline-follow"]')) button.disabled = !timeline.frames.length;
  playButton.textContent = timeline.playing ? 'Pause playback' : 'Play';
  playButton.setAttribute('aria-label', timeline.playing ? 'Pause playback' : 'Play');
  speedPicker.value = String(timeline.speed);
  loopCheckbox.checked = timeline.loop;
  if (playbackTimer !== undefined && (!timeline.playing || timerSpeed !== timeline.speed)) {
    window.clearTimeout(playbackTimer);
    playbackTimer = undefined;
  }
  if (timeline.playing && playbackTimer === undefined) {
    timerSpeed = timeline.speed;
    playbackTimer = window.setTimeout(() => {
      playbackTimer = undefined;
      state.navigate(timeline => timeline.advance());
    }, 1000 / timeline.speed);
  }
}
state.subscribe(render);
render();

window.addEventListener('keydown', event => {
  const target = event.target;
  if (event.ctrlKey || event.metaKey || event.altKey || (target instanceof HTMLElement && target.closest('input, textarea, select, button, [contenteditable="true"]'))) return;
  const key = event.key.toLowerCase();
  if (key === ' ') state.navigate(timeline => timeline.togglePlay());
  else if (key === 'arrowleft') state.navigate(timeline => event.shiftKey ? timeline.jumpKeyframe(-1) : timeline.step(-1));
  else if (key === 'arrowright') state.navigate(timeline => event.shiftKey ? timeline.jumpKeyframe(1) : timeline.step(1));
  else if (key === 'home') state.navigate(timeline => timeline.edge('first'));
  else if (key === 'end') state.navigate(timeline => timeline.edge('last'));
  else if (key === '[') state.navigate(timeline => timeline.changeSpeed(-1));
  else if (key === ']') state.navigate(timeline => timeline.changeSpeed(1));
  else if (key === 'l') state.navigate(timeline => timeline.follow());
  else if (key === 'p' && !event.repeat && !pauseButton.disabled) pauseButton.click();
  else if (key === 'k' && !event.repeat && !keyframeButton.disabled) keyframeButton.click();
  else if (key === 'd' && !event.repeat) {
    themePicker.value = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    themePicker.dispatchEvent(new Event('change'));
  } else if (key === '?') {
    const help = find<HTMLDetailsElement>('.help');
    help.open = !help.open;
  } else return;
  event.preventDefault();
});

client.connect();
