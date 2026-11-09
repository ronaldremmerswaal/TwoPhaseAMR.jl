import 'dockview/dist/styles/dockview.css';
import './style.css';
import { createDockview, themeDark, themeLight, type DockviewApi } from 'dockview';
import { StudioClient } from './client';
import { framesPanel, formatBytes, formatNumber, PanelRegistry, runPanel, type PanelDefinition } from './panels';
import { StudioState } from './state';
import { PLAYBACK_SPEEDS } from './timeline';

const root = document.querySelector<HTMLElement>('#app');
if (!root) throw new Error('Missing Studio root element');
root.innerHTML = `
<div class="studio-shell">
  <header class="studio-header">
    <div class="header-row">
      <h1>TwoPhaseAMR Studio</h1>
      <span class="status-pill" data-testid="connection"><span data-testid="connection-status" role="status">Connecting</span></span>
      <strong data-testid="run-state">unknown</strong>
      <span id="run-clock">t = — · step — · Δt —</span>
      <span id="store-size">Store —</span>
      <span class="recording-indicator" data-testid="recording-state"><span data-testid="recording-status" role="status">Recording: idle</span></span>
    </div>
    <div class="header-row controls" aria-label="Solver controls">
      <button type="button" data-testid="control-pause" title="Pause or resume solver (P)">Pause</button>
      <button type="button" data-testid="control-stop" title="Stop after the current step">Stop</button>
      <button type="button" data-testid="control-keyframe" title="Capture keyframe (K)">Keyframe</button>
      <button type="button" id="reconnect" hidden>Reconnect</button>
      <label>Panel <select id="panel-picker" aria-label="Panel to open"></select></label>
      <button type="button" data-testid="add-panel">Open panel</button>
      <label>Theme <select data-testid="theme-select" aria-label="Theme"><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
      <details class="help"><summary title="Keyboard shortcuts (?)">Shortcuts</summary>
        <div>Space: playback · P: pause/resume solver · L: follow live · K: keyframe · ←/→: frame · Shift+←/→: keyframe · Home/End: first/last · [ / ]: speed · D: theme</div>
      </details>
    </div>
    <details class="recording-controls" open>
      <summary>Recording controls</summary>
      <form id="recording-form">
        <label>Source <select data-testid="recording-source" aria-label="Recording source"><option value="live">Live</option><option value="stored">Stored</option></select></label>
        <label>Output path <input data-testid="recording-path" aria-label="Recording output path" type="text" required autocomplete="off" placeholder="output.mp4"></label>
        <label>Field <select data-testid="recording-field" aria-label="Recording field" required></select></label>
        <fieldset><legend>Diagnostics</legend><div id="diagnostic-options"></div></fieldset>
        <button type="submit" data-testid="recording-start">Start recording</button>
        <button type="button" data-testid="recording-stop">Stop recording</button>
        <span id="recording-detail"></span>
      </form>
    </details>
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
      <strong data-testid="timeline-mode">LIVE</strong>
      <span data-testid="timeline-new-count" aria-live="polite" hidden></span>
      <button type="button" data-testid="timeline-follow" title="Follow live (L)">Follow</button>
      <button type="button" id="refresh-frames" title="Refresh the stored frame index">Refresh frames</button>
    </div>
    <input data-testid="timeline-scrub" aria-label="Timeline" type="range" min="0" max="0" value="0" step="1" disabled>
    <div class="timeline-ends"><span id="timeline-start">—</span><span id="timeline-end">—</span></div>
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
const panelPicker = find<HTMLSelectElement>('#panel-picker');
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
    element.textContent = 'No panels open. Choose a panel above and select Open panel.';
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
};
declare global {
  interface Window {
    twoPhaseAMRStudio: StudioExtensionAPI;
  }
}
window.twoPhaseAMRStudio = extensionAPI;

function updatePanelPicker(): void {
  const selected = panelPicker.value;
  panelPicker.replaceChildren(...registry.all().map(definition => {
    const option = document.createElement('option');
    option.value = definition.id;
    option.textContent = definition.title;
    return option;
  }));
  if ([...panelPicker.options].some(option => option.value === selected)) panelPicker.value = selected;
}
registry.subscribe(updatePanelPicker);
updatePanelPicker();
let orientation: 'right' | 'bottom' = window.innerWidth / window.innerHeight > 1.6 ? 'right' : 'bottom';
openPanel('run');
openPanel('frames', { referencePanel: 'run', direction: orientation === 'right' ? 'right' : 'below' });
let automaticLayout = true;
let reflowing = false;
dock.onDidDrop(() => { automaticLayout = false; });
dock.onDidAddPanel(() => { if (!reflowing) automaticLayout = false; });
dock.onDidRemovePanel(() => { if (!reflowing) automaticLayout = false; });
dockElement.addEventListener('pointerdown', event => {
  if (event.target instanceof Element && event.target.closest('.dv-sash')) automaticLayout = false;
});
window.addEventListener('resize', () => {
  const next = window.innerWidth / window.innerHeight > 1.6 ? 'right' : 'bottom';
  if (!automaticLayout || next === orientation) return;
  const run = dock.getPanel('run');
  const frames = dock.getPanel('frames');
  if (!run || !frames) { automaticLayout = false; return; }
  reflowing = true;
  try {
    frames.api.moveTo({ group: run.group, position: next, skipSetActive: true });
    orientation = next;
  } finally {
    reflowing = false;
  }
});
find<HTMLButtonElement>('[data-testid="add-panel"]').addEventListener('click', () => openPanel(panelPicker.value));

function applyTheme(): void {
  const dark = themePicker.value === 'dark' || (themePicker.value === 'system' && darkPreference.matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  dock.updateOptions({ theme: dark ? themeDark : themeLight });
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
const reconnectButton = find<HTMLButtonElement>('#reconnect');
const recordingForm = find<HTMLFormElement>('#recording-form');
const recordingSource = find<HTMLSelectElement>('[data-testid="recording-source"]');
const recordingPath = find<HTMLInputElement>('[data-testid="recording-path"]');
const recordingField = find<HTMLSelectElement>('[data-testid="recording-field"]');
const diagnosticOptions = find<HTMLElement>('#diagnostic-options');
const recordingStart = find<HTMLButtonElement>('[data-testid="recording-start"]');
const recordingStop = find<HTMLButtonElement>('[data-testid="recording-stop"]');
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

pauseButton.addEventListener('click', () => client.control(state.runState === 'paused' ? 'resume' : 'pause'));
stopButton.addEventListener('click', () => client.control('stop'));
keyframeButton.addEventListener('click', () => client.control('keyframe'));
reconnectButton.addEventListener('click', () => client.connect());
find<HTMLButtonElement>('#dismiss-error').addEventListener('click', () => state.clearError());
find<HTMLButtonElement>('#refresh-frames').addEventListener('click', () => client.getFrames());
recordingForm.addEventListener('submit', event => {
  event.preventDefault();
  const path = recordingPath.value.trim();
  if (!path || !recordingField.value) { state.setError('Choose a recording field and output path.'); return; }
  const diagnostics = [...diagnosticOptions.querySelectorAll<HTMLInputElement>('input:checked')].map(input => input.value);
  client.recording({ type: 'recording', action: 'start', source: recordingSource.value === 'stored' ? 'stored' : 'live', path, field: recordingField.value, diagnostics });
});
recordingStop.addEventListener('click', () => client.recording({ type: 'recording', action: 'stop' }));
find<HTMLButtonElement>('[data-testid="timeline-follow"]').addEventListener('click', () => state.navigate(timeline => timeline.follow()));
find<HTMLButtonElement>('[data-testid="timeline-prev"]').addEventListener('click', () => state.navigate(timeline => timeline.step(-1)));
find<HTMLButtonElement>('[data-testid="timeline-next"]').addEventListener('click', () => state.navigate(timeline => timeline.step(1)));
find<HTMLButtonElement>('[data-testid="timeline-prev-keyframe"]').addEventListener('click', () => state.navigate(timeline => timeline.jumpKeyframe(-1)));
find<HTMLButtonElement>('[data-testid="timeline-next-keyframe"]').addEventListener('click', () => state.navigate(timeline => timeline.jumpKeyframe(1)));
playButton.addEventListener('click', () => state.navigate(timeline => timeline.togglePlay()));
slider.addEventListener('input', () => state.navigate(timeline => { timeline.playing = false; timeline.seek(Number(slider.value)); }));
speedPicker.addEventListener('change', () => state.navigate(timeline => { timeline.speed = Number(speedPicker.value); }));
loopCheckbox.addEventListener('change', () => state.navigate(timeline => { timeline.loop = loopCheckbox.checked; }));

let metadataSignature = '';
let playbackTimer: number | undefined;
let timerSpeed = 0;
function render(): void {
  if (!themeChosen && state.preferredTheme) {
    themePicker.value = state.preferredTheme;
    themeChosen = true;
    applyTheme();
  }
  const connected = state.connection === 'connected';
  find<HTMLElement>('[data-testid="connection-status"]').textContent = state.connection;
  find<HTMLElement>('[data-testid="run-state"]').textContent = state.runState.toUpperCase();
  find<HTMLElement>('#run-clock').textContent = `t = ${formatNumber(state.time)} · step ${state.step ?? '—'} · Δt ${formatNumber(state.dt)}`;
  find<HTMLElement>('#store-size').textContent = `Store ${formatBytes(state.storeBytes)}`;
  const recording = state.recording;
  find<HTMLElement>('[data-testid="recording-status"]').textContent = recording.active ? `● REC · ${recording.status}` : `Recording: ${recording.status}`;
  find<HTMLElement>('[data-testid="recording-state"]').classList.toggle('active', recording.active);
  find<HTMLElement>('#recording-detail').textContent = `${recording.encoded} encoded · ${recording.dropped} dropped${recording.active && recording.source ? ` · ${recording.source}` : ''}${recording.field ? ` · ${recording.field}` : ''}${recording.path ? ` · ${recording.path}` : ''}${recording.error ? ` · ${recording.error}` : ''}`;
  pauseButton.textContent = state.runState === 'paused' ? 'Resume' : 'Pause';
  pauseButton.disabled = !connected;
  stopButton.disabled = !connected;
  keyframeButton.disabled = !connected;
  reconnectButton.hidden = connected || state.connection === 'connecting';
  find<HTMLButtonElement>('#refresh-frames').disabled = !connected;
  const errorBox = find<HTMLElement>('#error');
  errorBox.hidden = !state.error;
  find<HTMLElement>('#error-text').textContent = state.error ?? '';

  const run = state.runs[0];
  const fields = run?.fields ?? [];
  const diagnostics = run?.diagnostics ?? [];
  const signature = JSON.stringify([fields, diagnostics]);
  if (signature !== metadataSignature) {
    metadataSignature = signature;
    const selectedField = recordingField.value;
    recordingField.replaceChildren(...fields.map(field => {
      const option = document.createElement('option');
      option.value = field.name;
      option.textContent = field.label || field.name;
      return option;
    }));
    if (fields.some(field => field.name === selectedField)) recordingField.value = selectedField;
    const checked = new Set([...diagnosticOptions.querySelectorAll<HTMLInputElement>('input:checked')].map(input => input.value));
    diagnosticOptions.replaceChildren(...diagnostics.map(diagnostic => {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = diagnostic.name;
      input.checked = checked.has(diagnostic.name);
      input.dataset.testid = 'recording-diagnostic';
      label.append(input, document.createTextNode(diagnostic.label || diagnostic.name));
      return label;
    }));
    if (!diagnostics.length) diagnosticOptions.textContent = 'None available';
  }
  recordingSource.disabled = !connected || recording.active;
  recordingPath.disabled = !connected || recording.active;
  recordingField.disabled = !connected || recording.active || !fields.length;
  for (const input of diagnosticOptions.querySelectorAll<HTMLInputElement>('input')) input.disabled = !connected || recording.active;
  recordingStart.disabled = !connected || recording.active || !fields.length;
  recordingStop.disabled = !connected || !recording.active;

  const timeline = state.timeline;
  const current = timeline.current;
  find<HTMLElement>('[data-testid="timeline-mode"]').textContent = timeline.mode.toUpperCase();
  const newCount = find<HTMLElement>('[data-testid="timeline-new-count"]');
  newCount.hidden = timeline.mode === 'live';
  newCount.textContent = `+${timeline.newCount} new`;
  find<HTMLElement>('[data-testid="timeline-current"]').textContent = current ? `#${current.id} · t = ${formatNumber(current.time)} · step ${current.step} · ${current.kind}` : 'No frame';
  slider.max = String(Math.max(0, timeline.frames.length - 1));
  slider.value = String(Math.max(0, timeline.index));
  slider.disabled = !timeline.frames.length;
  slider.setAttribute('aria-valuetext', current ? `Frame ${current.id}, time ${formatNumber(current.time)}, ${current.kind}` : 'No frame');
  find<HTMLElement>('#timeline-start').textContent = timeline.frames.length ? formatNumber(timeline.frames[0].time) : '—';
  find<HTMLElement>('#timeline-end').textContent = timeline.frames.length ? formatNumber(timeline.frames[timeline.frames.length - 1].time) : '—';
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
