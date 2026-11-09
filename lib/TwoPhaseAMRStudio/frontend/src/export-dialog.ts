import type { StudioClient } from './client';
import { groupDiagnostics, type DiagnosticMetadata } from './diagnostic-data';
import type { StudioState } from './state';

/** The visible settings of a field panel, sampled when export starts. */
export interface ExportView {
  id: string;
  title: string;
  field: string;
  mode: 'cell' | 'plic';
  camera: [number, number, number, number] | null;
  colorrange: [number, number] | null;
  colormap?: string;
  ready: boolean;
  overlays: { interface: boolean; solid: boolean; blocks: boolean; cells: boolean; levels: boolean };
}

export function installExportDialog(button: HTMLButtonElement, state: StudioState, client: StudioClient, views: () => ExportView[]): void {
  const dialog = document.createElement('dialog');
  dialog.className = 'export-dialog';
  dialog.dataset.testid = 'export-dialog';
  dialog.innerHTML = `
    <form>
      <h2>Export animation</h2>
      <label>Panel <select name="panel" data-testid="export-panel" required></select></label>
      <label>Source <select name="source" data-testid="export-source"><option value="stored">Stored frames</option><option value="live">Live stream</option></select></label>
      <fieldset><legend>Diagnostics by dimension</legend><div class="export-diagnostics"></div></fieldset>
      <label>Diagnostics layout <select name="arrangement" data-testid="export-arrangement"><option value="right">Stacked to the right</option><option value="below">Below the field</option></select></label>
      <label>Camera <select name="camera" data-testid="export-camera"><option value="current">Current view</option><option value="fit">Fit domain</option></select></label>
      <label>Frames <select name="frames" data-testid="export-frames"><option value="keyframes">Keyframes</option><option value="all">All stored frames</option></select></label>
      <div class="export-window"><label>From time <input name="from" type="number" step="any" placeholder="start"></label><label>To time <input name="to" type="number" step="any" placeholder="end"></label></div>
      <label>Colour range <select name="range"><option value="global">Global over selected frames</option><option value="current">Current panel range</option></select></label>
      <div class="export-dimensions"><label>Width <input name="width" type="number" min="2" max="8192" step="2" value="1280" required></label><label>Height <input name="height" type="number" min="2" max="8192" step="2" value="720" required></label><label>FPS <input name="fps" type="number" min="1" max="240" step="1" value="12" required></label></div>
      <label>Theme <select name="theme"><option value="current">Current</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
      <label>Format <select name="format" data-testid="export-format"><option value="mp4">MP4</option><option value="gif">GIF</option><option value="webm">WebM</option></select></label>
      <label>Output file on server <input name="path" data-testid="export-path" type="text" value="studio.mp4" required></label>
      <p class="export-note">Stored export renders a field and the selected diagnostic groups. Live export records new frames until stopped.</p>
      <div class="export-actions"><button type="submit" data-testid="export-submit">Export</button><button type="button" data-testid="export-stop">Stop live recording</button><button type="button" data-testid="export-cancel">Cancel export</button><button type="button" class="export-close">Close</button></div>
      <progress data-testid="export-progress" aria-label="Export progress" max="1" hidden></progress>
      <div class="export-status" data-testid="export-status" role="status" aria-live="polite"></div>
      <a class="export-download" data-testid="export-download" hidden download>Download animation</a>
    </form>`;
  document.body.append(dialog);
  const find = <T extends HTMLElement>(selector: string): T => {
    const node = dialog.querySelector<T>(selector);
    if (!node) throw new Error(`Missing export dialog element: ${selector}`);
    return node;
  };
  const form = find<HTMLFormElement>('form');
  const panel = find<HTMLSelectElement>('[name="panel"]');
  const source = find<HTMLSelectElement>('[name="source"]');
  const camera = find<HTMLSelectElement>('[name="camera"]');
  const arrangement = find<HTMLSelectElement>('[name="arrangement"]');
  const frames = find<HTMLSelectElement>('[name="frames"]');
  const range = find<HTMLSelectElement>('[name="range"]');
  const from = find<HTMLInputElement>('[name="from"]');
  const to = find<HTMLInputElement>('[name="to"]');
  const width = find<HTMLInputElement>('[name="width"]');
  const height = find<HTMLInputElement>('[name="height"]');
  const fps = find<HTMLInputElement>('[name="fps"]');
  const theme = find<HTMLSelectElement>('[name="theme"]');
  const format = find<HTMLSelectElement>('[name="format"]');
  const path = find<HTMLInputElement>('[name="path"]');
  const submit = find<HTMLButtonElement>('[data-testid="export-submit"]');
  const stop = find<HTMLButtonElement>('[data-testid="export-stop"]');
  const cancel = find<HTMLButtonElement>('[data-testid="export-cancel"]');
  const progress = find<HTMLProgressElement>('[data-testid="export-progress"]');
  const status = find<HTMLElement>('[data-testid="export-status"]');
  const download = find<HTMLAnchorElement>('[data-testid="export-download"]');
  const diagnostics = find<HTMLElement>('.export-diagnostics');
  const note = find<HTMLElement>('.export-note');

  function update(): void {
    const recording = state.recording;
    const live = source.value === 'live';
    for (const item of [frames, from, to, theme]) item.disabled = live || recording.active;
    for (const item of [camera, arrangement, range, width, height, fps, format]) item.disabled = recording.active;
    range.options[0].textContent = live ? 'Automatic as frames arrive' : 'Global over selected frames';
    panel.disabled = recording.active;
    source.disabled = recording.active;
    path.disabled = recording.active;
    for (const input of diagnostics.querySelectorAll<HTMLInputElement>('input')) input.disabled = recording.active;
    const selectedView = views().find(view => view.id === panel.value);
    note.textContent = 'Stored export renders the selected field and diagnostic groups; live export records new frames until stopped.' +
      (selectedView?.overlays.cells || selectedView?.overlays.levels ? ' Cell edges and level colours are not included in server-side output.' : '');
    submit.disabled = recording.active || !selectedView?.ready || state.connection !== 'connected';
    submit.textContent = live ? 'Start live recording' : 'Export';
    stop.hidden = !(recording.active && recording.source === 'live');
    cancel.hidden = !(recording.active && recording.source === 'stored');
    cancel.disabled = recording.status === 'cancelling';
    progress.hidden = recording.source !== 'stored' || (!recording.active && recording.status !== 'completed');
    if (typeof recording.progress === 'number' && Number.isFinite(recording.progress)) progress.value = Math.min(1, Math.max(0, recording.progress));
    else progress.removeAttribute('value');
    status.textContent = `${recording.status} · ${recording.encoded} encoded · ${recording.dropped} dropped${recording.error || state.error ? ` · ${recording.error || state.error}` : ''}`;
    download.hidden = true;
    if (recording.status === 'completed' && recording.url) {
      const url = new URL(recording.url, window.location.href);
      if (url.origin === window.location.origin) {
        const token = new URL(window.location.href).searchParams.get('token');
        if (token) url.searchParams.set('token', token);
        download.href = url.toString();
        download.hidden = false;
      }
    }
  }
  panel.addEventListener('change', update);
  window.addEventListener('studio-view-change', update);
  source.addEventListener('change', update);
  format.addEventListener('change', () => {
    const filename = path.value.trim();
    path.value = `${filename.replace(/\.[^./\\]+$/, '')}.${format.value}`;
  });
  path.addEventListener('input', () => {
    const extension = path.value.match(/\.([^.\\/]+)$/)?.[1]?.toLowerCase();
    if (extension && ['mp4', 'gif', 'webm'].includes(extension)) format.value = extension;
  });
  button.addEventListener('click', () => {
    const availableViews = views();
    panel.replaceChildren(...availableViews.map(view => {
      const option = document.createElement('option');
      option.value = view.id;
      option.textContent = view.title;
      return option;
    }));
    diagnostics.replaceChildren(...groupDiagnostics((state.runs[0]?.diagnostics ?? []) as DiagnosticMetadata[]).map(group => {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = JSON.stringify(group.names);
      input.dataset.testid = 'export-diagnostic';
      label.append(input, document.createTextNode(group.title));
      return label;
    }));
    if (!diagnostics.childNodes.length) diagnostics.textContent = 'None available';
    update();
    dialog.showModal();
  });
  find<HTMLButtonElement>('.export-close').addEventListener('click', () => dialog.close());
  stop.addEventListener('click', () => client.recording({ type: 'recording', action: 'stop' }));
  cancel.addEventListener('click', () => client.recording({ type: 'recording', action: 'cancel' }));
  form.addEventListener('submit', event => {
    event.preventDefault();
    const view = views().find(item => item.id === panel.value);
    if (!view?.ready) { state.setError('Wait for the selected field panel to finish loading.'); return; }
    const selected = [...diagnostics.querySelectorAll<HTMLInputElement>('input:checked')].flatMap(input => JSON.parse(input.value) as string[]);
    if (camera.value === 'current' && !view.camera) { state.setError('The selected panel has no current camera yet.'); return; }
    if (range.value === 'current' && !view.colorrange) { state.setError('The selected panel has no current colour range yet.'); return; }
    if (source.value === 'live') {
      client.recording({
        type: 'recording', action: 'start', source: 'live', path: path.value.trim(), field: view.field, diagnostics: selected,
        camera: camera.value === 'fit' ? 'fit' : view.camera!,
        ...(range.value === 'current' ? { colorrange: view.colorrange! } : {}),
        ...(view.colormap ? { colormap: view.colormap } : {}),
        phase_rendering: view.mode, size: [Number(width.value), Number(height.value)], framerate: Number(fps.value),
        arrangement: arrangement.value === 'below' ? 'below' : 'right',
        interface: view.overlays.interface, geometry: view.overlays.solid, blocks: view.overlays.blocks || view.overlays.levels,
      });
      return;
    }
    let window: [number, number] | undefined;
    if (from.value || to.value) {
      const start = from.value ? Number(from.value) : -Number.MAX_VALUE;
      const end = to.value ? Number(to.value) : Number.MAX_VALUE;
      if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) { state.setError('Choose a valid export time window.'); return; }
      window = [start, end];
    }
    client.recording({
      type: 'recording', action: 'export', source: 'stored', path: path.value.trim(), field: view.field, diagnostics: selected,
      frames: frames.value === 'all' ? 'all' : 'keyframes', ...(window ? { window } : {}),
      camera: camera.value === 'fit' ? 'fit' : view.camera!,
      colorrange: range.value === 'global' ? 'global' : view.colorrange!, ...(view.colormap ? { colormap: view.colormap } : {}),
      phase_rendering: view.mode, size: [Number(width.value), Number(height.value)], framerate: Number(fps.value),
      arrangement: arrangement.value === 'below' ? 'below' : 'right',
      interface: view.overlays.interface, geometry: view.overlays.solid, blocks: view.overlays.blocks || view.overlays.levels,
      theme: theme.value === 'current' ? (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light') : theme.value === 'dark' ? 'dark' : 'light',
    });
  });
  state.subscribe(update);
}
