import type { StudioClient } from './client';
import { groupDiagnostics, type DiagnosticMetadata } from './diagnostic-data';
import type { StudioState } from './state';
import type { DiagnosticExportOptions, FrameId, PreviewCommand } from './types';

/** The visible settings of a field panel, sampled when export starts. */
export interface ExportView {
  id: string;
  title: string;
  field: string;
  frame: FrameId | null;
  mode: 'cell' | 'plic';
  camera: [number, number, number, number] | null;
  colorrange: [number, number] | null;
  colormap?: string;
  ready: boolean;
  overlays: { interface: boolean; solid: boolean; blocks: boolean; cells: boolean; levels: boolean };
}

export interface ExportDiagnosticView extends DiagnosticExportOptions { panelId: string }

export function installExportDialog(button: HTMLButtonElement, state: StudioState, client: StudioClient, views: () => ExportView[], diagnosticViews: () => ExportDiagnosticView[]): void {
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
      <label>Every <input name="every" type="number" min="1" step="1" value="1" required> frame(s)</label>
      <label>Colour range <select name="range"><option value="global">Global over selected frames</option><option value="current">Current panel range</option></select></label>
      <div class="export-dimensions"><label>Width <input name="width" type="number" min="2" max="8192" step="2" value="1280" required></label><label>Height <input name="height" type="number" min="2" max="8192" step="2" value="720" required></label><label>FPS <input name="fps" type="number" min="1" max="240" step="1" value="12" required></label></div>
      <label>Theme <select name="theme"><option value="current">Current</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
      <label>Format <select name="format" data-testid="export-format"><option value="mp4">MP4</option><option value="gif">GIF</option><option value="webm">WebM</option></select></label>
      <label>Output file on server <input name="path" data-testid="export-path" type="text" value="studio.mp4" required></label>
      <p class="export-note">Stored export renders a field and the selected diagnostic groups. Live export records new frames until stopped.</p>
      <div class="export-actions"><button type="button" data-testid="export-preview">Preview frame</button><button type="submit" data-testid="export-submit">Export</button><button type="button" data-testid="export-stop">Stop live recording</button><button type="button" data-testid="export-cancel">Cancel export</button><button type="button" class="export-close">Close</button></div>
      <progress data-testid="export-progress" aria-label="Export progress" max="1" hidden></progress>
      <div class="export-status" data-testid="export-status" role="status" aria-live="polite"></div>
      <img class="export-preview" data-testid="export-preview-image" alt="Preview of the selected field and diagnostics" hidden>
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
  const every = find<HTMLInputElement>('[name="every"]');
  const width = find<HTMLInputElement>('[name="width"]');
  const height = find<HTMLInputElement>('[name="height"]');
  const fps = find<HTMLInputElement>('[name="fps"]');
  const theme = find<HTMLSelectElement>('[name="theme"]');
  const format = find<HTMLSelectElement>('[name="format"]');
  const path = find<HTMLInputElement>('[name="path"]');
  const submit = find<HTMLButtonElement>('[data-testid="export-submit"]');
  const preview = find<HTMLButtonElement>('[data-testid="export-preview"]');
  const previewImage = find<HTMLImageElement>('[data-testid="export-preview-image"]');
  const stop = find<HTMLButtonElement>('[data-testid="export-stop"]');
  const cancel = find<HTMLButtonElement>('[data-testid="export-cancel"]');
  const progress = find<HTMLProgressElement>('[data-testid="export-progress"]');
  const status = find<HTMLElement>('[data-testid="export-status"]');
  let previewPending = false;
  let pathWasEdited = false;
  const download = find<HTMLAnchorElement>('[data-testid="export-download"]');
  const diagnostics = find<HTMLElement>('.export-diagnostics');
  const note = find<HTMLElement>('.export-note');

  function update(): void {
    const recording = state.recording;
    const live = source.value === 'live';
    for (const item of [frames, from, to, theme]) item.disabled = live || recording.active;
    for (const item of [camera, arrangement, range, every, width, height, fps, format]) item.disabled = recording.active;
    range.options[0].textContent = live ? 'Automatic as frames arrive' : 'Global over selected frames';
    panel.disabled = recording.active;
    source.disabled = recording.active;
    path.disabled = recording.active;
    for (const input of diagnostics.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')) input.disabled = recording.active;
    const selectedView = views().find(view => view.id === panel.value);
    note.textContent = 'Diagnostic exports use the selected panel’s visible series and y-axis scale. Stored export renders selected groups; live export records new frames until stopped.' +
      (selectedView?.overlays.cells || selectedView?.overlays.levels ? ' Cell edges and level colours are not included in server-side output.' : '');
    submit.disabled = recording.active || previewPending || !selectedView?.ready || state.connection !== 'connected';
    preview.disabled = recording.active || previewPending || !selectedView?.ready || state.connection !== 'connected' || selectedView.frame === null;
    preview.textContent = previewPending ? 'Rendering preview…' : 'Preview frame';
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
  function setDefaultPath(): void {
    const view = views().find(item => item.id === panel.value);
    const name = `${state.runs[0]?.name || 'run'}-${view?.field || 'field'}`.replace(/[^a-zA-Z0-9_.-]+/g, '-').replace(/^-+|-+$/g, '');
    path.value = `${name || 'studio'}.${format.value}`;
  }
  panel.addEventListener('change', () => { if (!pathWasEdited) setDefaultPath(); update(); });
  window.addEventListener('studio-view-change', update);
  source.addEventListener('change', update);
  format.addEventListener('change', () => {
    const filename = path.value.trim();
    path.value = `${filename.replace(/\.[^./\\]+$/, '')}.${format.value}`;
  });
  path.addEventListener('input', () => {
    pathWasEdited = true;
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
    pathWasEdited = false;
    setDefaultPath();
    previewImage.hidden = true;
    const availableDiagnostics = diagnosticViews();
    diagnostics.replaceChildren(...groupDiagnostics((state.runs[0]?.diagnostics ?? []) as DiagnosticMetadata[]).map(group => {
      const row = document.createElement('div');
      row.className = 'export-diagnostic-row';
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = JSON.stringify(group.names);
      input.dataset.testid = 'export-diagnostic';
      label.append(input, document.createTextNode(group.title));
      row.append(label);
      const matching = availableDiagnostics.filter(view => view.names.length === group.names.length && view.names.every((name, i) => name === group.names[i]));
      if (matching.length === 1) row.dataset.panelId = matching[0].panelId;
      else if (matching.length > 1) {
        const select = document.createElement('select');
        select.setAttribute('aria-label', `${group.title} panel for export`);
        for (const view of matching) {
          const option = document.createElement('option');
          option.value = view.panelId;
          option.textContent = view.panelId;
          select.append(option);
        }
        row.append(select);
      }
      return row;
    }));
    if (!diagnostics.childNodes.length) diagnostics.textContent = 'None available';
    update();
    dialog.showModal();
  });
  find<HTMLButtonElement>('.export-close').addEventListener('click', () => dialog.close());
  stop.addEventListener('click', () => client.recording({ type: 'recording', action: 'stop' }));
  cancel.addEventListener('click', () => client.recording({ type: 'recording', action: 'cancel' }));
  function selectedOptions(): Omit<PreviewCommand, 'action' | 'frame'> | null {
    const view = views().find(item => item.id === panel.value);
    if (!view?.ready) { state.setError('Wait for the selected field panel to finish loading.'); return null; }
    const checkedDiagnostics = [...diagnostics.querySelectorAll<HTMLInputElement>('input:checked')];
    const selected = checkedDiagnostics.flatMap(input => JSON.parse(input.value) as string[]);
    const currentDiagnostics = diagnosticViews();
    const diagnosticOptions = checkedDiagnostics.flatMap(input => {
      const row = input.closest<HTMLElement>('.export-diagnostic-row')!;
      const panelId = row.querySelector<HTMLSelectElement>('select')?.value ?? row.dataset.panelId;
      const config = currentDiagnostics.find(item => item.panelId === panelId);
      return config ? [{ names: config.names, scale: config.scale, hidden: config.hidden }] : [];
    });
    if (camera.value === 'current' && !view.camera) { state.setError('The selected panel has no current camera yet.'); return null; }
    if (range.value === 'current' && !view.colorrange) { state.setError('The selected panel has no current colour range yet.'); return null; }
    const selectedRange = view.colorrange && view.colorrange[0] === view.colorrange[1] ? (() => {
      const value = view.colorrange![0];
      const padding = value === 0 ? 0.5 : Math.max(Math.abs(value) * 0.01, 2 ** -149);
      const limit = 3.4028234663852886e38;
      return [Math.max(-limit, value - padding), Math.min(limit, value + padding)] as [number, number];
    })() : view.colorrange;
    let window: [number, number] | undefined;
    if (from.value || to.value) {
      const start = from.value ? Number(from.value) : -Number.MAX_VALUE;
      const end = to.value ? Number(to.value) : Number.MAX_VALUE;
      if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) { state.setError('Choose a valid export time window.'); return null; }
      window = [start, end];
    }
    return {
      type: 'recording', source: 'stored', field: view.field, diagnostics: selected, diagnostic_options: diagnosticOptions,
      frames: frames.value === 'all' ? 'all' : 'keyframes', ...(window ? { window } : {}),
      camera: camera.value === 'fit' ? 'fit' : view.camera!,
      colorrange: range.value === 'global' ? 'global' : selectedRange!, ...(view.colormap ? { colormap: view.colormap } : {}),
      phase_rendering: view.mode, size: [Number(width.value), Number(height.value)], framerate: Number(fps.value),
      every: Number(every.value), arrangement: arrangement.value === 'below' ? 'below' : 'right',
      interface: view.overlays.interface, geometry: view.overlays.solid, blocks: view.overlays.blocks || view.overlays.levels,
      theme: theme.value === 'current' ? (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light') : theme.value === 'dark' ? 'dark' : 'light',
    };
  }
  preview.addEventListener('click', () => {
    for (const input of [width, height, fps, every]) if (!input.reportValidity()) return;
    const frame = views().find(item => item.id === panel.value)?.frame;
    const options = selectedOptions();
    if (frame == null || !options) return;
    previewPending = true;
    previewImage.hidden = true;
    update();
    client.preview({ ...options, action: 'preview', frame }).then(url => {
      previewImage.src = url;
      previewImage.hidden = false;
    }).catch(error => {
      state.setError(error instanceof Error ? error.message : 'Could not render preview.');
    }).finally(() => { previewPending = false; update(); });
  });
  form.addEventListener('submit', event => {
    event.preventDefault();
    const options = selectedOptions();
    if (!options) return;
    if (source.value === 'live') {
      client.recording({
        type: 'recording', action: 'start', source: 'live', path: path.value.trim(),
        field: options.field, diagnostics: options.diagnostics, diagnostic_options: options.diagnostic_options,
        camera: options.camera, ...(options.colorrange === 'global' ? {} : { colorrange: options.colorrange }),
        ...(options.colormap ? { colormap: options.colormap } : {}),
        phase_rendering: options.phase_rendering, size: options.size, framerate: options.framerate,
        every: options.every, arrangement: options.arrangement,
        interface: options.interface, geometry: options.geometry, blocks: options.blocks,
      });
    } else {
      client.recording({ ...options, action: 'export', path: path.value.trim() });
    }
  });
  state.subscribe(update);
}
