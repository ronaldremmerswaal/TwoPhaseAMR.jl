import type { StudioClient } from './client';
import { groupDiagnostics, type DiagnosticMetadata } from './diagnostic-data';
import {
  frameSelection, liveSummary, parseTimeWindow, scheduleAttributes, scheduleSummary,
} from './export-plan';
import type { StudioState } from './state';
import type {
  DiagnosticExportOptions, FrameId, LiveExportOptions, PreviewCommand, RecordingSchedule,
  ScheduleCommand, StoredExportOptions,
} from './types';

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
      <label class="export-stored">Frames <select name="frames" data-testid="export-frames"><option value="keyframes">Keyframes</option><option value="all">All stored frames</option></select></label>
      <div class="export-window export-stored"><label>From time <input name="from" type="number" step="any" placeholder="start"></label><label>To time <input name="to" type="number" step="any" placeholder="end"></label></div>
      <label class="export-stored">Duration <input name="duration" type="number" min="0.001" step="any" value="2" required> s</label>
      <label class="export-live">Simulation time per video second <input name="sim_per_second" type="number" min="0.000001" step="any" value="1" required></label>
      <label>FPS <input name="fps" type="number" min="1" max="240" step="1" value="20" required></label>
      <div class="export-schedule" data-testid="export-schedule" role="status" aria-live="polite"></div>
      <label>Colour range <select name="range" data-testid="export-range"><option value="global">Global over stored frames</option><option value="auto">Automatic as frames arrive</option><option value="current">Current panel range</option></select></label>
      <div class="export-dimensions"><label>Width <input name="width" type="number" min="2" max="8192" step="2" value="1280" required></label><label>Height <input name="height" type="number" min="2" max="8192" step="2" value="720" required></label></div>
      <label>Theme <select name="theme"><option value="current">Current</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
      <label>Format <select name="format" data-testid="export-format"><option value="mp4">MP4</option><option value="gif">GIF</option><option value="webm">WebM</option></select></label>
      <label>Output file on server <input name="path" data-testid="export-path" type="text" value="studio.mp4" required></label>
      <p class="export-note">Stored export samples the chosen frames by simulation time. Live export records produced frames from now until stopped. Both play time linearly in simulation time.</p>
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
  const duration = find<HTMLInputElement>('[name="duration"]');
  const simPerSecond = find<HTMLInputElement>('[name="sim_per_second"]');
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
  const scheduleElement = find<HTMLElement>('[data-testid="export-schedule"]');
  let previewPending = false;
  let pathWasEdited = false;
  let durationEdited = false;
  let simRateEdited = false;
  let lastSchedule: RecordingSchedule | null = null;
  let scheduleKey = '';
  let pendingKey = '';
  let scheduledKey = '';
  let scheduleTimer: number | undefined;
  let scheduleGeneration = 0;
  const download = find<HTMLAnchorElement>('[data-testid="export-download"]');
  const diagnostics = find<HTMLElement>('.export-diagnostics');
  const note = find<HTMLElement>('.export-note');
  const storedFields = [...dialog.querySelectorAll<HTMLElement>('.export-stored')];

  const isLive = (): boolean => source.value === 'live';

  function checkedDiagnostics(): HTMLInputElement[] {
    return [...diagnostics.querySelectorAll<HTMLInputElement>('input:checked')];
  }

  function diagnosticNames(): string[] {
    return checkedDiagnostics().flatMap(input => JSON.parse(input.value) as string[]);
  }

  function diagnosticOptions(): DiagnosticExportOptions[] {
    const currentDiagnostics = diagnosticViews();
    return checkedDiagnostics().flatMap(input => {
      const row = input.closest<HTMLElement>('.export-diagnostic-row')!;
      const panelId = row.querySelector<HTMLSelectElement>('select')?.value ?? row.dataset.panelId;
      const config = currentDiagnostics.find(item => item.panelId === panelId);
      return config ? [{ names: config.names, scale: config.scale, hidden: config.hidden }] : [];
    });
  }

  function selectedRange(view: ExportView): [number, number] | null {
    if (!view.colorrange) return null;
    if (view.colorrange[0] !== view.colorrange[1]) return view.colorrange;
    const value = view.colorrange[0];
    const padding = value === 0 ? 0.5 : Math.max(Math.abs(value) * 0.01, 2 ** -149);
    const limit = 3.4028234663852886e38;
    return [Math.max(-limit, value - padding), Math.min(limit, value + padding)];
  }

  function themeValue(): 'light' | 'dark' {
    return theme.value === 'current' ?
      (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light') :
      theme.value === 'dark' ? 'dark' : 'light';
  }

  function commonOptions(view: ExportView) {
    if (camera.value === 'current' && !view.camera) {
      state.setError('The selected panel has no current camera yet.');
      return null;
    }
    if (range.value === 'current' && !view.colorrange) {
      state.setError('The selected panel has no current colour range yet.');
      return null;
    }
    const current = selectedRange(view);
    return {
      field: view.field,
      diagnostics: diagnosticNames(),
      camera: camera.value === 'fit' ? 'fit' as const : view.camera!,
      ...(range.value === 'current' && current ? { colorrange: current } : {}),
      ...(view.colormap ? { colormap: view.colormap } : {}),
      phase_rendering: view.mode,
      size: [Number(width.value), Number(height.value)] as [number, number],
      fps: Number(fps.value),
      arrangement: arrangement.value === 'below' ? 'below' as const : 'right' as const,
      interface: view.overlays.interface,
      geometry: view.overlays.solid,
      blocks: view.overlays.blocks || view.overlays.levels,
      ...(diagnosticOptions().length ? { diagnostic_options: diagnosticOptions() } : {}),
    };
  }

  function storedOptions(view: ExportView): StoredExportOptions | null {
    const common = commonOptions(view);
    if (!common) return null;
    const window = parseTimeWindow(from.value, to.value);
    if (window.error) { state.setError(window.error); return null; }
    return {
      ...common,
      source: 'stored',
      frames: frameSelection(frames.value),
      ...(window.window ? { window: window.window } : {}),
      colorrange: range.value === 'current' ? common.colorrange ?? view.colorrange! : 'global',
      theme: themeValue(),
      duration: Number(duration.value),
    };
  }

  function liveOptions(view: ExportView): LiveExportOptions | null {
    const common = commonOptions(view);
    if (!common) return null;
    return {
      ...common,
      source: 'live',
      sim_per_second: Number(simPerSecond.value),
    };
  }

  function scheduleCommand(): ScheduleCommand | null {
    const view = views().find(item => item.id === panel.value);
    if (!view?.ready) return null;
    const window = parseTimeWindow(from.value, to.value);
    if (window.error) return null;
    const selectedFps = Number(fps.value);
    const selectedDuration = Number(duration.value);
    if (!Number.isInteger(selectedFps) || selectedFps < 1 || !(selectedDuration > 0)) return null;
    return {
      type: 'recording', action: 'schedule', source: 'stored', field: view.field,
      diagnostics: diagnosticNames(), frames: frameSelection(frames.value),
      ...(window.window ? { window: window.window } : {}),
      fps: selectedFps, duration: selectedDuration,
    };
  }

  function candidateTimes(): number[] {
    const window = parseTimeWindow(from.value, to.value);
    if (window.error) return [];
    return state.timeline.frames
      .filter(frame => (frames.value === 'all' || frame.kind !== 'preview') &&
        (!window.window || (frame.time >= window.window[0] && frame.time <= window.window[1])))
      .map(frame => frame.time);
  }

  function formatInput(value: number): string {
    return Number(value.toPrecision(4)).toString();
  }

  // Stored play time scales with the candidate count (2–20 s). Live aims at
  // ten seconds of video for the run's remaining time or, when that is
  // unknown, for the span of the stored frames.
  function applyStoredDurationDefault(): void {
    if (durationEdited) return;
    const count = candidateTimes().length;
    if (count === 0) return;
    duration.value = formatInput(Math.min(20, Math.max(2, count / Number(fps.value))));
  }

  function applyLiveRateDefault(): void {
    if (simRateEdited) return;
    const times = state.timeline.frames.map(frame => frame.time);
    const current = state.time;
    const span = state.finalTime !== null && current !== null && state.finalTime > current ?
      state.finalTime - current :
      times.length > 1 ? times[times.length - 1] - times[0] : 0;
    simPerSecond.value = formatInput(span > 0 ? span / 10 : 1);
  }

  function setScheduleText(text: string, schedule: RecordingSchedule | null = null): void {
    scheduleElement.textContent = text;
    // Clear the attributes written below; `dataset` also carries the element's
    // test id, which must survive so callers can still address the summary.
    for (const key of Object.keys(scheduleElement.dataset)) {
      if (key !== 'testid') delete scheduleElement.dataset[key];
    }
    if (schedule) {
      for (const [key, value] of Object.entries(scheduleAttributes(schedule))) {
        scheduleElement.setAttribute(key, value);
      }
      scheduleElement.dataset.tone = schedule.sparse ? 'warning' : 'ok';
    } else {
      scheduleElement.dataset.tone = 'idle';
    }
  }

  function renderSchedule(): void {
    if (isLive()) {
      const rate = Number(simPerSecond.value);
      const selectedFps = Number(fps.value);
      setScheduleText(rate > 0 && selectedFps > 0 ? liveSummary(rate, selectedFps) : 'Choose a positive rate and frame rate.');
      return;
    }
    const window = parseTimeWindow(from.value, to.value);
    if (window.error) { setScheduleText(window.error); return; }
    if (!lastSchedule) { setScheduleText('Computing the time-linear schedule…'); return; }
    setScheduleText(scheduleSummary(lastSchedule), lastSchedule);
  }

  function requestSchedule(command: ScheduleCommand, key: string): void {
    pendingKey = key;
    const generation = ++scheduleGeneration;
    setScheduleText('Computing the time-linear schedule…');
    client.schedule(command).then(schedule => {
      if (generation !== scheduleGeneration || isLive()) return;
      lastSchedule = schedule;
      scheduleKey = key;
      pendingKey = '';
      renderSchedule();
    }).catch(error => {
      pendingKey = '';
      if (generation !== scheduleGeneration) return;
      setScheduleText(error instanceof Error ? error.message : 'Could not compute the export schedule.');
    });
  }

  function maybeRequestSchedule(): void {
    if (!dialog.open || isLive() || state.recording.active) return;
    const command = scheduleCommand();
    if (!command) { renderSchedule(); return; }
    const key = JSON.stringify(command);
    // `update()` runs on every incoming frame; do not re-arm an unchanged
    // request, or the timer would never fire during a running simulation.
    if (key === scheduledKey || key === pendingKey) return;
    scheduledKey = key;
    window.clearTimeout(scheduleTimer);
    scheduleTimer = window.setTimeout(() => requestSchedule(command, key), 120);
  }

  function update(): void {
    const recording = state.recording;
    const live = isLive();
    for (const item of storedFields) item.hidden = live;
    simPerSecond.closest<HTMLElement>('.export-live')!.hidden = !live;
    for (const item of [panel, source, camera, arrangement, frames, range, from, to, duration, simPerSecond, width, height, fps, theme, format, path]) {
      item.disabled = recording.active;
    }
    frames.disabled = recording.active || live;
    from.disabled = recording.active || live;
    to.disabled = recording.active || live;
    duration.disabled = recording.active || live;
    simPerSecond.disabled = recording.active || !live;
    theme.disabled = recording.active || live;
    if (live) {
      range.options[0].hidden = true;
      range.options[1].hidden = false;
      if (range.value === 'global') range.value = 'auto';
    } else {
      range.options[0].hidden = false;
      range.options[1].hidden = true;
      if (range.value === 'auto') range.value = 'global';
    }
    for (const input of diagnostics.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')) input.disabled = recording.active;
    const selectedView = views().find(view => view.id === panel.value);
    note.textContent = (live
      ? 'Live export samples arriving frames on a uniform simulation-time grid from the start time until stopped; a sparse stream repeats the last frame. Theme applies to stored exports only.'
      : 'Stored export samples the candidate frames on a uniform simulation-time grid over the chosen window.') +
      ' Diagnostic exports use the selected panel’s visible series and y-axis scale.' +
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
    renderSchedule();
    maybeRequestSchedule();
  }
  function setDefaultPath(): void {
    const view = views().find(item => item.id === panel.value);
    const name = `${state.runs[0]?.name || 'run'}-${view?.field || 'field'}`.replace(/[^a-zA-Z0-9_.-]+/g, '-').replace(/^-+|-+$/g, '');
    path.value = `${name || 'studio'}.${format.value}`;
  }
  panel.addEventListener('change', () => { if (!pathWasEdited) setDefaultPath(); update(); });
  window.addEventListener('studio-view-change', update);
  source.addEventListener('change', () => {
    lastSchedule = null; scheduleKey = ''; scheduledKey = '';
    if (isLive()) applyLiveRateDefault(); else applyStoredDurationDefault();
    update();
  });
  for (const input of [from, to, fps, frames]) {
    input.addEventListener('input', () => {
      lastSchedule = null; scheduleKey = ''; scheduledKey = '';
      applyStoredDurationDefault();
      applyLiveRateDefault();
      update();
    });
  }
  duration.addEventListener('input', () => {
    durationEdited = true;
    lastSchedule = null; scheduleKey = ''; scheduledKey = '';
    update();
  });
  simPerSecond.addEventListener('input', () => {
    simRateEdited = true;
    update();
  });
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
    lastSchedule = null;
    scheduleKey = '';
    scheduledKey = '';
    applyStoredDurationDefault();
    applyLiveRateDefault();
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
    maybeRequestSchedule();
  });
  find<HTMLButtonElement>('.export-close').addEventListener('click', () => {
    window.clearTimeout(scheduleTimer);
    dialog.close();
  });
  stop.addEventListener('click', () => client.recording({ type: 'recording', action: 'stop' }));
  cancel.addEventListener('click', () => client.recording({ type: 'recording', action: 'cancel' }));
  preview.addEventListener('click', async () => {
    for (const input of [width, height, fps, duration, simPerSecond]) if (!input.reportValidity()) return;
    const view = views().find(item => item.id === panel.value);
    if (!view?.ready || view.frame === null) return;
    let frame = view.frame;
    let options: PreviewCommand | null = null;
    if (isLive()) {
      const stored = storedOptions(view);
      const live = liveOptions(view);
      if (!stored || !live) return;
      options = {
        type: 'recording', ...stored, source: 'stored', frames: 'keyframes',
        duration: 2, action: 'preview', frame,
      };
    } else {
      const command = scheduleCommand();
      if (!command) return;
      let schedule = lastSchedule;
      if (!schedule) {
        try { schedule = await client.schedule(command); }
        catch (error) { state.setError(error instanceof Error ? error.message : 'Could not compute the export schedule.'); return; }
        lastSchedule = schedule;
        renderSchedule();
      }
      frame = schedule.first_frame;
      const stored = storedOptions(view);
      if (!stored) return;
      options = { type: 'recording', ...stored, action: 'preview', frame };
    }
    if (!options) return;
    previewPending = true;
    previewImage.hidden = true;
    update();
    client.preview(options).then(url => {
      previewImage.src = url;
      previewImage.hidden = false;
    }).catch(error => {
      state.setError(error instanceof Error ? error.message : 'Could not render preview.');
    }).finally(() => { previewPending = false; update(); });
  });
  form.addEventListener('submit', event => {
    event.preventDefault();
    const view = views().find(item => item.id === panel.value);
    if (!view?.ready) return;
    if (isLive()) {
      const options = liveOptions(view);
      if (!options) return;
      client.recording({ type: 'recording', ...options, action: 'start', path: path.value.trim() });
    } else {
      const options = storedOptions(view);
      if (!options) return;
      client.recording({ type: 'recording', ...options, action: 'export', path: path.value.trim() });
    }
  });
  state.subscribe(update);
}
