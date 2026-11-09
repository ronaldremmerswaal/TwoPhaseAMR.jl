import './field-panel.css';
import { formatNumber, type PanelContext, type PanelDefinition, type PanelInstance } from './panels';
import type { StudioClient } from './client';
import type { FrameId, FrameSummary, NamedMetadata, RunSummary } from './types';
import {
  COLORMAP_NAMES, SharedRenderer, boxFitCamera, colormapGradient, finestCellCamera,
  fitCamera, interfaceBandBounds, panCamera, screenToWorld, visibleRegions, wheelZoomCameraAt, worldToScreen,
  type Camera, type FieldData, type PhaseData, type PickResult, type TopologyData,
} from './renderer';

export type ColorPolicy = 'global' | 'frame' | 'robust' | 'fixed';
export type CameraLinkGroup = 'A' | 'B' | 'none';
export interface FieldOverlays {
  interface: boolean;
  solid: boolean;
  blocks: boolean;
  cells: boolean;
  levels: boolean;
}
export interface FieldViewConfiguration {
  panelId: string;
  title: string;
  field: string;
  frame: FrameId | null;
  /** null means follow the shared playhead. */
  pinnedFrame: FrameId | null;
  mode: 'cell' | 'plic';
  camera: Camera | null;
  /** When true, subsequent panel resizes refit the domain. */
  fitMode: boolean;
  /** Visible world rectangle in [xmin, xmax, ymin, ymax] order. */
  worldRect: [number, number, number, number] | null;
  width: number;
  height: number;
  colormap: string;
  colorPolicy: ColorPolicy;
  /** The actual range displayed, after applying the symmetric modifier. */
  colorRange: [number, number] | null;
  /** Fixed policy input before applying the symmetric modifier. */
  fixedColorRange: [number, number] | null;
  symmetric: boolean;
  overlays: FieldOverlays;
  linkGroup: CameraLinkGroup;
  loading: boolean;
  /** The selected frame and field have both been loaded successfully. */
  ready: boolean;
}
export interface FieldPanelOptions {
  id?: string;
  title?: string;
  initialField?: string;
  /** Keyed by the dock panel instance ID, not the component ID. */
  initialFields?: Readonly<Record<string, string>>;
  linkGroup?: CameraLinkGroup;
  onViewChange?: (view: FieldViewConfiguration) => void;
}
export interface FieldPanelDefinition extends PanelDefinition {
  getView(panelId: string): FieldViewConfiguration | undefined;
  getViews(): FieldViewConfiguration[];
  /** Save configurations for matching IDs the next time those panels are created. */
  restoreViews(views: readonly FieldViewConfiguration[]): void;
}

type Range = [number, number];

/** Undefined denotes missing metadata; null denotes a known empty field. */
export function statisticsRange(frame: FrameSummary, field: string): Range | null | undefined {
  if (!frame.stats || typeof frame.stats !== 'object') return undefined;
  const entry = (frame.stats as Record<string, unknown>)[field];
  if (!entry || typeof entry !== 'object') return undefined;
  const stats = entry as Record<string, unknown>;
  if (stats.finite === 0) return null;
  if (typeof stats.minimum !== 'number' || !Number.isFinite(stats.minimum) ||
      typeof stats.maximum !== 'number' || !Number.isFinite(stats.maximum) ||
      stats.minimum > stats.maximum) return undefined;
  return [stats.minimum, stats.maximum];
}

/** Match the recorder's statistics: finite values from both intrinsic phases. */
export function valuesRange(...arrays: Float32Array[]): Range | null {
  let lo = Infinity, hi = -Infinity;
  for (const values of arrays) {
    for (const value of values) {
      if (Number.isFinite(value)) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
    }
  }
  return lo === Infinity ? null : [lo, hi];
}

/**
 * Nearest-rank 99.5th percentile from finite f32 values, without copying or
 * mutating the field arrays. A four-pass radix selection uses constant memory.
 * Signed ranges use the percentile of absolute values, not the largest outlier.
 */
export function robustValuesRange(symmetric: boolean, ...arrays: Float32Array[]): Range | null {
  const histogram = new Uint32Array(256);
  const float = new Float32Array(1);
  const bits = new Uint32Array(float.buffer);
  let count = 0, minimum = Infinity;
  for (const values of arrays) {
    for (const value of values) {
      if (!Number.isFinite(value)) continue;
      count++; minimum = Math.min(minimum, value);
      float[0] = symmetric ? Math.abs(value) : value;
      const raw = bits[0];
      const ordered = (raw & 0x80000000 ? ~raw : raw ^ 0x80000000) >>> 0;
      histogram[ordered >>> 24]++;
    }
  }
  if (!count) return null;
  let rank = Math.ceil(count * 0.995) - 1;
  let prefix = 0, mask = 0;
  for (let shift = 24; shift >= 0; shift -= 8) {
    if (shift !== 24) {
      histogram.fill(0);
      for (const values of arrays) {
        for (const value of values) {
          if (!Number.isFinite(value)) continue;
          float[0] = symmetric ? Math.abs(value) : value;
          const raw = bits[0];
          const ordered = (raw & 0x80000000 ? ~raw : raw ^ 0x80000000) >>> 0;
          if (((ordered & mask) >>> 0) === prefix) histogram[(ordered >>> shift) & 0xff]++;
        }
      }
    }
    let bucket = 0;
    while (rank >= histogram[bucket]) { rank -= histogram[bucket]; bucket++; }
    prefix = (prefix | (bucket << shift)) >>> 0;
    mask = (mask | (0xff << shift)) >>> 0;
  }
  bits[0] = (prefix & 0x80000000 ? prefix ^ 0x80000000 : ~prefix) >>> 0;
  const percentile = float[0];
  return symmetric ? [-percentile, percentile] : [minimum, percentile];
}

export function combineRanges(ranges: Iterable<Range | null | undefined>, symmetric = false): Range | null {
  let lo = Infinity, hi = -Infinity;
  for (const range of ranges) {
    if (!range) continue;
    lo = Math.min(lo, range[0]); hi = Math.max(hi, range[1]);
  }
  if (lo === Infinity) return null;
  if (symmetric) { const extent = Math.max(Math.abs(lo), Math.abs(hi)); return [-extent, extent]; }
  return [lo, hi];
}

function copyCamera(camera: Camera): Camera {
  return { center: [camera.center[0], camera.center[1]], scale: camera.scale };
}
function copyRange(range: Range | null): Range | null { return range ? [range[0], range[1]] : null; }
function integerId(value: FrameId): number {
  const id = typeof value === 'number' ? value : Number(value);
  if (!Number.isSafeInteger(id) || id < 0) throw new Error(`Invalid numeric frame or topology ID: ${String(value)}`);
  return id;
}
function domainBounds(domain: unknown): [number, number, number, number] | null {
  if (!domain || typeof domain !== 'object') return null;
  const { origin, extent } = domain as { origin?: unknown; extent?: unknown };
  if (!Array.isArray(origin) || !Array.isArray(extent) || origin.length !== 2 || extent.length !== 2 ||
      ![...origin, ...extent].every(value => typeof value === 'number' && Number.isFinite(value)) ||
      extent[0] <= 0 || extent[1] <= 0) return null;
  return [origin[0], origin[0] + extent[0], origin[1], origin[1] + extent[1]];
}
function fixedMetadataRange(field: NamedMetadata): Range | null {
  const value = field.colorrange;
  return Array.isArray(value) && value.length === 2 && value.every(n => typeof n === 'number' && Number.isFinite(n)) && value[0] <= value[1]
    ? [value[0], value[1]] : null;
}
function phaseIndex(field: NamedMetadata): number {
  return typeof field.interface === 'number' && Number.isSafeInteger(field.interface) && field.interface > 0 ? field.interface : 0;
}
function placement(field: NamedMetadata): FieldData['placement'] {
  if (field.placement === 'cell' || field.placement === 'face_x' || field.placement === 'face_y') return field.placement;
  throw new Error(`Unsupported field placement for ${field.name}: ${String(field.placement)}`);
}
function estimateBytes(value: unknown): number {
  if (value instanceof Float32Array) return value.byteLength;
  if (Array.isArray(value)) {
    let size = value.length * 8;
    for (const item of value) size += estimateBytes(item);
    return size;
  }
  if (value && typeof value === 'object') {
    let size = 0;
    for (const item of Object.values(value)) size += estimateBytes(item);
    return size;
  }
  return typeof value === 'number' ? 8 : typeof value === 'string' ? value.length * 2 : 0;
}

/** Shared, byte-bounded cache; concurrent panels reuse the same in-flight request. */
class DataCache {
  private readonly items = new Map<string, { value: unknown; bytes: number }>();
  private readonly pending = new Map<string, Promise<unknown>>();
  private bytes = 0;
  constructor(private readonly limit = 512 * 1024 * 1024) {}
  async get<T>(key: string, fetch: () => Promise<T>): Promise<T> {
    const cached = this.items.get(key);
    if (cached) {
      this.items.delete(key); this.items.set(key, cached);
      return cached.value as T;
    }
    const underway = this.pending.get(key);
    if (underway) return underway as Promise<T>;
    const promise = fetch().then(value => {
      const bytes = estimateBytes(value);
      if (bytes <= this.limit) {
        while (this.bytes + bytes > this.limit && this.items.size) {
          const oldest = this.items.keys().next().value as string;
          this.bytes -= this.items.get(oldest)!.bytes;
          this.items.delete(oldest);
        }
        this.items.set(key, { value, bytes }); this.bytes += bytes;
      }
      return value;
    }).finally(() => this.pending.delete(key));
    this.pending.set(key, promise);
    return promise;
  }
}

type LinkListener = (camera: Camera, owner: string | null) => void;
interface CameraGroup { camera: Camera | null; listeners: Map<string, LinkListener> }
class CameraLinks {
  private readonly groups = new Map<CameraLinkGroup, CameraGroup>();
  private group(name: CameraLinkGroup): CameraGroup {
    let group = this.groups.get(name);
    if (!group) { group = { camera: null, listeners: new Map() }; this.groups.set(name, group); }
    return group;
  }
  join(name: CameraLinkGroup, id: string, listener: LinkListener, camera: Camera | null): void {
    if (name === 'none') return;
    const group = this.group(name);
    group.listeners.set(id, listener);
    if (group.camera) listener(copyCamera(group.camera), null);
    else if (camera) this.publish(name, camera, id);
  }
  leave(name: CameraLinkGroup, id: string): void {
    if (name === 'none') return;
    const group = this.groups.get(name);
    if (!group) return;
    group.listeners.delete(id);
    if (!group.listeners.size) this.groups.delete(name);
  }
  publish(name: CameraLinkGroup, camera: Camera, owner: string): void {
    if (name === 'none') return;
    const group = this.group(name);
    group.camera = copyCamera(camera);
    for (const [id, listener] of group.listeners) if (id !== owner) listener(copyCamera(camera), owner);
  }
}
const rendererLinks = new WeakMap<SharedRenderer, CameraLinks>();

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function option(value: string, text: string): HTMLOptionElement {
  const node = document.createElement('option'); node.value = value; node.textContent = text; return node;
}
function labeled(text: string, control: HTMLElement): HTMLLabelElement {
  const label = element('label', 'field-label');
  label.append(element('span', 'field-label-text', text), control);
  return label;
}
function button(text: string, title: string): HTMLButtonElement {
  const node = element('button', '', text); node.type = 'button'; node.title = title; node.setAttribute('aria-label', title); return node;
}
function key(...parts: unknown[]): string { return JSON.stringify(parts); }
const cancelled = Symbol('superseded field request');
class FieldFetchError extends Error {}
async function requestFieldData<T>(request: () => Promise<T>): Promise<T> {
  try { return await request(); }
  catch (error) { throw new FieldFetchError(error instanceof Error ? error.message : String(error)); }
}

interface PanelHandle { getView(): FieldViewConfiguration; dispose(): void }

/**
 * Register this definition once. Use distinct dock IDs and initialFields to
 * select the initial field for each view. Cameras link across definitions that
 * use the same renderer; viewing history never issues a recording command.
 */
export function createFieldPanel(renderer: SharedRenderer, options: FieldPanelOptions = {}): FieldPanelDefinition {
  const views = new Map<string, PanelHandle>();
  const pendingRestores = new Map<string, FieldViewConfiguration>();
  const cache = new DataCache();
  const gradients = new Map<string, string>();
  const fallbackStats = new Map<string, Range | null>();
  const failedStats = new Map<string, string>();
  const stateEpochs = new WeakMap<PanelContext['state'], { run: RunSummary | undefined; epoch: number }>();
  let nextEpoch = 0;
  let links = rendererLinks.get(renderer);
  if (!links) { links = new CameraLinks(); rendererLinks.set(renderer, links); }
  const cameraLinks = links;

  function epochFor(context: PanelContext): number {
    const run = context.state.runs[0];
    const previous = stateEpochs.get(context.state);
    if (!previous) { const epoch = ++nextEpoch; stateEpochs.set(context.state, { run, epoch }); return epoch; }
    if (previous.run !== run) {
      for (const map of [fallbackStats, failedStats]) {
        for (const cacheKey of map.keys()) {
          if ((JSON.parse(cacheKey) as [number])[0] === previous.epoch) map.delete(cacheKey);
        }
      }
      previous.run = run; previous.epoch = ++nextEpoch;
    }
    return previous.epoch;
  }
  function rememberRange(id: string, range: Range | null): void { fallbackStats.set(id, range); }
  function frameKey(epoch: number, frame: FrameSummary, field: NamedMetadata): string {
    return key(epoch, frame.id, field.name, phaseIndex(field));
  }
  function fetchValues(client: StudioClient, epoch: number, frame: FrameSummary, field: NamedMetadata, phase?: 'liquid' | 'gas'): Promise<Float32Array> {
    return cache.get(key('field', epoch, frame.id, field.name, phase ?? ''),
      () => requestFieldData(() => client.getField(frame.id, field.name, 'f32', phase).then(result => result.values)));
  }

  function create(context: PanelContext): PanelInstance {
    const { state, client, panelId } = context;
    if (views.has(panelId)) throw new Error(`Field panel already exists: ${panelId}`);
    const restored = pendingRestores.get(panelId);
    pendingRestores.delete(panelId);
    const root = element('section', 'studio-field-panel');
    root.dataset.testid = 'field-panel'; root.setAttribute('aria-label', 'Field view');
    const toolbar = element('div', 'field-toolbar');
    const fieldPicker = element('details', 'field-picker-menu');
    const fieldSummary = element('summary', 'field-picker-summary', 'Field');
    fieldSummary.dataset.testid = 'field-picker-trigger'; fieldSummary.setAttribute('aria-label', 'Choose field');
    const fieldChoices = element('div', 'field-picker-options');
    const fieldSearch = element('input'); fieldSearch.type = 'search'; fieldSearch.placeholder = 'Search fields';
    fieldSearch.dataset.testid = 'field-search'; fieldSearch.setAttribute('aria-label', 'Search fields');
    const fieldSelect = element('select'); fieldSelect.dataset.testid = 'field-picker';
    fieldSelect.size = 6; fieldSelect.setAttribute('aria-label', 'Field');
    fieldChoices.append(fieldSearch, fieldSelect); fieldPicker.append(fieldSummary, fieldChoices);
    const linkSelect = element('select', 'field-link-badge'); linkSelect.dataset.testid = 'field-link-group';
    linkSelect.setAttribute('aria-label', 'Camera link group'); linkSelect.title = 'Camera link group';
    linkSelect.append(option('A', 'A'), option('B', 'B'), option('none', 'None'));
    const pinButton = button('Pin', 'Pin current frame'); pinButton.dataset.testid = 'field-pin';
    pinButton.setAttribute('aria-pressed', 'false');
    const fitButton = button('⤢', 'Fit domain (F)'); fitButton.dataset.testid = 'field-fit';
    const interfaceFitButton = button('≋', 'Fit interface band'); interfaceFitButton.dataset.testid = 'field-interface-fit';
    interfaceFitButton.hidden = true;
    const cellButton = button('1:1', 'Finest cell spans eight pixels (1)'); cellButton.dataset.testid = 'field-cell-zoom';
    const boxButton = button('□', 'Drag a box to zoom; Shift-drag also works');
    boxButton.dataset.testid = 'field-box-zoom'; boxButton.setAttribute('aria-pressed', 'false');
    const overlayMenu = element('details', 'field-overlay-menu');
    const overlaySummary = element('summary', 'field-icon', '◐'); overlaySummary.title = 'Overlays';
    overlaySummary.setAttribute('aria-label', 'Overlays'); overlaySummary.dataset.testid = 'field-overlays';
    overlayMenu.append(overlaySummary);
    const overlayControls = element('div', 'field-overlay-options field-popover');
    const overlays: FieldOverlays = restored ? { ...restored.overlays } : { interface: true, solid: true, blocks: false, cells: false, levels: false };
    const overlayInputs = new Map<keyof FieldOverlays, HTMLInputElement>();
    for (const [name, label] of [['interface', 'Interface'], ['solid', 'Solid'], ['blocks', 'Blocks'], ['cells', 'Cells'], ['levels', 'Levels']] as const) {
      const input = element('input'); input.type = 'checkbox'; input.checked = overlays[name]; input.dataset.testid = `field-overlay-${name}`;
      overlayInputs.set(name, input); overlayControls.append(labeled(label, input));
    }
    overlayMenu.append(overlayControls);
    const settingsMenu = element('details', 'field-settings-menu');
    const settingsSummary = element('summary', 'field-icon', '⚙'); settingsSummary.title = 'Field settings';
    settingsSummary.setAttribute('aria-label', 'Field settings'); settingsSummary.dataset.testid = 'field-settings';
    const settings = element('div', 'field-settings-options field-popover');
    const modeSelect = element('select'); modeSelect.dataset.testid = 'field-mode';
    modeSelect.append(option('cell', 'Cell'), option('plic', 'PLIC'));
    const policySelect = element('select'); policySelect.dataset.testid = 'field-color-policy';
    policySelect.append(option('robust', 'Frame 99.5%'), option('global', 'Global'), option('frame', 'Frame'), option('fixed', 'Fixed'));
    const paletteSelect = element('select'); paletteSelect.dataset.testid = 'field-colormap';
    for (const name of COLORMAP_NAMES) paletteSelect.append(option(name, name));
    const symmetricInput = element('input'); symmetricInput.type = 'checkbox';
    symmetricInput.dataset.testid = 'field-symmetric';
    const lockButton = button('Lock', 'Lock the current colour range'); lockButton.dataset.testid = 'field-color-lock';
    const rangeInputs = element('span', 'field-range-inputs');
    const lowInput = element('input'); lowInput.type = 'number'; lowInput.step = 'any'; lowInput.setAttribute('aria-label', 'Fixed colour minimum');
    const highInput = element('input'); highInput.type = 'number'; highInput.step = 'any'; highInput.setAttribute('aria-label', 'Fixed colour maximum');
    rangeInputs.append(lowInput, highInput);
    const colorStatus = element('span', 'field-color-status');
    settings.append(labeled('Mode', modeSelect), labeled('Colormap', paletteSelect), labeled('Range', policySelect),
      labeled('Symmetric', symmetricInput), lockButton, rangeInputs, colorStatus);
    settingsMenu.append(settingsSummary, settings);
    toolbar.append(fieldPicker, linkSelect, pinButton, fitButton, interfaceFitButton, cellButton, boxButton, overlayMenu, settingsMenu);
    const viewport = element('div', 'field-viewport');
    const plot = element('div', 'field-plot'); plot.dataset.testid = 'field-plot';
    plot.tabIndex = 0; plot.setAttribute('role', 'application');
    plot.setAttribute('aria-label', 'Field plot. Drag to pan, use the wheel to zoom, or shift-drag a zoom box.');
    const badge = button('', 'Unpin frame and follow playhead'); badge.className = 'field-badge';
    badge.dataset.testid = 'field-frame-badge'; badge.hidden = true;
    const colorbar = button('', 'Open field settings and colour range'); colorbar.className = 'field-colorbar';
    colorbar.dataset.testid = 'field-colorbar';
    const lowLabel = element('span', 'field-color-value');
    const gradient = element('span', 'field-color-gradient'); gradient.setAttribute('aria-hidden', 'true');
    const highLabel = element('span', 'field-color-value');
    colorbar.append(lowLabel, gradient, highLabel);
    viewport.append(plot, badge, colorbar);
    const status = element('div', 'field-status'); status.dataset.testid = 'field-status';
    const hoverStatus = element('span'); hoverStatus.dataset.testid = 'field-hover';
    const pinnedStatus = element('span'); pinnedStatus.dataset.testid = 'field-pinned'; pinnedStatus.setAttribute('aria-live', 'polite');
    const loadingStatus = element('span', '', 'Loading field…'); loadingStatus.hidden = true; loadingStatus.setAttribute('role', 'status');
    const errorStatus = element('span', 'field-error'); errorStatus.setAttribute('role', 'status');
    const retry = button('Retry', 'Retry loading the selected frame'); retry.dataset.testid = 'field-retry'; retry.hidden = true;
    status.append(hoverStatus, pinnedStatus, loadingStatus, errorStatus, retry);
    root.append(toolbar, viewport, status);

    // The shared renderer's canvas is above the dock. These pointer-transparent
    // annotations also live at document level, clipped to this plot's rectangle.
    const annotations = element('div', 'field-annotations');
    annotations.style.clipPath = 'inset(100%)';
    const selection = element('div', 'field-selection'); selection.hidden = true;
    const marker = element('div', 'field-marker'); marker.hidden = true;
    annotations.append(selection, marker); document.body.append(annotations);

    let selectedName = restored?.field ?? options.initialFields?.[panelId] ?? options.initialField ?? '';
    let selectedRun: RunSummary | undefined;
    let selectedFrame: FrameSummary | undefined;
    let pinnedFrame: FrameId | null = restored?.pinnedFrame ?? null;
    let field: NamedMetadata | undefined;
    let currentData: FieldData | null = null;
    let loadedKey = '';
    let camera: Camera | null = restored?.camera ? copyCamera(restored.camera) : null;
    let fitMode = restored?.fitMode ?? true;
    let fitPending = false;
    let preserveRestoredCamera = camera !== null && fitMode;
    let restoreFrame = 0;
    let restoreSize = '';
    let stableRestoreFrames = 0;
    let linkGroup: CameraLinkGroup = restored?.linkGroup ?? options.linkGroup ?? 'A';
    let mode: 'cell' | 'plic' = restored?.mode ?? 'cell';
    let colorPolicy: ColorPolicy = restored?.colorPolicy ?? 'robust';
    let fixedRange: Range | null = restored ? copyRange(restored.fixedColorRange) : null;
    let symmetric = restored?.symmetric ?? false;
    let colorRange: Range | null = restored ? copyRange(restored.colorRange) : null;
    let robustKey = '';
    let robustRange: Range | null = null;
    let palette = restored?.colormap ?? 'viridis';
    let restoreStylePending = !!restored;
    let shownPalette = '';
    let paletteWarning = '';
    let probeWorld: [number, number] | null = null;
    let boxArmed = false;
    let revision = 0;
    let running = false;
    let loading = false;
    let disposed = false;
    let stateSignature = '';
    let fieldOptionsSignature = '';
    let statsScopeEpoch = 0;
    let annotationClip = 'inset(100%)';
    let statsScopeSignature = '';
    let annotationFrame = 0;
    let lastErrorRetryable = false;
    let globalIncomplete = 0;
    let globalFailed = 0;
    let activeDrag: { pointer: number; start: [number, number]; last: [number, number]; box: boolean; moved: boolean } | null = null;
    let width = 0, height = 0;
    modeSelect.value = mode;
    paletteSelect.value = palette;
    policySelect.value = colorPolicy;
    symmetricInput.checked = symmetric;
    lowInput.value = fixedRange ? String(fixedRange[0]) : '';
    highInput.value = fixedRange ? String(fixedRange[1]) : '';

    function getView(): FieldViewConfiguration {
      const topLeft = camera && width > 0 && height > 0 ? screenToWorld(camera, [0, 0], width, height) : null;
      const bottomRight = camera && width > 0 && height > 0 ? screenToWorld(camera, [width, height], width, height) : null;
      return {
        panelId, title: field ? `${field.label || field.name}${field.units ? ` (${field.units})` : ''}` : selectedName || 'Field',
        field: selectedName, frame: selectedFrame?.id ?? null, pinnedFrame,
        mode, camera: camera ? copyCamera(camera) : null, fitMode,
        worldRect: topLeft && bottomRight ? [topLeft[0], bottomRight[0], bottomRight[1], topLeft[1]] : null,
        width, height, colormap: palette, colorPolicy, colorRange: copyRange(colorRange),
        fixedColorRange: copyRange(fixedRange), symmetric,
        overlays: { ...overlays }, linkGroup, loading, ready: !loading && !!currentData && loadedKey === desiredKey(),
      };
    }
    function notify(): void { options.onViewChange?.(getView()); }
    function renderView(): void {
      if (camera) {
        plot.dataset.cameraCenterX = String(camera.center[0]); plot.dataset.cameraCenterY = String(camera.center[1]);
        plot.dataset.cameraScale = String(camera.scale);
      } else {
        delete plot.dataset.cameraCenterX; delete plot.dataset.cameraCenterY; delete plot.dataset.cameraScale;
      }
      renderer.updateView(panelId, {
        field: currentData, mode, camera: camera ?? { center: [0, 0], scale: 1 },
        range: colorRange ?? [0, 1], colormap: palette, overlays: { ...overlays },
        theme: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
      });
      updateMarker();
    }
    function showError(error: string, retryable = false): void {
      lastErrorRetryable = retryable; errorStatus.textContent = error; retry.hidden = !retryable && !globalFailed;
    }
    function setLoading(value: boolean): void {
      if (loading === value) return;
      loading = value; loadingStatus.hidden = !value; root.classList.toggle('is-stale', value);
      root.setAttribute('aria-busy', String(value)); notify();
    }
    function updateBadge(): void {
      const pinned = pinnedFrame !== null && !!selectedFrame;
      badge.hidden = !pinned;
      badge.textContent = pinned ? `Pinned · t=${formatNumber(selectedFrame!.time)}${loading ? ' · loading' : ''}` : '';
      badge.title = pinned ? `#${selectedFrame!.id} · ${selectedFrame!.kind} · click to follow playhead` : '';
      badge.setAttribute('aria-label', pinned ? `Pinned at t=${formatNumber(selectedFrame!.time)}. Unpin and follow playhead.` : 'Unpin frame and follow playhead');
      pinButton.disabled = !selectedFrame;
      pinButton.setAttribute('aria-pressed', String(pinned));
      pinButton.setAttribute('aria-label', pinned ? 'Unpin frame and follow playhead' : 'Pin current frame');
      pinButton.title = pinned ? 'Unpin frame and follow playhead' : 'Pin current frame';
      interfaceFitButton.hidden = loading || loadedKey !== desiredKey() || !currentData?.phase?.segments?.length;
    }
    function updateAnnotationBounds(): DOMRect {
      const rect = plot.getBoundingClientRect();
      const positions = { left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` };
      for (const name of ['left', 'top', 'width', 'height'] as const) {
        if (annotations.style[name] !== positions[name]) annotations.style[name] = positions[name];
      }
      return rect;
    }
    function updateAnnotationClip(): void {
      annotationFrame = 0;
      if (disposed || (!probeWorld && !activeDrag?.box)) return;
      const rect = updateAnnotationBounds();
      const regions = visibleRegions(plot, rect);
      const paths = regions.map(region => {
        const left = region.left - rect.left, right = region.right - rect.left;
        const top = region.top - rect.top, bottom = region.bottom - rect.top;
        return `M ${left} ${top} H ${right} V ${bottom} H ${left} Z`;
      }).join(' ');
      const clip = paths ? `path(\"${paths}\")` : 'inset(100%)';
      if (clip !== annotationClip) { annotations.style.clipPath = clip; annotationClip = clip; }
    }
    function scheduleAnnotationClip(): void {
      if (!disposed && !annotationFrame && (probeWorld || activeDrag?.box)) {
        annotationFrame = requestAnimationFrame(updateAnnotationClip);
      }
    }
    function describePick(pick: PickResult | null): string {
      if (!pick) return 'Outside the mesh';
      const [x, y] = pick.world;
      const value = `value ${formatNumber(pick.value)}${pick.phase ? ` (${pick.phase})` : ''}`;
      return `x=${formatNumber(x)} y=${formatNumber(y)} · ${pick.solid ? `solid · ${value}` : value} · block (L${pick.key.level}, ${pick.key.i}, ${pick.key.j}) cell (${pick.cell[0]}, ${pick.cell[1]}) · Δ=(${formatNumber(pick.spacing[0])}, ${formatNumber(pick.spacing[1])})`;
    }
    function updateMarker(): void {
      if (!probeWorld || !camera || width <= 0 || height <= 0 || !currentData || loading) {
        marker.hidden = true;
        if (probeWorld && loading) pinnedStatus.textContent = 'Pinned point · loading';
        return;
      }
      const point = worldToScreen(camera, probeWorld, width, height);
      marker.hidden = point[0] < 0 || point[1] < 0 || point[0] > width || point[1] > height;
      marker.style.left = `${point[0]}px`; marker.style.top = `${point[1]}px`;
      scheduleAnnotationClip();
      const pick = marker.hidden ? null : renderer.pick(panelId, point[0], point[1]);
      pinnedStatus.textContent = pick ? `Pinned: ${describePick(pick)}` : 'Pinned point outside this view';
    }
    function updateCamera(next: Camera, fitted: boolean, publish = true): void {
      preserveRestoredCamera = false;
      if (restoreFrame) { cancelAnimationFrame(restoreFrame); restoreFrame = 0; }
      camera = copyCamera(next); fitMode = fitted; fitPending = false;
      renderView(); if (publish) cameraLinks.publish(linkGroup, camera, panelId); notify();
    }
    const receiveLinked: LinkListener = (next) => {
      if (disposed) return;
      if (camera && camera.scale === next.scale && camera.center[0] === next.center[0] && camera.center[1] === next.center[1]) return;
      updateCamera(next, false, false);
    };
    function fit(): void {
      fitMode = true;
      const rect = updateAnnotationBounds();
      width = rect.width; height = rect.height; scheduleAnnotationClip();
      const domain = currentData?.topology.domain ?? selectedRun?.domain;
      const bounds = domainBounds(domain);
      if (bounds && width > 0 && height > 0) updateCamera(fitCamera(bounds, width, height), true);
      else fitPending = true;
    }
    function fitInterface(): void {
      if (!currentData || loadedKey !== desiredKey() || loading) return;
      const domain = domainBounds(currentData.topology.domain);
      if (!domain) return;
      const bounds = interfaceBandBounds(currentData.phase?.segments ?? [], domain);
      const rect = updateAnnotationBounds();
      width = rect.width; height = rect.height; scheduleAnnotationClip();
      if (bounds && width > 0 && height > 0) updateCamera(fitCamera(bounds, width, height), false);
    }
    function settleRestoredCamera(): void {
      restoreFrame = 0;
      if (disposed || !preserveRestoredCamera) return;
      const rect = plot.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) { restoreSize = ''; stableRestoreFrames = 0; return; }
      const size = `${rect.width}:${rect.height}`;
      stableRestoreFrames = size === restoreSize ? stableRestoreFrames + 1 : 0;
      restoreSize = size;
      if (stableRestoreFrames >= 2) { preserveRestoredCamera = false; if (fitMode) fit(); }
      else restoreFrame = requestAnimationFrame(settleRestoredCamera);
    }
    function resize(): void {
      if (disposed) return;
      const rect = plot.getBoundingClientRect();
      const changed = width !== rect.width || height !== rect.height;
      width = rect.width; height = rect.height; updateAnnotationBounds(); scheduleAnnotationClip();
      if (changed && fitMode && !preserveRestoredCamera && width > 0 && height > 0) fit();
      else { updateMarker(); renderer.requestRender(); if (changed) notify(); }
      if (preserveRestoredCamera && width > 0 && height > 0 && !restoreFrame) {
        restoreFrame = requestAnimationFrame(settleRestoredCamera);
      }
    }
    function setLinkGroup(next: CameraLinkGroup): void {
      if (linkGroup === next) return;
      cameraLinks.leave(linkGroup, panelId); linkGroup = next;
      cameraLinks.join(linkGroup, panelId, receiveLinked, camera); notify();
    }
    function applyMetadata(next: NamedMetadata): void {
      selectedName = next.name; field = next;
      fixedRange = fixedMetadataRange(next);
      // The protocol has no derived-field flag. Use robust frame ranges for all
      // automatic fields; registered fixed ranges remain authoritative.
      colorPolicy = fixedRange ? 'fixed' : 'robust';
      symmetric = !fixedRange && (next.symmetric === true || next.colormap === 'balance' || next.colormap === 'coolwarm');
      const registered = typeof next.colormap === 'string' && next.colormap ? next.colormap : 'viridis';
      const supported = (COLORMAP_NAMES as readonly string[]).includes(registered);
      palette = supported ? registered : 'viridis';
      paletteWarning = supported ? '' : `${registered} unavailable; showing viridis`;
      if (restoreStylePending && restored?.field === next.name) {
        fixedRange = copyRange(restored.fixedColorRange); colorPolicy = restored.colorPolicy;
        symmetric = restored.symmetric; palette = restored.colormap;
        if (supported || palette !== 'viridis') paletteWarning = '';
      }
      restoreStylePending = false;
      paletteSelect.value = palette; policySelect.value = colorPolicy; symmetricInput.checked = symmetric;
      lowInput.value = fixedRange ? String(fixedRange[0]) : '';
      highInput.value = fixedRange ? String(fixedRange[1]) : '';
    }
    function frameStatsRange(frame: FrameSummary, epoch: number, meta: NamedMetadata): Range | null | undefined {
      const fromServer = statisticsRange(frame, meta.name);
      if (fromServer !== undefined) return fromServer;
      const id = frameKey(epoch, frame, meta);
      return fallbackStats.has(id) ? fallbackStats.get(id) : undefined;
    }
    function updateColor(): void {
      globalIncomplete = 0; globalFailed = 0;
      if (colorPolicy === 'robust') {
        const target = desiredKey();
        if (currentData && loadedKey === target && !loading) {
          const id = key(target, symmetric);
          if (robustKey !== id) {
            const arrays = currentData.values ? [currentData.values] : [currentData.liquid, currentData.gas]
              .filter((value): value is Float32Array => !!value);
            robustRange = robustValuesRange(symmetric, ...arrays);
            robustKey = id;
          }
          colorRange = copyRange(robustRange);
        } else colorRange = null;
      } else {
        let ranges: Array<Range | null | undefined> = [];
        if (colorPolicy === 'fixed') ranges = [fixedRange];
        else if (field) {
          const epoch = epochFor(context);
          const frames = colorPolicy === 'frame' ? (selectedFrame ? [selectedFrame] : []) : state.timeline.frames;
          ranges = frames.map(frame => {
            const value = frameStatsRange(frame, epoch, field!);
            if (value === undefined) { globalIncomplete++; if (failedStats.has(frameKey(epoch, frame, field!))) globalFailed++; }
            return value;
          });
        }
        colorRange = combineRanges(ranges, symmetric);
      }
      lowLabel.textContent = colorRange ? formatNumber(colorRange[0]) : '—';
      highLabel.textContent = colorRange ? formatNumber(colorRange[1]) : '—';
      colorbar.setAttribute('aria-label', colorRange
        ? `Colour range ${lowLabel.textContent} to ${highLabel.textContent}. Open field settings.`
        : 'Colour range unavailable. Open field settings.');
      if (shownPalette !== palette) {
        let css = gradients.get(palette);
        if (!css) { css = colormapGradient(palette); gradients.set(palette, css); }
        gradient.style.background = css; shownPalette = palette;
      }
      rangeInputs.hidden = colorPolicy !== 'fixed';
      const pending = globalIncomplete - globalFailed;
      colorStatus.textContent = [paletteWarning, pending ? `${pending} pending` : '', globalFailed ? `${globalFailed} unavailable` : ''].filter(Boolean).join(', ');
      colorStatus.title = globalFailed ? 'Colour range is incomplete because some frame data could not be retrieved.' : '';
      retry.hidden = !lastErrorRetryable && !globalFailed;
      renderView(); notify();
    }
    function refreshFieldChoices(fields: readonly NamedMetadata[]): void {
      const query = fieldSearch.value.trim().toLocaleLowerCase();
      const matches = fields.filter(item => `${item.name} ${item.label ?? ''} ${item.units ?? ''}`.toLocaleLowerCase().includes(query));
      fieldSelect.replaceChildren(...matches.map(item => option(item.name, `${item.label || item.name}${item.units ? ` (${item.units})` : ''}`)));
      fieldSelect.value = selectedName;
    }
    function syncFields(run: RunSummary | undefined): void {
      const fields = run?.fields ?? [];
      const signature = JSON.stringify(fields.map(item => [item.name, item.label, item.units, item.colormap, item.colorrange, item.interface, item.placement, item.symmetric]));
      const match = fields.find(item => item.name === selectedName) ?? fields[0];
      if (match && (field !== match || selectedRun !== run)) applyMetadata(match);
      if (!match) { field = undefined; if (run) selectedName = ''; }
      if (signature !== fieldOptionsSignature) {
        fieldOptionsSignature = signature;
        refreshFieldChoices(fields);
      } else fieldSelect.value = selectedName;
      fieldSummary.textContent = field?.label || selectedName || 'Field';
      fieldSummary.title = field ? `${field.label || field.name}${field.units ? ` (${field.units})` : ''}` : 'Choose field';
      fieldSummary.setAttribute('aria-label', `Choose field: ${fieldSummary.title}`);
    }
    function syncFrames(): void {
      const frames = state.timeline.frames;
      if (pinnedFrame !== null && state.connection === 'connected' && !frames.some(frame => frame.id === pinnedFrame)) pinnedFrame = null;
      const pinned = frames.find(frame => frame.id === pinnedFrame);
      selectedFrame = pinned ?? state.timeline.current;
    }
    function desiredKey(): string {
      return field && selectedFrame ? key(epochFor(context), selectedFrame.id, selectedFrame.topology, field.name, phaseIndex(field), field.placement) : '';
    }
    function check(version: number): void { if (disposed || version !== revision) throw cancelled; }
    async function loadSelected(version: number): Promise<void> {
      const meta = field, frame = selectedFrame;
      if (!meta || !frame) {
        currentData = null; loadedKey = ''; setLoading(false); showError(''); updateBadge(); renderView(); return;
      }
      const epoch = epochFor(context), target = desiredKey();
      if (loadedKey === target && currentData) { setLoading(false); updateBadge(); return; }
      setLoading(true); updateBadge(); showError('');
      const topologyId = integerId(frame.topology), frameId = integerId(frame.id);
      const topology = await cache.get(key('topology', epoch, frame.topology),
        () => requestFieldData(() => client.getData({ what: 'topology', topology: topologyId }).then(data => data as unknown as TopologyData)));
      check(version);
      const planes = await cache.get(key('planes', epoch, frame.id),
        () => requestFieldData(() => client.getData({ what: 'planes', frame: frameId }).then(data => data as unknown as { phases: PhaseData[] })));
      check(version);
      if (!Array.isArray(planes.phases)) throw new Error('Invalid phase data');
      const phaseNumber = phaseIndex(meta);
      const phase = planes.phases[phaseNumber ? phaseNumber - 1 : 0] ?? null;
      if (phaseNumber && !phase) throw new Error(`Missing interface ${phaseNumber} for ${meta.name}`);
      let data: FieldData;
      if (phaseNumber) {
        const liquid = await fetchValues(client, epoch, frame, meta, 'liquid'); check(version);
        const gas = await fetchValues(client, epoch, frame, meta, 'gas'); check(version);
        data = { topology, placement: placement(meta), liquid, gas, phase };
        if (statisticsRange(frame, meta.name) === undefined) rememberRange(frameKey(epoch, frame, meta), valuesRange(liquid, gas));
      } else {
        const values = await fetchValues(client, epoch, frame, meta); check(version);
        data = { topology, placement: placement(meta), values, phase };
        if (statisticsRange(frame, meta.name) === undefined) rememberRange(frameKey(epoch, frame, meta), valuesRange(values));
      }
      check(version);
      currentData = data; loadedKey = target;
      setLoading(false); updateBadge();
      if ((!camera || fitPending) && width > 0 && height > 0) fit();
      else renderView();
      updateColor();
    }
    async function scanMissing(version: number): Promise<void> {
      const meta = field;
      if (!meta || colorPolicy === 'fixed' || colorPolicy === 'robust') return;
      const epoch = epochFor(context);
      const frames = colorPolicy === 'frame' ? (selectedFrame ? [selectedFrame] : []) : state.timeline.frames;
      for (const frame of frames) {
        check(version);
        const id = frameKey(epoch, frame, meta);
        if (frameStatsRange(frame, epoch, meta) !== undefined || failedStats.has(id)) continue;
        try {
          const arrays: Float32Array[] = [];
          if (phaseIndex(meta)) {
            arrays.push(await fetchValues(client, epoch, frame, meta, 'liquid')); check(version);
            arrays.push(await fetchValues(client, epoch, frame, meta, 'gas')); check(version);
          } else { arrays.push(await fetchValues(client, epoch, frame, meta)); check(version); }
          rememberRange(id, valuesRange(...arrays));
        } catch (error) {
          if (error === cancelled) throw error;
          check(version); failedStats.set(id, error instanceof Error ? error.message : String(error));
        }
        updateColor();
      }
    }
    function schedule(): void {
      revision++;
      const target = desiredKey();
      if (target && (target !== loadedKey || !currentData)) {
        setLoading(true); updateBadge(); showError(''); hoverStatus.textContent = ''; updateMarker();
      } else if (!target) {
        currentData = null; loadedKey = ''; setLoading(false); updateBadge(); renderView();
      }
      if (running || disposed) return;
      running = true;
      void (async () => {
        while (!disposed) {
          const version = revision;
          try { await loadSelected(version); check(version); updateColor(); await scanMissing(version); }
          catch (error) {
            if (error !== cancelled && !disposed && version === revision) {
              currentData = null; loadedKey = ''; renderView();
              setLoading(false); updateBadge();
              showError(`Could not load field: ${error instanceof Error ? error.message : String(error)}`, error instanceof FieldFetchError);
            }
          }
          if (version === revision) break;
        }
        running = false;
      })();
    }
    function syncState(): void {
      if (disposed) return;
      const run = state.runs[0];
      syncFields(run); selectedRun = run; syncFrames();
      const epoch = epochFor(context);
      const scope = JSON.stringify([epoch, state.timeline.frames.map(frame => frame.id)]);
      if (scope !== statsScopeSignature) {
        const ids = new Set(state.timeline.frames.map(frame => frame.id));
        for (const map of [fallbackStats, failedStats]) {
          for (const cacheKey of map.keys()) {
            const [cachedEpoch, id] = JSON.parse(cacheKey) as [number, FrameId];
            if ((cachedEpoch === statsScopeEpoch && statsScopeEpoch !== epoch) || (cachedEpoch === epoch && !ids.has(id))) map.delete(cacheKey);
          }
        }
        statsScopeEpoch = epoch; statsScopeSignature = scope;
      }
      const frameStats = state.timeline.frames.map(frame => {
        const range = statisticsRange(frame, selectedName);
        return [frame.id, range === undefined ? 'missing' : range];
      });
      const signature = JSON.stringify([epoch, desiredKey(), state.connection, colorPolicy, pinnedFrame, frameStats]);
      updateBadge();
      if (signature !== stateSignature) { stateSignature = signature; updateColor(); schedule(); }
    }

    fieldSearch.addEventListener('input', () => refreshFieldChoices(selectedRun?.fields ?? []));
    fieldSelect.addEventListener('change', () => {
      const next = selectedRun?.fields.find(item => item.name === fieldSelect.value);
      if (!next) return;
      fieldPicker.open = false; fieldSearch.value = '';
      applyMetadata(next); refreshFieldChoices(selectedRun?.fields ?? []);
      stateSignature = ''; showError(''); syncState();
      fieldSummary.focus();
    });
    const menus: Array<[HTMLDetailsElement, HTMLElement]> = [
      [fieldPicker, fieldSummary], [overlayMenu, overlaySummary], [settingsMenu, settingsSummary],
    ];
    for (const [menu, summary] of menus) {
      menu.addEventListener('toggle', () => {
        if (!menu.open) return;
        for (const [other] of menus) if (other !== menu) other.open = false;
        if (menu === fieldPicker) fieldSearch.focus();
      });
      menu.addEventListener('keydown', event => {
        if (event.key === 'Escape' && menu.open) {
          menu.open = false; summary.focus(); event.preventDefault(); event.stopPropagation();
        }
      });
    }
    const closeMenusOnOutside = (event: PointerEvent): void => {
      if (event.target instanceof Node && !toolbar.contains(event.target) && !colorbar.contains(event.target)) {
        for (const [menu] of menus) menu.open = false;
      }
    };
    document.addEventListener('pointerdown', closeMenusOnOutside, true);
    colorbar.addEventListener('click', () => { settingsMenu.open = true; settingsSummary.focus(); });
    function togglePin(): void {
      pinnedFrame = pinnedFrame === null ? selectedFrame?.id ?? null : null;
      stateSignature = ''; syncState();
    }
    pinButton.addEventListener('click', togglePin);
    badge.addEventListener('click', togglePin);
    modeSelect.addEventListener('change', () => { mode = modeSelect.value === 'plic' ? 'plic' : 'cell'; renderView(); notify(); });
    linkSelect.value = linkGroup;
    linkSelect.addEventListener('change', () => setLinkGroup(linkSelect.value === 'A' ? 'A' : linkSelect.value === 'B' ? 'B' : 'none'));
    for (const [name, input] of overlayInputs) input.addEventListener('change', () => { overlays[name] = input.checked; renderView(); notify(); });
    fitButton.addEventListener('click', fit);
    interfaceFitButton.addEventListener('click', fitInterface);
    cellButton.addEventListener('click', () => { if (camera && currentData) updateCamera(finestCellCamera(camera, currentData.topology), false); });
    boxButton.addEventListener('click', () => { boxArmed = !boxArmed; boxButton.setAttribute('aria-pressed', String(boxArmed)); plot.focus(); });
    policySelect.addEventListener('change', () => {
      colorPolicy = policySelect.value === 'frame' ? 'frame' : policySelect.value === 'fixed' ? 'fixed'
        : policySelect.value === 'robust' ? 'robust' : 'global';
      if (colorPolicy === 'fixed' && !fixedRange) fixedRange = copyRange(colorRange);
      lowInput.value = fixedRange ? String(fixedRange[0]) : ''; highInput.value = fixedRange ? String(fixedRange[1]) : '';
      stateSignature = ''; syncState();
    });
    paletteSelect.addEventListener('change', () => { palette = paletteSelect.value; paletteWarning = ''; updateColor(); });
    symmetricInput.addEventListener('change', () => { symmetric = symmetricInput.checked; updateColor(); });
    lockButton.addEventListener('click', () => {
      if (!colorRange) return;
      fixedRange = copyRange(colorRange); colorPolicy = 'fixed'; policySelect.value = 'fixed';
      lowInput.value = String(fixedRange![0]); highInput.value = String(fixedRange![1]);
      stateSignature = ''; syncState();
    });
    const fixedChange = () => {
      const lo = lowInput.valueAsNumber, hi = highInput.valueAsNumber;
      const valid = Number.isFinite(lo) && Number.isFinite(hi) && lo <= hi;
      lowInput.setCustomValidity(valid ? '' : 'Enter a finite minimum not exceeding the maximum.');
      highInput.setCustomValidity(valid ? '' : 'Enter a finite maximum not below the minimum.');
      if (valid) { fixedRange = [lo, hi]; updateColor(); }
    };
    lowInput.addEventListener('change', fixedChange); highInput.addEventListener('change', fixedChange);
    retry.addEventListener('click', () => {
      failedStats.clear(); globalFailed = 0; loadedKey = ''; showError(''); updateColor(); schedule();
    });
    function localPoint(event: MouseEvent | PointerEvent | WheelEvent): [number, number] {
      const rect = plot.getBoundingClientRect(); return [event.clientX - rect.left, event.clientY - rect.top];
    }
    plot.addEventListener('wheel', event => {
      resize();
      if (!camera || width <= 0 || height <= 0 || event.deltaY === 0) return;
      event.preventDefault();
      updateCamera(wheelZoomCameraAt(camera, event.deltaY, event.deltaMode, localPoint(event), width, height), false);
    }, { passive: false });
    plot.addEventListener('pointerdown', event => {
      if (event.button !== 0 || !camera) return;
      plot.focus(); plot.setPointerCapture(event.pointerId);
      const point = localPoint(event);
      activeDrag = { pointer: event.pointerId, start: point, last: point, box: boxArmed || event.shiftKey, moved: false };
      if (activeDrag.box) scheduleAnnotationClip();
      if (activeDrag.box) { selection.hidden = false; selection.style.left = `${point[0]}px`; selection.style.top = `${point[1]}px`; selection.style.width = '0px'; selection.style.height = '0px'; }
      event.preventDefault();
    });
    plot.addEventListener('pointermove', event => {
      const point = localPoint(event);
      if (activeDrag && activeDrag.pointer === event.pointerId && camera) {
        const drag = activeDrag;
        drag.moved ||= Math.hypot(point[0] - drag.start[0], point[1] - drag.start[1]) >= 3;
        if (drag.box) {
          selection.style.left = `${Math.min(point[0], drag.start[0])}px`; selection.style.top = `${Math.min(point[1], drag.start[1])}px`;
          selection.style.width = `${Math.abs(point[0] - drag.start[0])}px`; selection.style.height = `${Math.abs(point[1] - drag.start[1])}px`;
        } else if (drag.moved) updateCamera(panCamera(camera, point[0] - drag.last[0], point[1] - drag.last[1]), false);
        drag.last = point;
      }
      hoverStatus.textContent = loading ? '' : describePick(renderer.pick(panelId, point[0], point[1]));
    });
    function endDrag(event: PointerEvent, cancel: boolean): void {
      if (!activeDrag || activeDrag.pointer !== event.pointerId) return;
      const drag = activeDrag, point = localPoint(event); activeDrag = null; selection.hidden = true;
      if (!cancel && camera && drag.box && drag.moved && width > 0 && height > 0) {
        const first = screenToWorld(camera, drag.start, width, height), last = screenToWorld(camera, point, width, height);
        const bounds: [number, number, number, number] = [Math.min(first[0], last[0]), Math.max(first[0], last[0]), Math.min(first[1], last[1]), Math.max(first[1], last[1])];
        if (bounds[0] < bounds[1] && bounds[2] < bounds[3]) updateCamera(boxFitCamera(bounds, width, height), false);
      } else if (!cancel && !drag.moved && !drag.box && !loading) {
        const pick = renderer.pick(panelId, point[0], point[1]);
        probeWorld = pick ? [pick.world[0], pick.world[1]] : null;
        if (probeWorld) scheduleAnnotationClip();
        pinnedStatus.textContent = pick ? `Pinned: ${describePick(pick)}` : 'Outside the mesh'; updateMarker();
      }
      if (drag.box && boxArmed) { boxArmed = false; boxButton.setAttribute('aria-pressed', 'false'); }
    }
    plot.addEventListener('pointerup', event => endDrag(event, false));
    plot.addEventListener('pointercancel', event => endDrag(event, true));
    plot.addEventListener('lostpointercapture', event => endDrag(event, true));
    plot.addEventListener('pointerleave', () => { if (!activeDrag) hoverStatus.textContent = ''; });
    plot.addEventListener('keydown', event => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const name = event.key.toLowerCase();
      if (name === 'f') fit();
      else if (name === '1' && camera && currentData) updateCamera(finestCellCamera(camera, currentData.topology), false);
      else if (name === 'c') {
        policySelect.value = colorPolicy === 'robust' ? 'frame' : colorPolicy === 'frame' ? 'global' : colorPolicy === 'global' ? 'fixed' : 'robust';
        policySelect.dispatchEvent(new Event('change'));
      } else if (name === 'i' || name === 'g' || name === 'e') {
        const input = overlayInputs.get(name === 'i' ? 'interface' : name === 'g' ? 'blocks' : 'cells')!;
        input.checked = !input.checked; input.dispatchEvent(new Event('change'));
      } else if (name === 'escape') {
        activeDrag = null; selection.hidden = true; boxArmed = false; boxButton.setAttribute('aria-pressed', 'false');
      } else return;
      event.preventDefault(); event.stopPropagation();
    });

    renderer.registerView(panelId, plot);
    cameraLinks.join(linkGroup, panelId, receiveLinked, camera);
    if (restored?.camera) {
      camera = copyCamera(restored.camera); fitMode = restored.fitMode; preserveRestoredCamera = fitMode;
      cameraLinks.publish(linkGroup, camera, panelId);
    }
    const unsubscribe = state.subscribe(syncState);
    const resizeObserver = new ResizeObserver(resize); resizeObserver.observe(plot);
    const themeObserver = new MutationObserver(renderView);
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    const clipObserver = new MutationObserver(records => {
      if (records.some(record => {
        if (record.target === renderer.canvas || (record.target instanceof Element && record.target.closest('.field-annotations'))) return false;
        return record.type === 'attributes' || [...record.addedNodes, ...record.removedNodes].some(node => node.nodeType === Node.ELEMENT_NODE);
      })) scheduleAnnotationClip();
    });
    clipObserver.observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ['class', 'style', 'hidden', 'open'] });
    const geometryChanged = () => { updateAnnotationBounds(); scheduleAnnotationClip(); renderer.requestRender(); };
    window.addEventListener('scroll', geometryChanged, true);
    window.addEventListener('resize', geometryChanged);
    const handle: PanelHandle = {
      getView,
      dispose() {
        if (disposed) return;
        disposed = true; revision++; unsubscribe(); resizeObserver.disconnect(); themeObserver.disconnect(); clipObserver.disconnect();
        if (annotationFrame) cancelAnimationFrame(annotationFrame);
        if (restoreFrame) cancelAnimationFrame(restoreFrame);
        document.removeEventListener('pointerdown', closeMenusOnOutside, true);
        window.removeEventListener('scroll', geometryChanged, true); window.removeEventListener('resize', geometryChanged);
        cameraLinks.leave(linkGroup, panelId); renderer.removeView(panelId); annotations.remove(); views.delete(panelId);
      },
    };
    views.set(panelId, handle);
    renderView(); syncState(); resize();
    return { element: root, dispose: handle.dispose };
  }

  return {
    id: options.id ?? 'field', title: options.title ?? 'Field', create,
    getView(panelId) { return views.get(panelId)?.getView(); },
    getViews() { return [...views.values()].map(view => view.getView()); },
    restoreViews(configurations) {
      for (const view of configurations) pendingRestores.set(view.panelId, {
        ...view,
        camera: view.camera ? copyCamera(view.camera) : null,
        worldRect: view.worldRect ? [view.worldRect[0], view.worldRect[1], view.worldRect[2], view.worldRect[3]] : null,
        colorRange: copyRange(view.colorRange), fixedColorRange: copyRange(view.fixedColorRange),
        overlays: { ...view.overlays },
      });
    },
  };
}
