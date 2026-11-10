// Pure camera, atlas and sampling operations shared by the renderer and panels.
// Physical y increases upwards; CSS-pixel y increases downwards.
export type Point = [number, number];
export type Bounds = [number, number, number, number]; // xmin, xmax, ymin, ymax
export type Plane = [number, number, number];
export type Placement = 'cell' | 'face_x' | 'face_y';
export type PhaseMode = 'cell' | 'plic';

export interface Camera { center: Point; scale: number }
export interface BlockData { level: number; i: number; j: number; origin: Point; spacing: Point }
export interface SolidData {
  full: boolean[];
  cut: number[];
  planes: Plane[];
  revision?: number;
  polygons?: { points: Point[]; triangles: number[][] };
}
export interface TopologyData {
  id?: number;
  generation?: number;
  blocks: BlockData[];
  block_size: Point;
  domain: { origin: Point; extent: Point };
  base_mesh?: Point;
  periodic?: [boolean, boolean];
  face_counts?: Point;
  face_tiles: [number[], number[]];
  solids: SolidData[];
  max_level?: number;
}
export interface PhaseData {
  liquid: boolean[];
  mixed: number[];
  planes: Plane[];
  segments: [Point, Point][];
  solid: number; // one-based; zero means no associated solid
}
export interface FieldData {
  topology: TopologyData;
  placement: Placement;
  // A single-valued field uses values. A two-phase field uses both phase
  // arrays and phase. Face arrays are in raw interior-then-boundary order.
  values?: Float32Array;
  liquid?: Float32Array;
  gas?: Float32Array;
  phase?: PhaseData | null;
  // Optional one-based associated solid override (0 means none). All solids
  // can be overlaid independently; only this one masks phase values.
  solid?: number;
}
export interface PickResult {
  world: Point;
  block: number; // one-based, as in Julia
  key: { level: number; i: number; j: number }; // block keys are zero-based
  cell: Point; // one-based within the block
  spacing: Point;
  index: number; // one-based cell/face value index; zero means missing face
  value: number;
  phase: 'liquid' | 'gas' | null;
  solid: boolean;
}

function positiveSize(width: number, height: number): void {
  if (!(width > 0 && height > 0 && Number.isFinite(width) && Number.isFinite(height))) {
    throw new Error('Camera viewport must have positive finite dimensions');
  }
}

/** The returned camera fully contains the requested world rectangle. */
export function fitCamera(bounds: Bounds, width: number, height: number): Camera {
  positiveSize(width, height);
  const [xmin, xmax, ymin, ymax] = bounds;
  if (!bounds.every(Number.isFinite) || !(xmax > xmin && ymax > ymin)) {
    throw new Error('Camera bounds must have positive finite extent');
  }
  return { center: [(xmin + xmax) / 2, (ymin + ymax) / 2], scale: Math.max((xmax - xmin) / width, (ymax - ymin) / height) };
}

export const boxFitCamera = fitCamera;

export function worldToScreen(camera: Camera, point: Point, width: number, height: number): Point {
  return [(point[0] - camera.center[0]) / camera.scale + width / 2,
    height / 2 - (point[1] - camera.center[1]) / camera.scale];
}

export function screenToWorld(camera: Camera, point: Point, width: number, height: number): Point {
  return [camera.center[0] + (point[0] - width / 2) * camera.scale,
    camera.center[1] - (point[1] - height / 2) * camera.scale];
}

/** factor multiplies world-units-per-pixel: greater than one zooms out. */
export function zoomCameraAt(camera: Camera, factor: number, anchor: Point, width: number, height: number): Camera {
  positiveSize(width, height);
  if (!(factor > 0 && Number.isFinite(factor) && camera.scale > 0 &&
      camera.scale * factor > 0 && Number.isFinite(camera.scale * factor))) {
    throw new Error('Camera zoom must have positive finite scale');
  }
  const scale = camera.scale * factor;
  return { center: [camera.center[0] + (anchor[0] - width / 2) * (camera.scale - scale),
    camera.center[1] - (anchor[1] - height / 2) * (camera.scale - scale)], scale };
}
/** Wheel up zooms in; deltaMode follows the browser's pixel/line/page units. */
export function wheelZoomCameraAt(
  camera: Camera, deltaY: number, deltaMode: number, anchor: Point, width: number, height: number,
): Camera {
  const delta = deltaY * (deltaMode === 1 ? 16 : deltaMode === 2 ? height : 1);
  return zoomCameraAt(camera, Math.exp(Math.max(-500, Math.min(500, delta)) * 0.0015), anchor, width, height);
}

/** Bounds of the stored interface, padded to show a useful band on either side. */
export function interfaceBandBounds(segments: PhaseData['segments'], domain: Bounds): Bounds | null {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const segment of segments) {
    for (const point of segment) {
      if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) continue;
      xmin = Math.min(xmin, point[0]); xmax = Math.max(xmax, point[0]);
      ymin = Math.min(ymin, point[1]); ymax = Math.max(ymax, point[1]);
    }
  }
  if (xmin === Infinity) return null;
  const spanX = xmax - xmin, spanY = ymax - ymin;
  const band = Math.max(0.25 * Math.max(spanX, spanY), 0.05 * Math.min(domain[1] - domain[0], domain[3] - domain[2]));
  const halfWidth = Math.max(spanX * 1.2, band) / 2;
  const halfHeight = Math.max(spanY * 1.2, band) / 2;
  const centerX = (xmin + xmax) / 2, centerY = (ymin + ymax) / 2;
  return [centerX - halfWidth, centerX + halfWidth, centerY - halfHeight, centerY + halfHeight];
}


/** Move the image by the pointer's CSS-pixel displacement. */
export function panCamera(camera: Camera, dx: number, dy: number): Camera {
  return { center: [camera.center[0] - dx * camera.scale, camera.center[1] + dy * camera.scale], scale: camera.scale };
}

export function finestCellCamera(camera: Camera, topology: TopologyData): Camera {
  let finest = Infinity;
  for (const block of topology.blocks) finest = Math.min(finest, block.spacing[0], block.spacing[1]);
  if (!(finest > 0 && Number.isFinite(finest))) throw new Error('Topology has no positive cell spacing');
  return { center: [...camera.center], scale: finest / 8 };
}

export function topologyBounds(topology: TopologyData): Bounds {
  const { origin, extent } = topology.domain;
  return [origin[0], origin[0] + extent[0], origin[1], origin[1] + extent[1]];
}

export interface AtlasPage { start: number; end: number; width: number; height: number }
export interface AtlasLayout { tileWidth: number; tileHeight: number; columns: number; pages: AtlasPage[] }

/** A block's value tile and its cell-plane tiles share the same atlas origin. */
export function atlasLayout(topology: TopologyData, placement: Placement, maximum: number): AtlasLayout {
  const [nx, ny] = topology.block_size;
  const tileWidth = placement === 'face_x' ? nx + 1 : placement === 'face_y' ? 2 * nx : nx;
  const tileHeight = placement === 'face_x' ? 2 * ny : placement === 'face_y' ? ny + 1 : ny;
  if (!Number.isInteger(maximum) || maximum < tileWidth || maximum < tileHeight ||
      !Number.isInteger(nx) || !Number.isInteger(ny) || nx <= 0 || ny <= 0) {
    throw new Error('Block tile exceeds the maximum texture size');
  }
  const count = topology.blocks.length;
  const columns = Math.min(Math.max(count, 1), Math.floor(maximum / tileWidth));
  const rows = Math.floor(maximum / tileHeight);
  const capacity = columns * rows;
  const pages: AtlasPage[] = [];
  for (let start = 0; start < count; start += capacity) {
    const end = Math.min(start + capacity, count);
    pages.push({ start, end, width: Math.min(columns, end - start) * tileWidth,
      height: Math.ceil((end - start) / columns) * tileHeight });
  }
  return { tileWidth, tileHeight, columns, pages };
}

export function atlasBlockOrigin(layout: AtlasLayout, page: AtlasPage, block: number): Point {
  if (block < page.start || block >= page.end) throw new Error('Block does not belong to atlas page');
  const local = block - page.start;
  return [(local % layout.columns) * layout.tileWidth, Math.floor(local / layout.columns) * layout.tileHeight];
}

function planeContains(plane: Plane, x: number, y: number): boolean {
  return plane[0] * x + plane[1] * y - plane[2] <= 0;
}

// Sparse indices are sorted in snapshots. Binary search preserves the CPU
// rasterizer's semantics without constructing a per-pick map.
function sparseIndex(indices: number[], value: number): number {
  let lo = 0, hi = indices.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (indices[mid] < value) lo = mid + 1; else hi = mid;
  }
  return lo < indices.length && indices[lo] === value ? lo : -1;
}

/** The solid associated with a phase field; unphased fields default to the first. */
export function associatedSolidIndex(field: FieldData): number {
  return field.solid ?? field.phase?.solid ?? (field.topology.solids.length ? 1 : 0);
}

function solidContains(solid: SolidData, cell: number, x: number, y: number): boolean {
  if (solid.full[cell - 1]) return true;
  const cut = sparseIndex(solid.cut, cell);
  return cut >= 0 && planeContains(solid.planes[cut], x, y);
}

/** Sample exactly as rasterize.jl, in world coordinates and without wrapping. */
export function pickField(field: FieldData, world: Point, mode: PhaseMode = 'cell'): PickResult | null {
  const { topology, placement } = field;
  const [nx, ny] = topology.block_size;
  for (let b = 0; b < topology.blocks.length; b++) {
    const block = topology.blocks[b];
    const x = (world[0] - block.origin[0]) / block.spacing[0];
    const y = (world[1] - block.origin[1]) / block.spacing[1];
    if (!(x >= 0 && x < nx && y >= 0 && y < ny)) continue;
    const ci = Math.floor(x), cj = Math.floor(y);
    const cellIndex = b * nx * ny + cj * nx + ci + 1;
    let index = cellIndex;
    if (placement === 'face_x') {
      const columns = nx + 1, rows = 2 * ny;
      const column = Math.min(nx, Math.floor(x + 0.5));
      const row = Math.min(rows - 1, Math.floor(2 * y));
      index = topology.face_tiles[0][b * columns * rows + row * columns + column] ?? 0;
    } else if (placement === 'face_y') {
      const columns = 2 * nx, rows = ny + 1;
      const column = Math.min(columns - 1, Math.floor(2 * x));
      const row = Math.min(ny, Math.floor(y + 0.5));
      index = topology.face_tiles[1][b * columns * rows + row * columns + column] ?? 0;
    }
    const localX = x - ci - 0.5, localY = y - cj - 0.5;
    const phase = field.phase;
    const solidData = topology.solids[associatedSolidIndex(field) - 1];
    const fullSolid = !!solidData?.full[cellIndex - 1];
    const associatedSolid = !!solidData && solidContains(solidData, cellIndex, localX, localY);
    const solid = topology.solids.some(item => solidContains(item, cellIndex, localX, localY));
    const twoPhase = !!field.liquid && !!field.gas && !!phase;
    let phaseName: PickResult['phase'] = null;
    let value = NaN;
    if (twoPhase && phase && field.liquid && field.gas) {
      let liquid = !!phase.liquid[cellIndex - 1];
      const mixed = mode === 'plic' ? sparseIndex(phase.mixed, cellIndex) : -1;
      if (mixed >= 0) liquid = planeContains(phase.planes[mixed], localX, localY);
      if (!fullSolid && !(mode === 'plic' && associatedSolid)) {
        phaseName = liquid ? 'liquid' : 'gas';
        if (index > 0) value = (liquid ? field.liquid : field.gas)[index - 1] ?? NaN;
      }
    } else if (index > 0) {
      value = field.values?.[index - 1] ?? NaN;
    }
    return { world: [...world], block: b + 1, key: { level: block.level, i: block.i, j: block.j },
      cell: [ci + 1, cj + 1], spacing: [...block.spacing], index, value, phase: phaseName, solid };
  }
  return null;
}

// Fixed anchors produce deterministic 256-entry RGBA8 lookup tables. The same
// tables drive both the WebGL texture and the CSS colourbar.
export const COLORMAP_NAMES = ['viridis', 'balance', 'magma', 'inferno', 'plasma', 'coolwarm', 'gray', 'turbo'] as const;
type RGB = [number, number, number];
const anchors: Record<(typeof COLORMAP_NAMES)[number], RGB[]> = {
  viridis: [[68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,141],[39,173,129],[92,200,99],[170,220,50],[253,231,37]],
  balance: [[24,28,67],[29,67,110],[34,105,141],[91,151,166],[241,236,228],[202,133,114],[166,73,68],[111,32,51],[60,9,32]],
  magma: [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]],
  inferno: [[0,0,4],[31,12,72],[85,15,109],[136,34,106],[186,54,85],[227,89,51],[249,140,10],[249,201,50],[252,255,164]],
  plasma: [[13,8,135],[75,3,161],[126,3,168],[168,34,150],[203,70,121],[229,107,93],[248,148,65],[253,195,40],[240,249,33]],
  coolwarm: [[59,76,192],[98,130,234],[141,176,254],[184,208,249],[221,221,221],[245,196,173],[244,154,123],[221,95,75],[180,4,38]],
  gray: [[0,0,0],[255,255,255]],
  turbo: [[48,18,59],[70,107,227],[40,188,235],[52,246,161],[164,252,60],[239,203,58],[251,126,33],[212,56,7],[122,4,3]],
};
const tables = new Map<string, Uint8Array>();

export function colormapLUT(name: string): Uint8Array {
  const key = Object.hasOwn(anchors, name) ? name : 'viridis';
  const cached = tables.get(key);
  if (cached) return cached;
  const stops = anchors[key as keyof typeof anchors];
  const table = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const p = i / 255 * (stops.length - 1), lo = Math.floor(p), hi = Math.min(lo + 1, stops.length - 1), f = p - lo;
    for (let c = 0; c < 3; c++) table[4 * i + c] = Math.round(stops[lo][c] * (1 - f) + stops[hi][c] * f);
    table[4 * i + 3] = 255;
  }
  tables.set(key, table);
  return table;
}

/** Return interpolated RGBA components in [0, 1], with nonfinite values transparent. */
export function mapColor(value: number, range: [number, number], name = 'viridis'): [number, number, number, number] {
  if (!Number.isFinite(value)) return [0, 0, 0, 0];
  const t = range[1] === range[0] ? 0.5 : Math.max(0, Math.min(1, (value - range[0]) / (range[1] - range[0])));
  const position = t * 255, lo = Math.floor(position), hi = Math.min(255, lo + 1), fraction = position - lo;
  const lut = colormapLUT(name);
  return [0, 1, 2, 3].map(c => (lut[4 * lo + c] * (1 - fraction) + lut[4 * hi + c] * fraction) / 255) as [number, number, number, number];
}

export function colormapGradient(name: string): string {
  const lut = colormapLUT(name);
  const stops: string[] = [];
  for (let i = 0; i < 256; i++) {
    stops.push(`rgb(${lut[4 * i]}, ${lut[4 * i + 1]}, ${lut[4 * i + 2]}) ${i / 255 * 100}%`);
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}
