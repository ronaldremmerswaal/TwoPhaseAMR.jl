import {
  associatedSolidIndex, atlasBlockOrigin, atlasLayout, colormapLUT, pickField, screenToWorld,
  type AtlasLayout, type AtlasPage, type Camera, type FieldData, type PhaseMode, type PickResult,
} from './renderer-math';
export * from './renderer-math';

export interface Overlays {
  interface?: boolean;
  solid?: boolean;
  blocks?: boolean;
  cells?: boolean;
  levels?: boolean;
}
export interface ViewOptions {
  field: FieldData | null;
  camera: Camera;
  mode: PhaseMode;
  range: [number, number];
  colormap: string;
  overlays: Overlays;
  theme: 'light' | 'dark';
}
interface View { element: HTMLElement; options: ViewOptions }
export interface Rect { left: number; top: number; right: number; bottom: number }
interface PageGPU {
  vertices: WebGLBuffer;
  count: number;
  values: WebGLTexture;
  gas: WebGLTexture | null;
  phase: WebGLTexture | null;
  solids: WebGLTexture[];
}
interface FieldGPU { source: FieldData; pages: PageGPU[]; segments: WebGLBuffer | null; segmentCount: number }
interface Program { handle: WebGLProgram; uniforms: Map<string, WebGLUniformLocation | null> }

const BLOCK_VERTEX = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_origin;
layout(location=1) in vec2 a_spacing;
layout(location=2) in vec2 a_tile;
layout(location=3) in float a_level;
uniform vec2 u_center;
uniform vec2 u_viewSize;
uniform float u_scale;
uniform ivec2 u_blockSize;
out vec2 v_local;
flat out ivec2 v_tile;
flat out vec2 v_spacing;
flat out float v_level;
void main() {
  vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  v_local = corner * vec2(u_blockSize);
  v_tile = ivec2(a_tile);
  v_spacing = a_spacing;
  v_level = a_level;
  vec2 world = a_origin + v_local * a_spacing;
  gl_Position = vec4((world - u_center) * 2.0 / (u_scale * u_viewSize), 0.0, 1.0);
}`;

const BLOCK_FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_values;
uniform highp sampler2D u_gas;
uniform highp sampler2D u_phase;
uniform highp sampler2D u_solid;
uniform sampler2D u_lut;
uniform ivec2 u_blockSize;
uniform int u_placement;
uniform bool u_twoPhase;
uniform bool u_hasSolid;
uniform bool u_sharp;
uniform bool u_solidPass;
uniform bool u_cells;
uniform bool u_blocks;
uniform bool u_levels;
uniform float u_scale;
uniform vec2 u_range;
uniform vec3 u_interfaceColor;
uniform vec3 u_blockColor;
uniform vec3 u_solidColor;
in vec2 v_local;
flat in ivec2 v_tile;
flat in vec2 v_spacing;
flat in float v_level;
out vec4 outColor;
vec4 over(vec4 base, vec3 color, float alpha) {
  return vec4(color * alpha + base.rgb * (1.0 - alpha), alpha + base.a * (1.0 - alpha));
}
vec3 levelColor(float level) {
  int k = int(mod(level, 6.0));
  if (k == 0) return vec3(0.16, 0.48, 0.85);
  if (k == 1) return vec3(0.07, 0.65, 0.49);
  if (k == 2) return vec3(0.93, 0.57, 0.10);
  if (k == 3) return vec3(0.83, 0.26, 0.31);
  if (k == 4) return vec3(0.64, 0.36, 0.79);
  return vec3(0.10, 0.66, 0.76);
}
void main() {
  ivec2 cell = clamp(ivec2(floor(v_local)), ivec2(0), u_blockSize - 1);
  vec2 local = v_local - vec2(cell) - 0.5;
  ivec2 texel = cell;
  if (u_placement == 1) {
    texel = ivec2(clamp(int(floor(v_local.x + 0.5)), 0, u_blockSize.x),
                  clamp(int(floor(2.0 * v_local.y)), 0, 2 * u_blockSize.y - 1));
  } else if (u_placement == 2) {
    texel = ivec2(clamp(int(floor(2.0 * v_local.x)), 0, 2 * u_blockSize.x - 1),
                  clamp(int(floor(v_local.y + 0.5)), 0, u_blockSize.y));
  }
  vec4 solid = u_hasSolid ? texelFetch(u_solid, v_tile + cell, 0) : vec4(0.0);
  float solidSide = dot(solid.xy, local) - solid.z;
  bool fullSolid = solid.w == 1.0;
  bool cutSolid = solid.w == 2.0 && solidSide <= 0.0;
  bool liquid = true;
  if (u_twoPhase && !u_solidPass) {
    vec4 phase = texelFetch(u_phase, v_tile + cell, 0);
    liquid = phase.w == 1.0;
    // For mixed cells the majority mask is encoded in the sign of the flag.
    if (abs(phase.w) == 2.0) liquid = u_sharp ? dot(phase.xy, local) - phase.z <= 0.0 : phase.w > 0.0;
  }
  vec4 color = vec4(0.0);
  if (u_solidPass) {
    if (fullSolid || cutSolid) color = vec4(u_solidColor, 1.0);
  } else if (!(u_twoPhase && (fullSolid || (u_sharp && cutSolid)))) {
    float value = (u_twoPhase && !liquid) ? texelFetch(u_gas, v_tile + texel, 0).r : texelFetch(u_values, v_tile + texel, 0).r;
    if (!isnan(value) && !isinf(value)) {
      float t = u_range.y == u_range.x ? 0.5 : clamp((value - u_range.x) / (u_range.y - u_range.x), 0.0, 1.0);
      color = vec4(texture(u_lut, vec2((t * 255.0 + 0.5) / 256.0, 0.5)).rgb, 1.0);
    }
  }
  if (u_solidPass && solid.w == 2.0) {
    float gradient = length(solid.xy / v_spacing);
    if (gradient > 0.0 && abs(solidSide) / gradient / u_scale < 0.75) color = over(color, u_interfaceColor, 0.85);
  }
  vec2 edge = min(fract(v_local), 1.0 - fract(v_local)) * v_spacing / u_scale;
  float distanceToCellEdge = min(edge.x, edge.y);
  if (u_solidPass && fullSolid && distanceToCellEdge < 0.75) {
    bool boundary = false;
    if (edge.x <= edge.y) {
      int stepX = fract(v_local.x) < 0.5 ? -1 : 1;
      int neighbor = cell.x + stepX;
      if (neighbor >= 0 && neighbor < u_blockSize.x) boundary = texelFetch(u_solid, v_tile + cell + ivec2(stepX, 0), 0).w != 1.0;
    } else {
      int stepY = fract(v_local.y) < 0.5 ? -1 : 1;
      int neighbor = cell.y + stepY;
      if (neighbor >= 0 && neighbor < u_blockSize.y) boundary = texelFetch(u_solid, v_tile + cell + ivec2(0, stepY), 0).w != 1.0;
    }
    if (boundary) color = over(color, u_interfaceColor, 0.85);
  }
  if (u_solidPass && color.a == 0.0) discard;
  float cellPixels = min(v_spacing.x, v_spacing.y) / u_scale;
  if (u_cells && distanceToCellEdge < 0.6) color = over(color, u_blockColor, 0.5 * smoothstep(6.0, 10.0, cellPixels));
  vec2 blockEdge = min(v_local, vec2(u_blockSize) - v_local) * v_spacing / u_scale;
  if ((u_blocks || u_levels) && min(blockEdge.x, blockEdge.y) < 0.9) {
    color = over(color, u_levels ? levelColor(v_level) : u_blockColor, 0.9);
  }
  if (color.a == 0.0) discard;
  outColor = color;
}`;

const LINE_VERTEX = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_from;
layout(location=1) in vec2 a_to;
uniform vec2 u_center;
uniform vec2 u_viewSize;
uniform float u_scale;
uniform float u_width;
void main() {
  vec2 delta = a_to - a_from;
  float lengthWorld = length(delta);
  vec2 normal = lengthWorld > 0.0 ? vec2(-delta.y, delta.x) / lengthWorld : vec2(0.0);
  float along = float((gl_VertexID >> 1) & 1);
  float side = float(gl_VertexID & 1) * 2.0 - 1.0;
  vec2 world = mix(a_from, a_to, along) + normal * side * u_width * u_scale * 0.5;
  gl_Position = vec4((world - u_center) * 2.0 / (u_scale * u_viewSize), 0.0, 1.0);
}`;
const LINE_FRAGMENT = `#version 300 es
precision highp float;
uniform vec3 u_color;
out vec4 outColor;
void main() { outColor = vec4(u_color, 1.0); }`;

function compile(gl: WebGL2RenderingContext, vertexSource: string, fragmentSource: string): Program {
  const shader = (type: number, source: string): WebGLShader => {
    const result = gl.createShader(type);
    if (!result) throw new Error('Unable to allocate a Studio shader');
    gl.shaderSource(result, source);
    gl.compileShader(result);
    if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) {
      const error = gl.getShaderInfoLog(result);
      gl.deleteShader(result);
      throw new Error(`Studio shader compilation failed: ${error}`);
    }
    return result;
  };
  const vs = shader(gl.VERTEX_SHADER, vertexSource), fs = shader(gl.FRAGMENT_SHADER, fragmentSource);
  const handle = gl.createProgram();
  if (!handle) { gl.deleteShader(vs); gl.deleteShader(fs); throw new Error('Unable to allocate a Studio program'); }
  gl.attachShader(handle, vs); gl.attachShader(handle, fs); gl.linkProgram(handle);
  gl.deleteShader(vs); gl.deleteShader(fs);
  if (!gl.getProgramParameter(handle, gl.LINK_STATUS)) {
    const error = gl.getProgramInfoLog(handle); gl.deleteProgram(handle);
    throw new Error(`Studio shader link failed: ${error}`);
  }
  return { handle, uniforms: new Map() };
}
function uniform(gl: WebGL2RenderingContext, program: Program, name: string): WebGLUniformLocation | null {
  if (!program.uniforms.has(name)) program.uniforms.set(name, gl.getUniformLocation(program.handle, name));
  return program.uniforms.get(name) ?? null;
}
function texture(gl: WebGL2RenderingContext, width: number, height: number, channels: 1 | 4, data: Float32Array): WebGLTexture {
  const result = gl.createTexture();
  if (!result) throw new Error('Unable to allocate a Studio texture');
  gl.bindTexture(gl.TEXTURE_2D, result);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texStorage2D(gl.TEXTURE_2D, 1, channels === 1 ? gl.R32F : gl.RGBA32F, width, height);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, channels === 1 ? gl.RED : gl.RGBA, gl.FLOAT, data);
  return result;
}
function buffer(gl: WebGL2RenderingContext, data: Float32Array): WebGLBuffer {
  const result = gl.createBuffer();
  if (!result) throw new Error('Unable to allocate a Studio buffer');
  gl.bindBuffer(gl.ARRAY_BUFFER, result);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return result;
}
function sameField(a: FieldData, b: FieldData): boolean {
  return a.topology === b.topology && a.placement === b.placement && a.values === b.values &&
    a.liquid === b.liquid && a.gas === b.gas && a.phase === b.phase;
}

/** Scatter raw cell or face vectors into their block tiles. */
export function scatterValues(field: FieldData, source: Float32Array, layout: AtlasLayout, page: AtlasPage): Float32Array {
  const result = new Float32Array(page.width * page.height);
  result.fill(NaN);
  const [nx, ny] = field.topology.block_size;
  const columns = layout.tileWidth, rows = layout.tileHeight;
  const tiles = field.placement === 'face_x' ? field.topology.face_tiles[0] : field.topology.face_tiles[1];
  for (let block = page.start; block < page.end; block++) {
    const [ox, oy] = atlasBlockOrigin(layout, page, block);
    if (field.placement === 'cell') {
      for (let j = 0; j < ny; j++) {
        const begin = block * nx * ny + j * nx;
        result.set(source.subarray(begin, begin + nx), (oy + j) * page.width + ox);
      }
    } else {
      const base = block * columns * rows;
      for (let j = 0; j < rows; j++) for (let i = 0; i < columns; i++) {
        const index = tiles[base + j * columns + i];
        if (index > 0) result[(oy + j) * page.width + ox + i] = source[index - 1] ?? NaN;
      }
    }
  }
  return result;
}

function cellOffset(layout: AtlasLayout, page: AtlasPage, cell: number, nx: number, ny: number): number {
  const zero = cell - 1, block = Math.floor(zero / (nx * ny));
  if (block < page.start || block >= page.end) return -1;
  const local = zero % (nx * ny), slot = block - page.start;
  const x = (slot % layout.columns) * layout.tileWidth + local % nx;
  const y = Math.floor(slot / layout.columns) * layout.tileHeight + Math.floor(local / nx);
  return 4 * (y * page.width + x);
}
function buildField(gl: WebGL2RenderingContext, field: FieldData, maximum: number): FieldGPU {
  const twoPhase = !!field.liquid && !!field.gas && !!field.phase;
  if (!twoPhase && !field.values) throw new Error('A field needs either cell/face values or both phase arrays and phase planes');
  const topology = field.topology, [nx, ny] = topology.block_size;
  const phase = field.phase;
  const layout = atlasLayout(topology, field.placement, maximum);
  const pages: PageGPU[] = [];
  for (const page of layout.pages) {
    const attributes = new Float32Array((page.end - page.start) * 7);
    for (let b = page.start; b < page.end; b++) {
      const block = topology.blocks[b], [x, y] = atlasBlockOrigin(layout, page, b);
      attributes.set([block.origin[0], block.origin[1], block.spacing[0], block.spacing[1], x, y, block.level], (b - page.start) * 7);
    }
    const values = texture(gl, page.width, page.height, 1, scatterValues(field, twoPhase ? field.liquid! : field.values!, layout, page));
    const gas = twoPhase ? texture(gl, page.width, page.height, 1, scatterValues(field, field.gas!, layout, page)) : null;
    let phaseTexture: WebGLTexture | null = null;
    if (twoPhase && phase) {
      const data = new Float32Array(page.width * page.height * 4);
      for (let cell = page.start * nx * ny + 1; cell <= page.end * nx * ny; cell++) {
        const at = cellOffset(layout, page, cell, nx, ny);
        data[at + 3] = phase.liquid[cell - 1] ? 1 : 0;
      }
      for (let k = 0; k < phase.mixed.length; k++) {
        const cell = phase.mixed[k], at = cellOffset(layout, page, cell, nx, ny);
        if (at < 0) continue;
        const plane = phase.planes[k];
        data[at] = plane[0]; data[at + 1] = plane[1]; data[at + 2] = plane[2];
        data[at + 3] = phase.liquid[cell - 1] ? 2 : -2;
      }
      phaseTexture = texture(gl, page.width, page.height, 4, data);
    }
    const solidTextures: WebGLTexture[] = [];
    for (const solid of topology.solids) {
      const data = new Float32Array(page.width * page.height * 4);
      for (let cell = page.start * nx * ny + 1; cell <= page.end * nx * ny; cell++) {
        if (solid.full[cell - 1]) data[cellOffset(layout, page, cell, nx, ny) + 3] = 1;
      }
      for (let k = 0; k < solid.cut.length; k++) {
        const at = cellOffset(layout, page, solid.cut[k], nx, ny);
        if (at < 0) continue;
        const plane = solid.planes[k];
        data[at] = plane[0]; data[at + 1] = plane[1]; data[at + 2] = plane[2]; data[at + 3] = 2;
      }
      solidTextures.push(texture(gl, page.width, page.height, 4, data));
    }
    pages.push({ vertices: buffer(gl, attributes), count: page.end - page.start, values, gas, phase: phaseTexture, solids: solidTextures });
  }
  const segments = phase?.segments ?? [];
  let segmentBuffer: WebGLBuffer | null = null;
  if (segments.length) {
    const vertices = new Float32Array(4 * segments.length);
    for (let i = 0; i < segments.length; i++) vertices.set([...segments[i][0], ...segments[i][1]], 4 * i);
    segmentBuffer = buffer(gl, vertices);
  }
  return { source: field, pages, segments: segmentBuffer, segmentCount: segments.length };
}
function deleteField(gl: WebGL2RenderingContext, field: FieldGPU): void {
  for (const page of field.pages) {
    gl.deleteBuffer(page.vertices); gl.deleteTexture(page.values);
    if (page.gas) gl.deleteTexture(page.gas);
    if (page.phase) gl.deleteTexture(page.phase);
    for (const solid of page.solids) gl.deleteTexture(solid);
  }
  if (field.segments) gl.deleteBuffer(field.segments);
}

function intersection(a: Rect, b: Rect): Rect | null {
  const result = { left: Math.max(a.left, b.left), top: Math.max(a.top, b.top), right: Math.min(a.right, b.right), bottom: Math.min(a.bottom, b.bottom) };
  return result.right > result.left && result.bottom > result.top ? result : null;
}
/** Split a visible rectangle around one higher-layer rectangle. */
export function subtractOcclusion(rect: Rect, obstruction: Rect): Rect[] {
  const overlap = intersection(rect, obstruction);
  if (!overlap) return [rect];
  const parts: Rect[] = [];
  if (rect.top < overlap.top) parts.push({ ...rect, bottom: overlap.top });
  if (overlap.bottom < rect.bottom) parts.push({ ...rect, top: overlap.bottom });
  if (rect.left < overlap.left) parts.push({ left: rect.left, right: overlap.left, top: overlap.top, bottom: overlap.bottom });
  if (overlap.right < rect.right) parts.push({ left: overlap.right, right: rect.right, top: overlap.top, bottom: overlap.bottom });
  return parts;
}

/** WebGL scissor pixels whose centers lie in a CSS rectangle, with y flipped. */
export function scissorPixels(region: Rect, canvasHeight: number, dprX: number, dprY: number): [number, number, number, number] {
  const x = Math.max(0, Math.ceil(region.left * dprX - 0.5));
  const y = Math.max(0, Math.ceil((canvasHeight - region.bottom) * dprY - 0.5));
  const right = Math.ceil(region.right * dprX - 0.5);
  const top = Math.ceil((canvasHeight - region.top) * dprY - 0.5);
  return [x, y, Math.max(0, right - x), Math.max(0, top - y)];
}

/** Intersect with viewport/overflow ancestors and subtract higher DOM layers. */
export function visibleRegions(element: HTMLElement, rect: DOMRect = element.getBoundingClientRect(),
  width = window.innerWidth, height = window.innerHeight): Rect[] {
  if (!element.isConnected || rect.width <= 0 || rect.height <= 0) return [];
  let region: Rect | null = intersection(rect, { left: 0, top: 0, right: width, bottom: height });
  for (let parent: Element | null = element; region && parent; parent = parent.parentElement) {
    const style = getComputedStyle(parent);
    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) return [];
    if (parent !== element) {
      const bounds = parent.getBoundingClientRect();
      if (['hidden', 'clip', 'scroll', 'auto'].includes(style.overflowX)) {
        region = intersection(region, { left: bounds.left, right: bounds.right, top: region.top, bottom: region.bottom });
      }
      if (region && ['hidden', 'clip', 'scroll', 'auto'].includes(style.overflowY)) {
        region = intersection(region, { left: region.left, right: region.right, top: bounds.top, bottom: bounds.bottom });
      }
    }
  }
  if (!region) return [];
  const obstructions: Rect[] = [];
  for (const candidate of document.querySelectorAll('*')) {
    if (candidate === element || candidate.contains(element) || element.contains(candidate)) continue;
    const bounds = intersection(region, candidate.getBoundingClientRect());
    if (!bounds) continue;
    const stack = document.elementsFromPoint((bounds.left + bounds.right) / 2, (bounds.top + bounds.bottom) / 2);
    const candidateIndex = stack.indexOf(candidate);
    const viewIndex = stack.findIndex(item => item === element || element.contains(item));
    if (candidateIndex >= 0 && viewIndex >= 0 && candidateIndex < viewIndex) obstructions.push(bounds);
  }
  obstructions.sort((a, b) => (b.right - b.left) * (b.bottom - b.top) - (a.right - a.left) * (a.bottom - a.top));
  let regions = [region];
  for (const obstruction of obstructions) {
    regions = regions.flatMap(part => subtractOcclusion(part, obstruction));
    if (!regions.length) break;
  }
  return regions;
}

function themeColor(element: HTMLElement, property: string, fallback: [number, number, number]): [number, number, number] {
  const value = getComputedStyle(element).getPropertyValue(property).trim();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(value);
  if (hex) {
    const text = hex[1].length === 3 ? [...hex[1]].map(char => char + char).join('') : hex[1];
    return [0, 2, 4].map(at => parseInt(text.slice(at, at + 2), 16) / 255) as [number, number, number];
  }
  const rgb = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(value);
  return rgb ? [Number(rgb[1]) / 255, Number(rgb[2]) / 255, Number(rgb[3]) / 255] : fallback;
}

/** One WebGL2 canvas for every field panel in a browser window. */
export class SharedRenderer {
  readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private readonly views = new Map<string, View>();
  private resources: FieldGPU[] = [];
  private lutTextures = new Map<string, WebGLTexture>();
  private blockProgram!: Program;
  private lineProgram!: Program;
  private blockVAO!: WebGLVertexArrayObject;
  private lineVAO!: WebGLVertexArrayObject;
  private emptyTexture!: WebGLTexture;
  private maxTexture = 4096;
  private frame = 0;
  private disposed = false;
  private lost = false;
  private pointerDown = false;
  private pixelRatioQuery: MediaQueryList | null = null;
  private readonly resizeObserver: ResizeObserver;
  private readonly mutationObserver: MutationObserver;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.setAttribute('aria-hidden', 'true');
    Object.assign(this.canvas.style, { position: 'fixed', left: '0', top: '0', width: '100%', height: '100%',
      zIndex: '2147483646', pointerEvents: 'none' });
    const gl = this.canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
    if (!gl) throw new Error('TwoPhaseAMR Studio requires WebGL2');
    this.gl = gl;
    this.initialize();
    this.canvas.addEventListener('webglcontextlost', this.onContextLost);
    this.canvas.addEventListener('webglcontextrestored', this.onContextRestored);
    document.body.append(this.canvas);
    this.resizeObserver = new ResizeObserver(this.requestRender);
    this.mutationObserver = new MutationObserver(records => {
      if (records.some(record => record.target !== this.canvas)) this.requestRender();
    });
    this.mutationObserver.observe(document.documentElement, { subtree: true, childList: true, attributes: true,
      attributeFilter: ['class', 'style', 'hidden', 'open', 'aria-hidden', 'data-theme'] });
    window.addEventListener('resize', this.requestRender);
    window.addEventListener('scroll', this.requestRender, true);
    window.addEventListener('toggle', this.requestRender, true);
    window.addEventListener('pointerdown', this.onPointerDown, true);
    window.addEventListener('pointermove', this.onPointerMove, true);
    window.addEventListener('pointerup', this.onPointerUp, true);
    window.addEventListener('pointercancel', this.onPointerUp, true);
    this.watchPixelRatio();
    this.requestRender();
  }

  private initialize(): void {
    const gl = this.gl;
    this.blockProgram = compile(gl, BLOCK_VERTEX, BLOCK_FRAGMENT);
    this.lineProgram = compile(gl, LINE_VERTEX, LINE_FRAGMENT);
    const blockVAO = gl.createVertexArray(), lineVAO = gl.createVertexArray();
    if (!blockVAO || !lineVAO) throw new Error('Unable to allocate Studio vertex arrays');
    this.blockVAO = blockVAO; this.lineVAO = lineVAO;
    this.emptyTexture = texture(gl, 1, 1, 4, new Float32Array(4));
    this.maxTexture = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE) as number);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
  }

  registerView(id: string, element: HTMLElement): void {
    if (this.disposed) throw new Error('Renderer is disposed');
    if (this.views.has(id)) throw new Error(`Renderer view already registered: ${id}`);
    this.views.set(id, { element, options: { field: null, camera: { center: [0, 0], scale: 1 }, mode: 'cell',
      range: [0, 1], colormap: 'viridis', overlays: {}, theme: 'light' } });
    this.resizeObserver.observe(element);
    this.requestRender();
  }

  updateView(id: string, options: Partial<ViewOptions>): void {
    const view = this.views.get(id);
    if (!view) throw new Error(`Unknown renderer view: ${id}`);
    view.options = { ...view.options, ...options };
    this.requestRender();
  }

  removeView(id: string): void {
    const view = this.views.get(id);
    if (!view) return;
    this.resizeObserver.unobserve(view.element);
    this.views.delete(id);
    this.releaseUnused();
    this.requestRender();
  }

  /** CSS-local x/y are converted through this panel's current isotropic camera. */
  pick(id: string, x: number, y: number): PickResult | null {
    const view = this.views.get(id);
    if (!view?.options.field) return null;
    const rect = view.element.getBoundingClientRect();
    if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return null;
    return pickField(view.options.field, screenToWorld(view.options.camera, [x, y], rect.width, rect.height), view.options.mode);
  }

  readonly requestRender = (): void => {
    if (!this.disposed && !this.lost && !this.frame) this.frame = requestAnimationFrame(this.draw);
  };

  /** Draw immediately; useful for controlled captures and external benchmarks. */
  renderNow(): void {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.draw();
  }
  private watchPixelRatio(): void {
    this.pixelRatioQuery?.removeEventListener('change', this.onPixelRatioChange);
    this.pixelRatioQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    this.pixelRatioQuery.addEventListener('change', this.onPixelRatioChange);
  }
  private readonly onPixelRatioChange = (): void => { this.watchPixelRatio(); this.requestRender(); };
  private readonly onPointerDown = (): void => { this.pointerDown = true; this.requestRender(); };
  private readonly onPointerMove = (): void => { if (this.pointerDown) this.requestRender(); };
  private readonly onPointerUp = (): void => { this.pointerDown = false; this.requestRender(); };
  private readonly onContextLost = (event: Event): void => {
    event.preventDefault(); this.lost = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0; this.resources = []; this.lutTextures.clear();
  };
  private readonly onContextRestored = (): void => { this.lost = false; this.initialize(); this.requestRender(); };

  private releaseUnused(): void {
    const active = [...this.views.values()].map(view => view.options.field).filter((field): field is FieldData => !!field);
    this.resources = this.resources.filter(resource => {
      if (active.some(field => sameField(field, resource.source))) return true;
      if (!this.lost) deleteField(this.gl, resource);
      return false;
    });
  }

  private colorTexture(name: string): WebGLTexture {
    let result = this.lutTextures.get(name);
    if (result) return result;
    const gl = this.gl;
    result = gl.createTexture() ?? undefined;
    if (!result) throw new Error('Unable to allocate a colormap texture');
    gl.bindTexture(gl.TEXTURE_2D, result);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 256, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.UNSIGNED_BYTE, colormapLUT(name));
    this.lutTextures.set(name, result);
    return result;
  }

  private bindTexture(unit: number, value: WebGLTexture): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, value);
  }
  private setCamera(program: Program, camera: Camera, width: number, height: number): void {
    const gl = this.gl;
    gl.uniform2f(uniform(gl, program, 'u_center'), camera.center[0], camera.center[1]);
    gl.uniform2f(uniform(gl, program, 'u_viewSize'), width, height);
    gl.uniform1f(uniform(gl, program, 'u_scale'), camera.scale);
  }
  private drawView(view: View, rect: DOMRect, regions: Rect[], dprX: number, dprY: number, height: number): void {
    const gl = this.gl, options = view.options, source = options.field;
    if (!source || !regions.length || !(options.camera.scale > 0 && Number.isFinite(options.camera.scale))) return;
    let field = this.resources.find(resource => sameField(source, resource.source));
    if (!field) { field = buildField(gl, source, this.maxTexture); this.resources.push(field); }
    const light = options.theme === 'light';
    const interfaceColor = themeColor(view.element, '--overlay-interface', light ? [0.094, 0.137, 0.2] : [0.94, 0.96, 0.97]);
    const blockColor = themeColor(view.element, '--overlay-blocks', light ? [0.27, 0.35, 0.44] : [0.75, 0.81, 0.87]);
    const solidColor = themeColor(view.element, '--overlay-solid', light ? [0.38, 0.46, 0.55] : [0.61, 0.68, 0.76]);
    const [nx, ny] = source.topology.block_size;
    gl.viewport(Math.round(rect.left * dprX), Math.round((height - rect.bottom) * dprY), Math.round(rect.width * dprX), Math.round(rect.height * dprY));
    gl.useProgram(this.blockProgram.handle);
    gl.bindVertexArray(this.blockVAO);
    this.setCamera(this.blockProgram, options.camera, rect.width, rect.height);
    gl.uniform2i(uniform(gl, this.blockProgram, 'u_blockSize'), nx, ny);
    gl.uniform1i(uniform(gl, this.blockProgram, 'u_placement'), source.placement === 'cell' ? 0 : source.placement === 'face_x' ? 1 : 2);
    gl.uniform1i(uniform(gl, this.blockProgram, 'u_twoPhase'), Number(!!source.liquid && !!source.gas && !!source.phase));
    gl.uniform1i(uniform(gl, this.blockProgram, 'u_sharp'), Number(options.mode === 'plic'));
    gl.uniform1i(uniform(gl, this.blockProgram, 'u_cells'), Number(!!options.overlays.cells));
    gl.uniform1i(uniform(gl, this.blockProgram, 'u_blocks'), Number(!!options.overlays.blocks));
    gl.uniform1i(uniform(gl, this.blockProgram, 'u_levels'), Number(!!options.overlays.levels));
    gl.uniform2f(uniform(gl, this.blockProgram, 'u_range'), options.range[0], options.range[1]);
    gl.uniform3f(uniform(gl, this.blockProgram, 'u_interfaceColor'), ...interfaceColor);
    gl.uniform3f(uniform(gl, this.blockProgram, 'u_blockColor'), ...blockColor);
    gl.uniform3f(uniform(gl, this.blockProgram, 'u_solidColor'), ...solidColor);
    for (const [unit, name] of ['u_values', 'u_gas', 'u_phase', 'u_solid', 'u_lut'].entries()) gl.uniform1i(uniform(gl, this.blockProgram, name), unit);
    this.bindTexture(4, this.colorTexture(options.colormap));
    for (const region of regions) {
      const scissor = scissorPixels(region, height, dprX, dprY);
      if (!scissor[2] || !scissor[3]) continue;
      gl.scissor(...scissor);
      for (const page of field.pages) {
        this.bindTexture(0, page.values); this.bindTexture(1, page.gas ?? page.values);
        this.bindTexture(2, page.phase ?? this.emptyTexture);
        const associated = page.solids[associatedSolidIndex(source) - 1];
        this.bindTexture(3, associated ?? this.emptyTexture);
        gl.uniform1i(uniform(gl, this.blockProgram, 'u_hasSolid'), Number(!!associated));
        gl.uniform1i(uniform(gl, this.blockProgram, 'u_solidPass'), 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, page.vertices);
        for (const [location, size, offset] of [[0, 2, 0], [1, 2, 8], [2, 2, 16], [3, 1, 24]]) {
          gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, size, gl.FLOAT, false, 28, offset); gl.vertexAttribDivisor(location, 1);
        }
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, page.count);
        if (options.overlays.solid) {
          gl.uniform1i(uniform(gl, this.blockProgram, 'u_solidPass'), 1);
          gl.uniform1i(uniform(gl, this.blockProgram, 'u_hasSolid'), 1);
          for (const solid of page.solids) {
            this.bindTexture(3, solid);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, page.count);
          }
        }
      }
    }
    if (options.overlays.interface && field.segments && field.segmentCount) {
      gl.useProgram(this.lineProgram.handle); gl.bindVertexArray(this.lineVAO);
      this.setCamera(this.lineProgram, options.camera, rect.width, rect.height);
      gl.uniform1f(uniform(gl, this.lineProgram, 'u_width'), 1.5);
      gl.uniform3f(uniform(gl, this.lineProgram, 'u_color'), ...interfaceColor);
      gl.bindBuffer(gl.ARRAY_BUFFER, field.segments);
      for (const [location, offset] of [[0, 0], [1, 8]]) {
        gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 16, offset); gl.vertexAttribDivisor(location, 1);
      }
      for (const region of regions) {
        const scissor = scissorPixels(region, height, dprX, dprY);
        if (!scissor[2] || !scissor[3]) continue;
        gl.scissor(...scissor);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, field.segmentCount);
      }
    }
  }

  private readonly draw = (): void => {
    this.frame = 0;
    if (this.disposed || this.lost) return;
    const gl = this.gl, width = window.innerWidth, height = window.innerHeight;
    if (!(width > 0 && height > 0)) return;
    const ratio = window.devicePixelRatio || 1;
    const pixelsX = Math.max(1, Math.round(width * ratio)), pixelsY = Math.max(1, Math.round(height * ratio));
    if (this.canvas.width !== pixelsX) this.canvas.width = pixelsX;
    if (this.canvas.height !== pixelsY) this.canvas.height = pixelsY;
    gl.disable(gl.SCISSOR_TEST); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.enable(gl.SCISSOR_TEST);
    this.releaseUnused();
    for (const view of this.views.values()) {
      const rect = view.element.getBoundingClientRect();
      const regions = visibleRegions(view.element, rect, width, height);
      this.drawView(view, rect, regions, pixelsX / width, pixelsY / height, height);
    }
    gl.bindVertexArray(null); gl.disable(gl.SCISSOR_TEST);
  };

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect(); this.mutationObserver.disconnect();
    this.pixelRatioQuery?.removeEventListener('change', this.onPixelRatioChange);
    window.removeEventListener('resize', this.requestRender);
    window.removeEventListener('scroll', this.requestRender, true);
    window.removeEventListener('toggle', this.requestRender, true);
    window.removeEventListener('pointerdown', this.onPointerDown, true);
    window.removeEventListener('pointermove', this.onPointerMove, true);
    window.removeEventListener('pointerup', this.onPointerUp, true);
    window.removeEventListener('pointercancel', this.onPointerUp, true);
    this.canvas.removeEventListener('webglcontextlost', this.onContextLost);
    this.canvas.removeEventListener('webglcontextrestored', this.onContextRestored);
    if (!this.lost) {
      for (const resource of this.resources) deleteField(this.gl, resource);
      for (const lut of this.lutTextures.values()) this.gl.deleteTexture(lut);
      this.gl.deleteTexture(this.emptyTexture);
      this.gl.deleteVertexArray(this.blockVAO); this.gl.deleteVertexArray(this.lineVAO);
      this.gl.deleteProgram(this.blockProgram.handle); this.gl.deleteProgram(this.lineProgram.handle);
    }
    this.resources = []; this.lutTextures.clear(); this.views.clear(); this.canvas.remove();
  }
}
