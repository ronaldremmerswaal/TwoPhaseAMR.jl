import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  atlasBlockOrigin, atlasLayout, boxFitCamera, colormapGradient, colormapLUT, finestCellCamera,
  fitCamera, interfaceBandBounds, mapColor, panCamera, pickField, scatterValues, scissorPixels, screenToWorld, subtractOcclusion,
  topologyBounds, wheelZoomCameraAt, worldToScreen, zoomCameraAt,
  type FieldData, type PhaseData, type TopologyData,
} from '../src/renderer';

function topology(): TopologyData {
  return {
    block_size: [2, 2], domain: { origin: [0, 0], extent: [3, 2] },
    blocks: [
      { level: 0, i: 0, j: 0, origin: [0, 0], spacing: [1, 1] },
      { level: 1, i: 2, j: 0, origin: [2, 0], spacing: [0.5, 0.5] },
      { level: 1, i: 2, j: 1, origin: [2, 1], spacing: [0.5, 0.5] },
    ],
    face_tiles: [Array(36).fill(0), Array(36).fill(0)], solids: [],
  };
}

function closePoint(actual: number[], expected: number[]): void {
  expect(actual[0]).toBeCloseTo(expected[0], 12);
  expect(actual[1]).toBeCloseTo(expected[1], 12);
}

describe('isotropic camera', () => {
  it('fits the whole domain and uses the extra width for a square box', () => {
    const camera = fitCamera([0, 2, 0, 1], 800, 400);
    expect(camera).toEqual({ center: [1, 0.5], scale: 0.0025 });
    expect(topologyBounds(topology())).toEqual([0, 3, 0, 2]);
    const square = boxFitCamera([0, 1, 0, 1], 800, 400);
    expect(square.center).toEqual([0.5, 0.5]);
    expect(square.scale).toBe(1 / 400);
    closePoint(screenToWorld(square, [0, 0], 800, 400), [-0.5, 1]);
    closePoint(screenToWorld(square, [800, 400], 800, 400), [1.5, 0]);
    expect(() => fitCamera([0, 1, 0, 1], 0, 400)).toThrow();
  });

  it('transforms points reversibly, anchors zoom, and pans with the pointer', () => {
    const camera = { center: [2, -1] as [number, number], scale: 0.125 };
    const world: [number, number] = [8, 3];
    closePoint(screenToWorld(camera, worldToScreen(camera, world, 700, 300), 700, 300), world);
    const anchor: [number, number] = [111, 222];
    const before = screenToWorld(camera, anchor, 700, 300);
    const zoomed = zoomCameraAt(camera, 0.4, anchor, 700, 300);
    expect(zoomed.scale).toBeCloseTo(0.05);
    closePoint(screenToWorld(zoomed, anchor, 700, 300), before);
    const panned = panCamera(camera, 20, -16);
    const originalScreen = worldToScreen(camera, world, 700, 300);
    closePoint(worldToScreen(panned, world, 700, 300), [originalScreen[0] + 20, originalScreen[1] - 16]);
    expect(finestCellCamera(camera, topology())).toEqual({ center: [2, -1], scale: 0.5 / 8 });
  });

  it('zooms in and out with wheel direction and keeps the cursor anchored', () => {
    const camera = fitCamera([0, 1, 0, 3], 600, 900);
    const anchor: [number, number] = [174, 627];
    const world = screenToWorld(camera, anchor, 600, 900);
    const inward = wheelZoomCameraAt(camera, -120, 0, anchor, 600, 900);
    const outward = wheelZoomCameraAt(camera, 120, 0, anchor, 600, 900);
    expect(inward.scale).toBeLessThan(camera.scale);
    expect(outward.scale).toBeGreaterThan(camera.scale);
    closePoint(screenToWorld(inward, anchor, 600, 900), world);
    closePoint(screenToWorld(outward, anchor, 600, 900), world);
    closePoint(wheelZoomCameraAt(inward, 120, 0, anchor, 600, 900).center, camera.center);
    expect(wheelZoomCameraAt(camera, -1, 1, anchor, 600, 900).scale)
      .toBeCloseTo(wheelZoomCameraAt(camera, -16, 0, anchor, 600, 900).scale);
    expect(wheelZoomCameraAt(camera, 1, 2, anchor, 600, 900).scale)
      .toBeCloseTo(wheelZoomCameraAt(camera, 900, 0, anchor, 600, 900).scale);
  });

  it('refits the full domain to the current viewport after zoom and resize', () => {
    const bounds: [number, number, number, number] = [0, 1, 0, 3];
    const initial = fitCamera(bounds, 600, 900);
    const zoomed = wheelZoomCameraAt(initial, -200, 0, [100, 400], 600, 900);
    expect(zoomed.scale).toBeLessThan(initial.scale);
    expect(fitCamera(bounds, 600, 900)).toEqual(initial);
    const resized = fitCamera(bounds, 300, 400);
    expect(resized.center).toEqual(initial.center);
    expect(resized.scale).toBe(3 / 400);
    closePoint(worldToScreen(resized, [0.5, 3], 300, 400), [150, 0]);
    closePoint(worldToScreen(resized, [0.5, 0], 300, 400), [150, 400]);
  });

  it('fits a padded interface band including flat, vertical and absent interfaces', () => {
    const domain: [number, number, number, number] = [0, 1, 0, 3];
    const flat = interfaceBandBounds([[[0, 1], [1, 1]]], domain)!;
    expect(flat[0]).toBeLessThan(0); expect(flat[1]).toBeGreaterThan(1);
    expect(flat[2]).toBeLessThan(1); expect(flat[3]).toBeGreaterThan(1);
    expect(flat[3] - flat[2]).toBeGreaterThanOrEqual(0.25);
    const vertical = interfaceBandBounds([[[0.5, 0.5], [0.5, 2.5]]], domain)!;
    expect(vertical[1] - vertical[0]).toBeGreaterThanOrEqual(0.5);
    expect(interfaceBandBounds([], domain)).toBeNull();
    expect(interfaceBandBounds([[[NaN, NaN], [Infinity, Infinity]]], domain)).toBeNull();
  });

  it('matches the Julia-generated camera parity vectors', () => {
    type Vector = {
      rect: [number, number, number, number]; size: [number, number];
      fit: { center: [number, number]; scale: number };
      visible: [number, number, number, number];
      zoom: { factor: number; anchor: [number, number]; center: [number, number]; scale: number };
    };
    const vectors = JSON.parse(readFileSync(new URL('./camera-vectors.json', import.meta.url), 'utf8')) as Vector[];
    expect(vectors.length).toBeGreaterThan(0);
    for (const vector of vectors) {
      const [width, height] = vector.size;
      const camera = fitCamera(vector.rect, width, height);
      const boxed = boxFitCamera(vector.rect, width, height);
      closePoint(camera.center, vector.fit.center);
      expect(camera.scale).toBeCloseTo(vector.fit.scale, 12);
      closePoint(boxed.center, vector.fit.center);
      expect(boxed.scale).toBeCloseTo(vector.fit.scale, 12);
      const lower = screenToWorld(camera, [0, height], width, height);
      const upper = screenToWorld(camera, [width, 0], width, height);
      [lower[0], upper[0], lower[1], upper[1]].forEach((value, i) => expect(value).toBeCloseTo(vector.visible[i], 12));
      const zoom = zoomCameraAt(camera, vector.zoom.factor, vector.zoom.anchor, width, height);
      closePoint(zoom.center, vector.zoom.center);
      expect(zoom.scale).toBeCloseTo(vector.zoom.scale, 12);
    }
  });
});

describe('block atlas layout', () => {
  it('places distinct block tiles across texture pages', () => {
    const mesh = topology();
    mesh.blocks.push({ ...mesh.blocks[1], i: 3 }, { ...mesh.blocks[2], i: 3 });
    const layout = atlasLayout(mesh, 'cell', 4);
    expect([layout.tileWidth, layout.tileHeight, layout.columns]).toEqual([2, 2, 2]);
    expect(layout.pages).toEqual([{ start: 0, end: 4, width: 4, height: 4 }, { start: 4, end: 5, width: 2, height: 2 }]);
    expect([0, 1, 2, 3].map(i => atlasBlockOrigin(layout, layout.pages[0], i))).toEqual([[0, 0], [2, 0], [0, 2], [2, 2]]);
    expect(atlasBlockOrigin(layout, layout.pages[1], 4)).toEqual([0, 0]);
    expect(() => atlasBlockOrigin(layout, layout.pages[1], 3)).toThrow();
    const xFace = atlasLayout(mesh, 'face_x', 8);
    expect([xFace.tileWidth, xFace.tileHeight]).toEqual([3, 4]);
    const yFace = atlasLayout(mesh, 'face_y', 8);
    expect([yFace.tileWidth, yFace.tileHeight]).toEqual([4, 3]);
    expect(() => atlasLayout(mesh, 'face_x', 3)).toThrow();
  });

  it('scatters block-major cells and one-based face vectors without changing tile placement', () => {
    const mesh = topology();
    const values = Float32Array.from({ length: 12 }, (_, i) => i + 1);
    const field: FieldData = { topology: mesh, placement: 'cell', values };
    const cells = atlasLayout(mesh, 'cell', 4);
    expect([...scatterValues(field, values, cells, cells.pages[0])]).toEqual([
      1, 2, 5, 6, 3, 4, 7, 8, 9, 10, NaN, NaN, 11, 12, NaN, NaN,
    ]);
    const faces: FieldData = { topology: mesh, placement: 'face_x', values };
    mesh.face_tiles[0][2] = 11;
    mesh.face_tiles[0][5] = 12;
    mesh.face_tiles[0][12] = 11;
    mesh.face_tiles[0][15] = 12;
    const tiles = atlasLayout(mesh, 'face_x', 8);
    const page = tiles.pages[0];
    const atlas = scatterValues(faces, values, tiles, page);
    expect(atlas[2]).toBe(11);
    expect(atlas[page.width + 2]).toBe(12);
    expect(atlas[3]).toBe(11);
    expect(atlas[page.width + 3]).toBe(12);
    expect(atlas[0]).toBeNaN(); // zero tile index is missing, never face 1
  });
});

describe('scissored panel geometry', () => {
  it('subtracts floating occluders into disjoint visible rectangles', () => {
    const panel = { left: 0, top: 0, right: 100, bottom: 100 };
    expect(subtractOcclusion(panel, { left: 25, top: 20, right: 75, bottom: 60 })).toEqual([
      { left: 0, top: 0, right: 100, bottom: 20 },
      { left: 0, top: 60, right: 100, bottom: 100 },
      { left: 0, top: 20, right: 25, bottom: 60 },
      { left: 75, top: 20, right: 100, bottom: 60 },
    ]);
    expect(subtractOcclusion(panel, { left: -1, top: -1, right: 101, bottom: 101 })).toEqual([]);
    expect(subtractOcclusion(panel, { left: 100, top: 0, right: 110, bottom: 100 })).toEqual([panel]);
  });

  it('maps fractional CSS rectangles to covered high-DPI pixels and flips y', () => {
    expect(scissorPixels({ left: 10.25, top: 20.25, right: 30.25, bottom: 40.25 }, 100, 2, 2)).toEqual([20, 119, 40, 40]);
    expect(scissorPixels({ left: 0, top: 0, right: 800, bottom: 400 }, 400, 1.5, 1.5)).toEqual([0, 0, 1200, 600]);
  });
});

describe('fixed colormap LUTs', () => {
  it('matches lookup endpoints, clamps and makes nonfinite values transparent', () => {
    expect(mapColor(-1, [0, 1], 'viridis')).toEqual([68 / 255, 1 / 255, 84 / 255, 1]);
    expect(mapColor(2, [0, 1], 'viridis')).toEqual([253 / 255, 231 / 255, 37 / 255, 1]);
    expect(mapColor(NaN, [0, 1])).toEqual([0, 0, 0, 0]);
    expect(mapColor(Infinity, [0, 1])).toEqual([0, 0, 0, 0]);
    expect(mapColor(8, [8, 8], 'gray')[0]).toBeCloseTo(0.5, 2);
    expect(colormapLUT('unknown')).toBe(colormapLUT('viridis'));
  });

  it('constructs the colourbar from the same 256 quantized samples as the shader', () => {
    const lut = colormapLUT('balance');
    const gradient = colormapGradient('balance');
    expect(lut).toHaveLength(1024);
    expect(gradient).toContain(`rgb(${lut[0]}, ${lut[1]}, ${lut[2]}) 0%`);
    expect(gradient).toContain(`rgb(${lut[1020]}, ${lut[1021]}, ${lut[1022]}) 100%`);
    expect((gradient.match(/rgb\(/g) ?? [])).toHaveLength(256);
  });
});

describe('analytic picking', () => {
  it('locates block-major cells and keeps Julia indices distinct from block keys', () => {
    const mesh = topology();
    const field: FieldData = { topology: mesh, placement: 'cell', values: Float32Array.from({ length: 12 }, (_, i) => i + 1) };
    const hit = pickField(field, [2.6, 1.1]);
    expect(hit).toMatchObject({ world: [2.6, 1.1], block: 3, key: { level: 1, i: 2, j: 1 }, cell: [2, 1], spacing: [0.5, 0.5], index: 10, value: 10, phase: null, solid: false });
    expect(pickField(field, [3, 1])).toBeNull();
    expect(pickField(field, [-0.01, 1])).toBeNull();
  });

  it('uses x-face dual tiles on both sides of a coarse–fine boundary', () => {
    const mesh = topology();
    const x = mesh.face_tiles[0], columns = 3, rows = 4;
    // Coarse boundary column: each half-row addresses a different fine face.
    [11, 12, 13, 14].forEach((value, row) => { x[row * columns + 2] = value; });
    [11, 11, 12, 12].forEach((value, row) => { x[columns * rows + row * columns] = value; });
    [13, 13, 14, 14].forEach((value, row) => { x[2 * columns * rows + row * columns] = value; });
    const field: FieldData = { topology: mesh, placement: 'face_x', values: Float32Array.from({ length: 14 }, (_, i) => i + 1) };
    expect(pickField(field, [1.8, 0.25])).toMatchObject({ block: 1, index: 11, value: 11 });
    expect(pickField(field, [2.1, 0.25])).toMatchObject({ block: 2, index: 11, value: 11 });
    expect(pickField(field, [1.8, 0.75])).toMatchObject({ block: 1, index: 12, value: 12 });
    expect(pickField(field, [2.1, 0.75])).toMatchObject({ block: 2, index: 12, value: 12 });
    expect(pickField(field, [1.8, 1.25])).toMatchObject({ block: 1, index: 13, value: 13 });
    expect(pickField(field, [2.1, 1.25])).toMatchObject({ block: 3, index: 13, value: 13 });
    expect(pickField(field, [0.1, 0.1])?.index).toBe(0);
    expect(pickField(field, [0.1, 0.1])?.value).toBeNaN();
  });

  it('uses y-face half-columns and the containing cell phase', () => {
    const mesh = topology();
    [1, 2, 3, 4].forEach((value, col) => { mesh.face_tiles[1][4 + col] = value; });
    const phase: PhaseData = { liquid: [true, true, true, true], mixed: [1], planes: [[1, 0, 0]], segments: [], solid: 0 };
    const field: FieldData = { topology: mesh, placement: 'face_y', liquid: new Float32Array([11, 12, 13, 14]),
      gas: new Float32Array([21, 22, 23, 24]), phase };
    expect(pickField(field, [0.25, 0.75], 'plic')).toMatchObject({ index: 1, value: 11, phase: 'liquid', cell: [1, 1] });
    expect(pickField(field, [0.75, 0.75], 'plic')).toMatchObject({ index: 2, value: 22, phase: 'gas', cell: [1, 1] });
    expect(pickField(field, [0.75, 0.75], 'cell')).toMatchObject({ index: 2, value: 12, phase: 'liquid' });
  });

  it('matches PLIC, majority, solid planes and missing data rules', () => {
    const mesh = topology();
    mesh.solids = [{ full: [false, false, true, false], cut: [1], planes: [[0, 1, 0]] }];
    const phase: PhaseData = { liquid: [true, false, true, false], mixed: [1, 2],
      planes: [[1, 0, 0], [1, 0, 0]], segments: [], solid: 1 };
    const field: FieldData = { topology: mesh, placement: 'cell', liquid: new Float32Array([10, 20, 30, 40]),
      gas: new Float32Array([90, 80, 70, NaN]), phase };
    expect(pickField(field, [0.25, 0.75], 'plic')).toMatchObject({ phase: 'liquid', value: 10, solid: false });
    expect(pickField(field, [0.5, 0.75], 'plic')).toMatchObject({ phase: 'liquid', value: 10, solid: false });
    expect(pickField(field, [0.75, 0.75], 'plic')).toMatchObject({ phase: 'gas', value: 90, solid: false });
    expect(pickField(field, [0.75, 0.75], 'cell')).toMatchObject({ phase: 'liquid', value: 10 });
    expect(pickField(field, [1.25, 0.75], 'cell')).toMatchObject({ phase: 'gas', value: 80 });
    expect(pickField(field, [1.25, 0.75], 'plic')).toMatchObject({ phase: 'liquid', value: 20 });
    expect(pickField(field, [0.25, 0.25], 'plic')).toMatchObject({ phase: null, value: NaN, solid: true });
    expect(pickField(field, [0.25, 0.5], 'plic')).toMatchObject({ phase: null, value: NaN, solid: true });
    expect(pickField(field, [0.25, 0.25], 'cell')).toMatchObject({ phase: 'liquid', value: 10, solid: true });
    expect(pickField(field, [0.25, 1.25], 'cell')).toMatchObject({ phase: null, value: NaN, solid: true });
    expect(pickField(field, [1.25, 1.25], 'cell')?.value).toBeNaN();
    const single: FieldData = { topology: mesh, placement: 'cell', values: new Float32Array([7, 8, 9, 10]), phase };
    expect(pickField(single, [0.25, 0.25], 'plic')).toMatchObject({ value: 7, phase: null, solid: true });
    expect(pickField(single, [0.25, 1.25], 'cell')).toMatchObject({ value: 9, phase: null, solid: true });
  });

  it('reports standalone and multiple solids without changing the associated phase mask', () => {
    const mesh = topology();
    mesh.solids = [
      { full: [true, false, false, false], cut: [], planes: [] },
      { full: [false, true, false, false], cut: [], planes: [] },
    ];
    const phase: PhaseData = { liquid: [true, true, true, true], mixed: [], planes: [], segments: [], solid: 1 };
    const paired: FieldData = { topology: mesh, placement: 'cell', phase,
      liquid: new Float32Array([10, 20, 30, 40]), gas: new Float32Array([90, 80, 70, 60]) };
    expect(pickField(paired, [0.25, 0.25], 'plic')).toMatchObject({ value: NaN, phase: null, solid: true });
    expect(pickField(paired, [1.25, 0.25], 'plic')).toMatchObject({ value: 20, phase: 'liquid', solid: true });
    expect(pickField({ ...paired, solid: 2 }, [1.25, 0.25], 'plic')).toMatchObject({ value: NaN, phase: null, solid: true });
    const single: FieldData = { topology: mesh, placement: 'cell', values: new Float32Array([1, 2, 3, 4]) };
    expect(pickField(single, [1.25, 0.25], 'plic')).toMatchObject({ value: 2, phase: null, solid: true });
  });
});
