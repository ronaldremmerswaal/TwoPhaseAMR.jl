import { afterEach, describe, expect, it, vi } from 'vitest';
import { combineRanges, createFieldPanel, defaultFieldColor, resolvePanelFrame, robustValuesRange, sameFieldDataKey, statisticsRange, valuesRange } from '../src/field-panel';
import { StudioState } from '../src/state';
import type { SharedRenderer } from '../src/renderer';
import type { StudioClient } from '../src/client';
import type { FrameSummary } from '../src/types';

const frame = (stats?: unknown): FrameSummary => ({ id: 4, time: 0.5, step: 12, kind: 'keyframe', topology: 2, stats });

describe('field colour-range data', () => {
  it('defaults automatic fields to robust ranges and respects signed and fixed metadata', () => {
    expect(defaultFieldColor({ name: 'velocity_magnitude', colormap: 'viridis' }))
      .toEqual({ fixedRange: null, colorPolicy: 'robust', symmetric: false });
    expect(defaultFieldColor({ name: 'velocity_vorticity', symmetric: true, colormap: 'viridis' }))
      .toEqual({ fixedRange: null, colorPolicy: 'robust', symmetric: true });
    expect(defaultFieldColor({ name: 'pressure', colormap: 'balance' }))
      .toEqual({ fixedRange: null, colorPolicy: 'robust', symmetric: true });
    expect(defaultFieldColor({ name: 'volume_fraction', colorrange: [0, 1], symmetric: true }))
      .toEqual({ fixedRange: [0, 1], colorPolicy: 'fixed', symmetric: false });
    expect(defaultFieldColor({ name: 'bad_range', colorrange: [4, -4] }))
      .toEqual({ fixedRange: null, colorPolicy: 'robust', symmetric: false });
  });

  it('distinguishes missing, empty, and finite server statistics', () => {
    expect(statisticsRange(frame(), 'velocity')).toBeUndefined();
    expect(statisticsRange(frame({ velocity: { finite: 0, minimum: null, maximum: null } }), 'velocity')).toBeNull();
    expect(statisticsRange(frame({ velocity: { finite: 3, minimum: -2.5, maximum: 4 } }), 'velocity')).toEqual([-2.5, 4]);
    expect(statisticsRange(frame({ velocity: { finite: 3, minimum: 4, maximum: -2 } }), 'velocity')).toBeUndefined();
    expect(statisticsRange(frame({ velocity: { finite: 3, minimum: null, maximum: 4 } }), 'velocity')).toBeUndefined();
  });

  it('falls back to both intrinsic phase arrays and ignores non-finite entries', () => {
    expect(valuesRange(new Float32Array([NaN, -3, 1]), new Float32Array([Infinity, 5, -Infinity]))).toEqual([-3, 5]);
    expect(valuesRange(new Float32Array([NaN, Infinity]))).toBeNull();
    expect(valuesRange(new Float32Array([7, 7]))).toEqual([7, 7]);
  });

  it('clips the 99.5th percentile per frame using both phase arrays', () => {
    const liquid = Float32Array.from(Array.from({ length: 100 }, (_, i) => i));
    const gas = Float32Array.from([...Array.from({ length: 99 }, (_, i) => 100 + i), 10000, NaN, Infinity]);
    expect(robustValuesRange(false, liquid, gas)).toEqual([0, 198]);
    expect(robustValuesRange(true,
      Float32Array.from([...Array(100).fill(-2), NaN]),
      Float32Array.from([...Array(99).fill(3), -1000, Infinity]))).toEqual([-3, 3]);
    expect(robustValuesRange(false, new Float32Array([-5, -2, -3, NaN]))).toEqual([-5, -2]);
    expect(robustValuesRange(false, new Float32Array([NaN, Infinity]))).toBeNull();
    expect(robustValuesRange(true, new Float32Array([0, 0]))).toEqual([-0, 0]);
  });

  it('selects the exact finite percentile without modifying the source values', () => {
    const first = Float32Array.from(Array.from({ length: 1201 }, (_, i) => Math.sin(i * 1.7) * (i % 47)));
    const second = Float32Array.from([...Array.from({ length: 800 }, (_, i) => Math.cos(i * 0.9) * (i % 39)), NaN, Infinity]);
    const originalFirst = first.slice(), originalSecond = second.slice();
    const values = [...first, ...second].filter(Number.isFinite).sort((a, b) => a - b);
    const rank = Math.ceil(values.length * 0.995) - 1;
    expect(robustValuesRange(false, first, second)).toEqual([values[0], values[rank]]);
    const magnitudes = values.map(Math.abs).sort((a, b) => a - b);
    expect(robustValuesRange(true, first, second)).toEqual([-magnitudes[rank], magnitudes[rank]]);
    expect(first).toEqual(originalFirst);
    expect(second).toEqual(originalSecond);
  });

  it('merges stored-frame ranges and can apply a symmetric signed range', () => {
    expect(combineRanges([null, [-3, 1], [-2, 5], undefined])).toEqual([-3, 5]);
    expect(combineRanges([[-3, 1], [-2, 5]], true)).toEqual([-5, 5]);
    expect(combineRanges([null, undefined])).toBeNull();
    expect(combineRanges([[0, 0]], true)).toEqual([-0, 0]);
  });
});

describe('field frame pins', () => {
  it('follows the playhead only when unpinned and retains thinned preview frames', () => {
    const pinned = { ...frame(), id: 1, time: 0.1, kind: 'preview' };
    const playhead = { ...frame(), id: 2, time: 0.2 };
    expect(resolvePanelFrame(null, pinned, [pinned, playhead], playhead)).toBe(playhead);
    expect(resolvePanelFrame(1, pinned, [pinned, playhead], playhead)).toBe(pinned);
    expect(resolvePanelFrame(1, pinned, [playhead], playhead)).toBe(pinned);
    expect(resolvePanelFrame(1, undefined, [playhead], playhead)).toBeUndefined();
    expect(resolvePanelFrame(1, playhead, [playhead], playhead)).toBeUndefined();
    const promoted = { ...pinned, kind: 'keyframe' };
    expect(resolvePanelFrame(1, pinned, [promoted, playhead], playhead)).toBe(promoted);
  });

  it('retains cached data across epochs only for the same frame, topology and field', () => {
    const original = JSON.stringify([1, 4, 2, 'velocity', 1, 'cell']);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 2, 'velocity', 1, 'cell']))).toBe(true);
    expect(sameFieldDataKey(original, JSON.stringify([2, 5, 2, 'velocity', 1, 'cell']))).toBe(false);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 3, 'velocity', 1, 'cell']))).toBe(false);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 2, 'pressure', 1, 'cell']))).toBe(false);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 2, 'velocity', 0, 'cell']))).toBe(false);
    expect(sameFieldDataKey(original, JSON.stringify([2, 4, 2, 'velocity', 1, 'face_x']))).toBe(false);
  });
});

describe('field camera links', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('shares cameras only on the same known domain, supports unlinking, and restores the linked state', () => {
    class FakeElement extends EventTarget {
      readonly dataset: Record<string, string> = {};
      readonly style: Record<string, string> = {};
      readonly children: FakeElement[] = [];
      readonly attributes = new Map<string, string>();
      readonly classList = { toggle() {} };
      className = '';
      textContent: string | null = '';
      value = '';
      hidden = false;
      disabled = false;
      constructor(readonly tagName: string) { super(); }
      append(...children: FakeElement[]): void { this.children.push(...children); }
      replaceChildren(...children: FakeElement[]): void { this.children.splice(0, this.children.length, ...children); }
      setAttribute(name: string, value: string): void { this.attributes.set(name, value); }
      getAttribute(name: string): string | null { return this.attributes.get(name) ?? null; }
      getBoundingClientRect() { return { x: 0, y: 0, left: 0, top: 0, right: 400, bottom: 200, width: 400, height: 200 }; }
      remove(): void {}
      focus(): void {}
      click(): void { this.dispatchEvent(new Event('click')); }
    }
    class FakeObserver { observe(): void {} disconnect(): void {} }
    const documentStub = Object.assign(new EventTarget(), {
      documentElement: new FakeElement('HTML'),
      body: new FakeElement('BODY'),
      createElement: (tag: string) => new FakeElement(tag.toUpperCase()),
      createElementNS: (_namespace: string, tag: string) => new FakeElement(tag.toUpperCase()),
    });
    vi.stubGlobal('document', documentStub);
    vi.stubGlobal('window', new EventTarget());
    vi.stubGlobal('ResizeObserver', FakeObserver);
    vi.stubGlobal('MutationObserver', FakeObserver);

    const renderer = {
      registerView: vi.fn(), removeView: vi.fn(), updateView: vi.fn(), requestRender: vi.fn(),
      canvas: new FakeElement('CANVAS'),
    } as unknown as SharedRenderer;
    const client = {} as StudioClient;
    const run = (domain: unknown) => ({ id: 1, name: 'Run', domain, block_size: [4, 4], t0: 0, t_final: 1, fields: [], diagnostics: [] });
    const stateFor = (domain: unknown) => {
      const state = new StudioState();
      state.connection = 'connected'; state.runs = [run(domain)];
      return state;
    };
    const domainA = { origin: [0, 0], extent: [1, 1] };
    const domainB = { origin: [4, -1], extent: [2, 3] };
    const a = stateFor(domainA), b = stateFor({ origin: [0, 0], extent: [1, 1] }), c = stateFor(domainB);
    const definition = createFieldPanel(renderer);
    const otherDefinition = createFieldPanel(renderer);
    const independentDefinition = createFieldPanel(renderer, { linked: false });
    const first = definition.create({ state: a, client, panelId: 'first' });
    let second = otherDefinition.create({ state: b, client, panelId: 'second' });
    const third = definition.create({ state: c, client, panelId: 'third' });
    const fourth = independentDefinition.create({ state: a, client, panelId: 'fourth' });
    const find = (root: FakeElement, testid: string): FakeElement => {
      const queue = [root];
      while (queue.length) {
        const item = queue.shift()!;
        if (item.dataset.testid === testid) return item;
        queue.push(...item.children);
      }
      throw new Error(`Missing ${testid}`);
    };
    const control = (instance: typeof first, id: string) => find(instance.element as unknown as FakeElement, id);
    const zoom = (instance: typeof first) => {
      const event = new Event('wheel', { cancelable: true });
      Object.defineProperties(event, { deltaY: { value: -120 }, deltaMode: { value: 0 }, clientX: { value: 100 }, clientY: { value: 80 } });
      control(instance, 'field-plot').dispatchEvent(event);
    };
    const viewA = () => definition.getView('first')!;
    const viewB = () => otherDefinition.getView('second')!;
    const viewC = () => definition.getView('third')!;
    const link = () => control(second, 'field-link');

    expect(link().tagName).toBe('BUTTON');
    expect(link().getAttribute('aria-label')).toBeTruthy();
    expect(link().getAttribute('aria-pressed')).toBe('true');
    expect([viewA().linked, viewB().linked, viewC().linked]).toEqual([true, true, true]);
    const separate = viewC().camera;
    const independent = independentDefinition.getView('fourth')!;
    expect(independent.linked).toBe(false);
    expect(control(fourth, 'field-link').getAttribute('aria-pressed')).toBe('false');
    zoom(first);
    expect(viewB().camera).toEqual(viewA().camera);
    expect(viewC().camera).toEqual(separate);
    expect(independentDefinition.getView('fourth')!.camera).toEqual(independent.camera);

    link().click();
    expect(link().getAttribute('aria-pressed')).toBe('false');
    const detached = viewB().camera;
    zoom(first);
    expect(viewB().camera).toEqual(detached);
    link().click();
    expect(viewB().camera).toEqual(viewA().camera);

    b.runs = [run(domainB)]; b.changed();
    expect(viewB().camera).toEqual(viewC().camera);
    const newDomainCamera = viewB().camera;
    zoom(first);
    expect(viewB().camera).toEqual(newDomainCamera);
    zoom(second);
    expect(viewB().camera).toEqual(viewC().camera);
    b.runs = [run(null)]; b.changed();
    const firstCamera = viewA().camera, thirdCamera = viewC().camera;
    zoom(second);
    expect(viewC().camera).toEqual(thirdCamera);
    expect(viewA().camera).toEqual(firstCamera);
    b.runs = [run(domainA)]; b.changed();
    expect(viewB().camera).toEqual(viewA().camera);

    link().click();
    const saved = viewB();
    expect(saved.linked).toBe(false);
    otherDefinition.restoreViews([saved]);
    second.dispose?.();
    second = otherDefinition.create({ state: b, client, panelId: 'second' });
    expect(link().getAttribute('aria-pressed')).toBe('false');
    expect(viewB().linked).toBe(false);
    const restoredCamera = viewB().camera;
    zoom(first);
    expect(viewB().camera).toEqual(restoredCamera);
    link().click();
    expect(viewB().camera).toEqual(viewA().camera);

    first.dispose?.(); second.dispose?.(); third.dispose?.(); fourth.dispose?.();
  });
});
