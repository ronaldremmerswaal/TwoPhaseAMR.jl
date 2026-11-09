import { afterEach, describe, expect, it, vi } from 'vitest';
import type { StudioClient } from '../src/client';
import { formatBlockSize, formatBytes, formatCount, formatDomain, formatNumber, formatRate, formatStatusNumber, formatStore, PanelRegistry } from '../src/panels';
import { StudioState } from '../src/state';

describe('run metadata formatting', () => {
  it('shows physical domain bounds and block dimensions', () => {
    expect(formatDomain({ origin: [-1, 0.25], extent: [2, 1.75] })).toBe('[-1, 1] × [0.25, 2]');
    expect(formatBlockSize([4, 4])).toBe('4 × 4');
    expect(formatDomain({ origin: [0, 0], extent: [1, 1] })).toBe('[0, 1] × [0, 1]');
    expect(formatDomain({ origin: [0, null], extent: [1, 1] })).toBe('—');
    expect(formatDomain(null)).toBe('—');
    expect(formatBlockSize([4, '4'])).toBe('—');
    expect(formatBlockSize(null)).toBe('—');
  });

  it('rounds solver throughput to three significant figures', () => {
    expect(formatRate(31.12345)).toBe('31.1');
    expect(formatRate(0.00123456)).toBe('0.00123');
    expect(formatRate(12345.6)).toBe('12300');
    expect(formatRate(null)).toBe('—');
  });

  it('formats status times and the last step to four significant figures', () => {
    expect(formatStatusNumber(0.798283)).toBe('0.7983');
    expect(formatStatusNumber(5.451234e-14)).toBe('5.451e-14');
    expect(formatStatusNumber(null)).toBe('—');
  });

  it('formats byte units and actual frame counts without inventing missing values', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1000)).toBe('1 KB');
    expect(formatBytes(412_000_000)).toBe('412 MB');
    expect(formatBytes(2_000_000_000)).toBe('2 GB');
    expect(formatBytes(2_147_483_648)).toBe('2.15 GB');
    expect(formatBytes(-1)).toBe('—');
    expect(formatBytes(null)).toBe('—');
    expect(formatStore(412_000_000, 2_000_000_000, 37)).toBe('412 MB / 2 GB (37 frames)');
    expect(formatStore(null, null, null)).toBe('— / — (— frames)');
    expect(formatStore(1, 2, 1)).toBe('1 B / 2 B (1 frame)');
    expect(formatNumber(NaN)).toBe('—');
    expect(formatCount(1_234_567)).toBe('1234567');
    expect(formatCount(NaN)).toBe('—');
    expect(formatNumber(Infinity)).toBe('—');
  });
});

describe('panel registry', () => {
  it('notifies the panel picker only for successful registrations', () => {
    const registry = new PanelRegistry();
    const listener = vi.fn();
    const unsubscribe = registry.subscribe(listener);
    const create = () => ({ element: {} as HTMLElement });
    const field = { id: 'field', title: 'Field', create };
    const energy = { id: 'energy', title: 'Energy', create };
    registry.register(field);
    registry.register(energy);
    expect(registry.get('field')).toBe(field);
    expect(registry.get('missing')).toBeUndefined();
    expect(registry.all()).toEqual([field, energy]);
    expect(listener).toHaveBeenCalledTimes(2);
    expect(() => registry.register(field)).toThrow('Panel already registered: field');
    expect(() => registry.register({ id: '', title: 'Invalid', create })).toThrow();
    expect(listener).toHaveBeenCalledTimes(2);
    const copy = registry.all();
    copy.pop();
    expect(registry.all()).toEqual([field, energy]);
    unsubscribe();
    registry.register({ id: 'run', title: 'Run', create });
    expect(listener).toHaveBeenCalledTimes(2);
  });
});

describe('diagnostic panel interactions', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.doUnmock('uplot');
    vi.resetModules();
  });

  it('keeps toggle and scale state, excludes hidden data, reads sparse samples, and seeks', async () => {
    vi.useFakeTimers();
    class FakeElement extends EventTarget {
      readonly dataset: Record<string, string> = {};
      readonly style: Record<string, string> = {};
      readonly children: FakeElement[] = [];
      readonly attributes = new Map<string, string>();
      className = '';
      textContent: string | null = '';
      type = '';
      value = '';
      id = '';
      hidden = false;
      clientWidth = 400;
      clientHeight = 240;
      constructor(readonly tagName: string) { super(); }
      get childElementCount(): number { return this.children.length; }
      append(...children: FakeElement[]): void { this.children.push(...children); }
      replaceChildren(...children: FakeElement[]): void { this.children.splice(0, this.children.length, ...children); }
      setAttribute(name: string, value: string): void { this.attributes.set(name, value); }
      getAttribute(name: string): string | null { return this.attributes.get(name) ?? null; }
      contains(element: FakeElement): boolean { return this === element || this.children.some(child => child.contains(element)); }
      focus(): void { documentStub.activeElement = this; }
      click(): void { this.dispatchEvent(new Event('click')); }
      getBoundingClientRect(): { left: number } { return { left: 10 }; }
    }
    class FakeButton extends FakeElement {}
    const documentStub = {
      activeElement: null as FakeElement | null,
      documentElement: new FakeElement('HTML'),
      createElement(tag: string) { return tag === 'button' ? new FakeButton('BUTTON') : new FakeElement(tag.toUpperCase()); },
    };
    class FakeObserver { observe(): void {} disconnect(): void {} }
    interface FakePlotOptions {
      axes: { label: string }[];
      scales: { y: { distr?: number } };
      cursor: { dataIdx: (plot: FakePlot, series: number, closest: number, time: number) => number | null };
      hooks: { setCursor: ((plot: FakePlot) => void)[] };
    }
    class FakePlot {
      static instances: FakePlot[] = [];
      readonly over = new FakeElement('DIV');
      readonly cursor: { idx: number | null; left: number | null } = { idx: null, left: null };
      readonly redraw = vi.fn();
      readonly destroy = vi.fn();
      constructor(readonly options: FakePlotOptions, readonly data: (number | null | undefined)[][]) {
        FakePlot.instances.push(this);
      }
      posToVal(position: number): number { return position / 100; }
      setSize(): void {}
    }
    vi.stubGlobal('document', documentStub);
    vi.stubGlobal('HTMLButtonElement', FakeButton);
    vi.stubGlobal('window', { setTimeout, clearTimeout, devicePixelRatio: 1 });
    vi.stubGlobal('getComputedStyle', () => ({ getPropertyValue: () => '#000' }));
    vi.stubGlobal('ResizeObserver', FakeObserver);
    vi.stubGlobal('MutationObserver', FakeObserver);
    vi.doMock('uplot', () => ({ default: FakePlot }));
    // Install the plot mock before loading the panel to test this module boundary without a browser.
    const { diagnosticPanel } = await import('../src/diagnostic-panel');

    const state = new StudioState();
    state.connection = 'connected';
    state.timeline.initialize([
      { id: 1, time: 0, step: 0, kind: 'initial', topology: 1 },
      { id: 2, time: 5, step: 1, kind: 'keyframe', topology: 1 },
      { id: 3, time: 10, step: 2, kind: 'final', topology: 1 },
    ]);
    const histories = [
      { name: 'slow', info: { name: 'slow', label: 'Slow' }, times: [0, 10], components: [], values: [[1], [2]] },
      { name: 'fast', info: { name: 'fast', label: 'Fast' }, times: [2, 4], components: [], values: [[-1], [400]] },
    ];
    const client = { getData: vi.fn().mockResolvedValue({ series: histories }) } as unknown as StudioClient;
    const definition = diagnosticPanel({ key: 'energy', title: 'Energy (M L T⁻²)', kind: 'physical',
      scale: 'linear', dimension: null, names: ['slow', 'fast'] }, 'energy');
    const context = { state, client, panelId: 'energy-view' };
    const elements = (root: FakeElement, testid: string): FakeElement[] => {
      const pending = [root], found: FakeElement[] = [];
      while (pending.length) {
        const element = pending.shift()!;
        if (element.dataset.testid === testid) found.push(element);
        pending.push(...element.children);
      }
      return found;
    };

    let instance = definition.create(context);
    await vi.advanceTimersByTimeAsync(150);
    let root = instance.element as unknown as FakeElement;
    let [slow, fast] = elements(root, 'diagnostic-toggle');
    expect([slow.tagName, slow.type, fast.tagName, fast.type]).toEqual(['BUTTON', 'button', 'BUTTON', 'button']);
    expect(fast.getAttribute('aria-pressed')).toBe('true');
    expect(elements(root, 'diagnostic-show-all')[0].getAttribute('aria-disabled')).toBe('true');
    let plot = FakePlot.instances.at(-1)!;
    expect(plot.options.axes.map(axis => axis.label)).toEqual(['t', 'Energy (M L T⁻²)']);
    expect(plot.options.cursor.dataIdx(plot, 1, 1, 2)).toBe(0);
    expect(plot.options.cursor.dataIdx(plot, 2, 1, 2)).toBe(1);
    plot.cursor.idx = 1;
    plot.cursor.left = 200;
    plot.options.hooks.setCursor[0](plot);
    const readout = elements(root, 'diagnostic-readout')[0];
    expect(readout.textContent).toContain('t = 2');
    expect(readout.textContent).toContain('Slow: 1 (t = 0)');
    expect(readout.textContent).toContain('Fast: -1 (t = 2)');

    const scale = elements(root, 'diagnostic-scale')[0];
    expect(elements(root, 'diagnostic-scale-note')[0].hidden).toBe(true);
    scale.value = 'log';
    scale.dispatchEvent(new Event('change'));
    expect(elements(root, 'diagnostic-scale-note')[0].hidden).toBe(false);
    plot = FakePlot.instances.at(-1)!;
    expect(plot.options.scales.y.distr).toBe(3);
    plot.cursor.idx = 1;
    plot.cursor.left = 200;
    plot.options.hooks.setCursor[0](plot);
    expect(readout.textContent).toContain('Fast: -1 (omitted) (t = 2)');

    fast.focus();
    fast.click();
    expect(fast.getAttribute('aria-pressed')).toBe('false');
    expect(elements(root, 'diagnostic-show-all')[0].getAttribute('aria-disabled')).toBe('false');
    expect(documentStub.activeElement).toBe(fast);
    plot = FakePlot.instances.at(-1)!;
    expect(plot.data).toEqual([[0, 10], [1, 2]]);
    plot.cursor.idx = 0;
    plot.cursor.left = 200;
    plot.options.hooks.setCursor[0](plot);
    expect(readout.textContent).toContain('Slow: 1 (t = 0)');
    expect(readout.textContent).not.toContain('Fast');
    state.step = 42;
    state.changed();
    await vi.advanceTimersByTimeAsync(150);
    expect(elements(root, 'diagnostic-toggle')[1]).toBe(fast);
    expect(fast.getAttribute('aria-pressed')).toBe('false');
    expect(documentStub.activeElement).toBe(fast);
    expect(definition.getConfigs()[0]).toMatchObject({
      scale: 'log', hidden: [{ name: 'fast', component: 1 }],
    });

    instance.dispose?.();
    expect(definition.getConfigs()).toEqual([]);
    instance = definition.create(context);
    await vi.advanceTimersByTimeAsync(150);
    root = instance.element as unknown as FakeElement;
    [slow, fast] = elements(root, 'diagnostic-toggle');
    expect(fast.getAttribute('aria-pressed')).toBe('false');
    expect(elements(root, 'diagnostic-scale')[0].value).toBe('log');
    plot = FakePlot.instances.at(-1)!;
    state.timeline.playing = true;
    const click = new Event('click');
    Object.defineProperty(click, 'clientX', { value: 510 });
    plot.over.dispatchEvent(click);
    expect(state.timeline.mode).toBe('history');
    expect(state.timeline.selectedId).toBe(2);
    expect(state.timeline.playing).toBe(false);
    const redraws = plot.redraw.mock.calls.length;
    state.timeline.current!.time = 6;
    state.changed();
    expect(plot.redraw).toHaveBeenCalledTimes(redraws + 1);
    elements(root, 'diagnostic-show-all')[0].click();
    expect(fast.getAttribute('aria-pressed')).toBe('true');
    expect(definition.getConfigs()[0].hidden).toEqual([]);
    instance.dispose?.();
  });
});
