import { describe, expect, it } from 'vitest';
import {
  alignDiagnostics, diagnosticComponentKey, diagnosticSampleIndexes, DiagnosticPanelConfigurations, DiagnosticPanelSettings,
  dimensionText, groupDiagnostics, isolatedDiagnosticPoints, nearestDiagnosticSampleIndex, type DiagnosticGroup,
} from '../src/diagnostic-data';

describe('diagnostic panels', () => {
  it('groups by kind, dimension and scale, and separates unknown dimensions', () => {
    const energy = { mass: 1, length: 1, time: -2 };
    const area = { mass: 0, length: 2, time: 0 };
    const groups = groupDiagnostics([
      { name: 'kinetic', label: 'Kinetic', dimension: energy },
      { name: 'surface', label: 'Surface', dimension: energy },
      { name: 'volume', dimension: area, components: ['liquid', 'gas'] },
      { name: 'count', dimension: area, kind: 'count' },
      { name: 'a' }, { name: 'b' },
      { name: 'c', units: 'm/s' }, { name: 'd', units: 'm/s' },
    ]);
    expect(dimensionText(energy)).toBe('M L T⁻²');
    expect(groups.map(group => group.names)).toEqual([
      ['kinetic', 'surface'], ['volume'], ['count'], ['a'], ['b'], ['c', 'd'],
    ]);
    expect(groups[0].title).toBe('Energy (M L T⁻²)');
    expect(groups[1].title).toBe('Area (L²)');
  });

  it('aligns differently scheduled components and preserves missing samples', () => {
    const data = alignDiagnostics([
      { name: 'volume', info: { name: 'volume', label: 'Volume' }, times: [0, 2], components: ['liquid', 'gas'], values: [[1, 2], [3, 4]] },
      { name: 'energy', info: { name: 'energy' }, times: [0, 1, 2], components: [], values: [[5], [null], [7]] },
    ]);
    expect(data.times).toEqual([0, 1, 2]);
    expect(data.series).toEqual([
      { name: 'volume', component: 1, label: 'Volume · liquid', values: [1, undefined, 3] },
      { name: 'volume', component: 2, label: 'Volume · gas', values: [2, undefined, 4] },
      { name: 'energy', component: 1, label: 'energy', values: [5, null, 7] },
    ]);
  });

  it('omits nonpositive and nonfinite values on log scales without taking absolute values', () => {
    const history = {
      name: 'residual', info: { name: 'residual' }, times: [0, 1, 2, 3, 4, 5], components: [],
      values: [[-1], [0], [1e-6], [NaN], [Infinity], [-0]],
    };
    expect(alignDiagnostics([history], true).series[0].values).toEqual([null, null, 1e-6, null, null, null]);
    expect(alignDiagnostics([history]).series[0].values).toEqual([-1, 0, 1e-6, null, null, -0]);
  });

  it('keeps log omissions as genuine gaps even when another history adds intermediate times', () => {
    const histories = [
      { name: 'a', info: { name: 'a' }, times: [0, 2, 4, 6], components: [], values: [[5], [-2], [9], [12]] },
      { name: 'b', info: { name: 'b' }, times: [1, 3, 5], components: [], values: [[1], [2], [3]] },
    ];
    const linear = alignDiagnostics(histories);
    const logarithmic = alignDiagnostics(histories, true);
    expect(linear.series[0].values).toEqual([5, undefined, -2, undefined, 9, undefined, 12]);
    expect(logarithmic.series[0].values).toEqual([5, null, null, null, 9, undefined, 12]);
    expect(isolatedDiagnosticPoints(logarithmic.series[0].values, true)).toEqual([0]);
    expect(isolatedDiagnosticPoints([null, 2, null, 4, null], true)).toEqual([1, 3]);
    expect(isolatedDiagnosticPoints([null, 2, null, 4, null], false)).toEqual([]);
    expect(isolatedDiagnosticPoints([undefined, null, 2, undefined], false)).toEqual([2]);
  });

  it('uses exact names and one-based component indices rather than display labels for visibility', () => {
    const histories = [
      { name: 'a', info: { name: 'a', label: 'Same' }, times: [0], components: ['x', 'x'], values: [[1, 2]] },
      { name: 'a:1', info: { name: 'a:1', label: 'Same' }, times: [0], components: ['x'], values: [[3]] },
    ];
    const settings = new DiagnosticPanelSettings('linear');
    settings.setVisible({ name: 'a', component: 2 }, false);
    expect(diagnosticComponentKey({ name: 'a', component: 2 })).not.toBe(diagnosticComponentKey({ name: 'a:1', component: 2 }));
    expect(alignDiagnostics(histories, false, settings).series.map(({ name, component }) => [name, component]))
      .toEqual([['a', 1], ['a:1', 1]]);
    expect(alignDiagnostics([...histories].reverse(), false, settings).series.map(({ name, component }) => [name, component]))
      .toEqual([['a:1', 1], ['a', 1]]);
    const refetched = [
      { ...histories[0], times: [0, 1], components: ['u', 'v', 'w'], values: [[1, 2, 3], [4, 5, 6]] },
      histories[1],
    ];
    expect(alignDiagnostics(refetched, false, settings).series.map(({ name, component }) => [name, component]))
      .toEqual([['a', 1], ['a', 3], ['a:1', 1]]);
    settings.setVisible({ name: 'a', component: 2 }, true);
    expect(settings.hasHidden()).toBe(false);
  });

  it('passes only visible, scale-valid values to the plot for autoscaling', () => {
    const histories = [
      { name: 'velocity', info: { name: 'velocity' }, times: [0, 1], components: ['x', 'y'], values: [[2, -1000], [3, 2000]] },
      { name: 'pressure', info: { name: 'pressure' }, times: [0, 1], components: [], values: [[-4], [4]] },
    ];
    const settings = new DiagnosticPanelSettings('linear');
    settings.setVisible({ name: 'velocity', component: 2 }, false);
    let plotted = alignDiagnostics(histories, false, settings).series;
    let values = plotted.flatMap(series => series.values.filter((value): value is number => typeof value === 'number'));
    expect([Math.min(...values), Math.max(...values)]).toEqual([-4, 4]);
    settings.scale = 'log';
    plotted = alignDiagnostics(histories, settings.scale === 'log', settings).series;
    values = plotted.flatMap(series => series.values.filter((value): value is number => typeof value === 'number'));
    expect([Math.min(...values), Math.max(...values)]).toEqual([2, 4]);
    settings.setVisible({ name: 'velocity', component: 1 }, false);
    settings.setVisible({ name: 'pressure', component: 1 }, false);
    expect(alignDiagnostics(histories, true, settings)).toEqual({ times: [], series: [] });
    settings.showAll();
    expect(alignDiagnostics(histories, true, settings).series).toHaveLength(3);
  });

  it('removes times from entirely hidden histories as well as their y values', () => {
    const histories = [
      { name: 'visible', info: { name: 'visible' }, times: [1, 2], components: ['x', 'y'], values: [[3, 30], [4, 40]] },
      { name: 'hidden', info: { name: 'hidden' }, times: [-100, 1, 100], components: [], values: [[1000], [2000], [3000]] },
    ];
    const settings = new DiagnosticPanelSettings('linear');
    settings.setVisible({ name: 'hidden', component: 1 }, false);
    settings.setVisible({ name: 'visible', component: 2 }, false);
    const plotted = alignDiagnostics(histories, false, settings);
    expect(plotted.times).toEqual([1, 2]);
    expect(plotted.series.map(({ name, component, values }) => ({ name, component, values }))).toEqual([
      { name: 'visible', component: 1, values: [3, 4] },
    ]);
    expect(alignDiagnostics(histories, true, settings).times).toEqual([1, 2]);
    settings.showAll();
    expect(alignDiagnostics(histories, false, settings).times).toEqual([-100, 1, 2, 100]);
  });

  it('finds each component’s nearest actual sample by time without skipping explicit gaps', () => {
    const times = [0, 1, 2, 50, 100];
    const indexes = diagnosticSampleIndexes([4, undefined, null, undefined, 8]);
    expect(indexes).toEqual([0, 2, 4]);
    expect(nearestDiagnosticSampleIndex(times, indexes, -1)).toBe(0);
    expect(nearestDiagnosticSampleIndex(times, indexes, 1)).toBe(0);
    expect(nearestDiagnosticSampleIndex(times, indexes, 3)).toBe(2);
    expect(nearestDiagnosticSampleIndex(times, indexes, 50)).toBe(2);
    expect(nearestDiagnosticSampleIndex(times, indexes, 99)).toBe(4);
    expect(nearestDiagnosticSampleIndex(times, indexes, 101)).toBe(4);
    expect(nearestDiagnosticSampleIndex(times, [], 1)).toBeNull();
    expect(nearestDiagnosticSampleIndex(times, indexes, NaN)).toBeNull();
  });

  it('keeps component controls available before samples arrive and after all values are omitted', () => {
    const history = {
      name: 'vector', info: { name: 'vector' }, times: [], components: ['x', 'y'], values: [],
    };
    const empty = alignDiagnostics([history], true);
    expect(empty.times).toEqual([]);
    expect(empty.series.map(({ name, component, values }) => ({ name, component, values }))).toEqual([
      { name: 'vector', component: 1, values: [] },
      { name: 'vector', component: 2, values: [] },
    ]);
    const settings = new DiagnosticPanelSettings('log');
    settings.setVisible({ name: 'vector', component: 1 }, false);
    settings.setVisible({ name: 'vector', component: 2 }, false);
    expect(alignDiagnostics([history], true, settings).series).toEqual([]);
    settings.showAll();
    const nonpositive = [{
      name: 'vector', info: { name: 'vector' }, times: [0, 1], components: ['x', 'y'],
      values: [[0, -1], [-2, 0]],
    }];
    const shown = alignDiagnostics(nonpositive, true, settings).series;
    expect(shown).toHaveLength(2);
    expect(shown.every(series => series.values.every(value => value == null))).toBe(true);
  });

  it('retains independent settings across panel recreation and exports only active instances', () => {
    const group: DiagnosticGroup = {
      key: 'energy', title: 'Energy', kind: 'physical', scale: 'linear', dimension: null, names: ['a', 'b'],
    };
    const panels = new DiagnosticPanelConfigurations(group);
    expect(panels.getConfigs()).toEqual([]);
    const first = panels.activate('left');
    const second = panels.activate('right');
    expect(panels.getConfigs()).toEqual([
      { panelId: 'left', names: ['a', 'b'], scale: 'linear', hidden: [] },
      { panelId: 'right', names: ['a', 'b'], scale: 'linear', hidden: [] },
    ]);
    first.scale = 'log';
    first.setVisible({ name: 'a', component: 2 }, false);
    second.setVisible({ name: 'b', component: 1 }, false);
    expect(panels.getConfigs()).toEqual([
      { panelId: 'left', names: ['a', 'b'], scale: 'log', hidden: [{ name: 'a', component: 2 }] },
      { panelId: 'right', names: ['a', 'b'], scale: 'linear', hidden: [{ name: 'b', component: 1 }] },
    ]);
    expect(() => panels.activate('left')).toThrow('Diagnostic panel already exists: left');
    const snapshot = panels.getConfigs();
    snapshot[0].names.push('not a member');
    snapshot[0].hidden[0].component = 99;
    expect(panels.getConfigs()[0]).toEqual({
      panelId: 'left', names: ['a', 'b'], scale: 'log', hidden: [{ name: 'a', component: 2 }],
    });
    panels.deactivate('left');
    expect(panels.getConfigs().map(config => config.panelId)).toEqual(['right']);
    expect(panels.activate('left')).toBe(first);
    expect(first.scale).toBe('log');
    expect(first.isVisible({ name: 'a', component: 2 })).toBe(false);
    expect(second.isVisible({ name: 'a', component: 2 })).toBe(true);
    group.names.splice(0, group.names.length, 'b');
    expect(panels.getConfigs().find(config => config.panelId === 'left')).toEqual({
      panelId: 'left', names: ['b'], scale: 'log', hidden: [],
    });
    group.names.unshift('a');
    expect(panels.getConfigs().find(config => config.panelId === 'left')?.hidden).toEqual([{ name: 'a', component: 2 }]);
    first.showAll();
    expect(first.hasHidden()).toBe(false);
    expect(second.hasHidden()).toBe(true);
  });

  it('starts a new panel with the group scale and leaves other ids independent', () => {
    const group: DiagnosticGroup = {
      key: 'defect', title: 'Defect', kind: 'defect', scale: 'log', dimension: null, names: ['residual'],
    };
    const panels = new DiagnosticPanelConfigurations(group);
    const first = panels.activate('first');
    expect(first.scale).toBe('log');
    first.scale = 'linear';
    expect(panels.activate('second').scale).toBe('log');
    panels.deactivate('first');
    expect(panels.activate('first').scale).toBe('linear');
  });
});
