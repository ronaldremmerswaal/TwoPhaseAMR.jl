import { describe, expect, it } from 'vitest';
import { alignDiagnostics, dimensionText, groupDiagnostics } from '../src/diagnostic-data';

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
      { label: 'Volume · liquid', values: [1, null, 3] },
      { label: 'Volume · gas', values: [2, null, 4] },
      { label: 'energy', values: [5, null, 7] },
    ]);
  });

  it('does not pass nonpositive values to log scales', () => {
    const data = alignDiagnostics([{ name: 'residual', info: { name: 'residual' }, times: [0, 1, 2], components: [], values: [[-1], [0], [1e-6]] }], true);
    expect(data.series[0].values).toEqual([null, null, 1e-6]);
  });
});
