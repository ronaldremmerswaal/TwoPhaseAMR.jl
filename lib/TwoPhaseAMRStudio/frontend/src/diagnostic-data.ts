import type { NamedMetadata } from './types';

export interface DiagnosticDimension { mass: number; length: number; time: number }
export interface DiagnosticMetadata extends NamedMetadata {
  kind?: string;
  scale?: string;
  dimension?: DiagnosticDimension | null;
  components?: string[];
}
export interface DiagnosticGroup {
  key: string;
  title: string;
  kind: string;
  scale: string;
  dimension: DiagnosticDimension | null;
  names: string[];
}
export interface DiagnosticSeries {
  name: string;
  info: DiagnosticMetadata;
  times: (number | null)[];
  components: string[];
  values: (number | null)[][];
}
export type DiagnosticScale = 'linear' | 'log';
export interface DiagnosticComponent { name: string; component: number }
export interface AlignedSeries extends DiagnosticComponent {
  label: string;
  /** undefined is an alignment artifact; null is an explicitly missing sample. */
  values: (number | null | undefined)[];
}
export interface DiagnosticPanelConfig {
  panelId: string;
  names: string[];
  scale: DiagnosticScale;
  hidden: DiagnosticComponent[];
}

/** Format axis splits with enough precision to distinguish adjacent ticks. */
export function formatDiagnosticTicks(ticks: readonly (number | null)[], increment: number): (string | null)[] {
  let maxMagnitude = 0;
  let spacing = Number.isFinite(increment) && increment !== 0 ? Math.abs(increment) : Infinity;
  let previous: number | undefined;
  for (const value of ticks) {
    if (value === null || !Number.isFinite(value)) continue;
    maxMagnitude = Math.max(maxMagnitude, Math.abs(value));
    if (previous !== undefined) {
      const difference = Math.abs(value - previous);
      if (difference > 0) spacing = Math.min(spacing, difference);
    }
    previous = value;
  }
  // Keep one guard digit for increments such as 2.5 × 10ⁿ.
  let digits = maxMagnitude > 0 && Number.isFinite(spacing)
    ? Math.max(1, Math.min(17, Math.floor(Math.log10(maxMagnitude)) - Math.floor(Math.log10(spacing)) + 2))
    : 6;
  while (true) {
    const labels = ticks.map(value => value === null ? null :
      Number.isFinite(value) ? Number(value.toPrecision(digits)).toString() : '—');
    let lastValue: number | undefined, lastLabel: string | null = null;
    let collision = false;
    for (let index = 0; index < ticks.length; index++) {
      const value = ticks[index];
      if (value === null || !Number.isFinite(value)) continue;
      if (lastValue !== undefined && value !== lastValue && labels[index] === lastLabel) {
        collision = true;
        break;
      }
      lastValue = value;
      lastLabel = labels[index];
    }
    if (!collision || digits >= 17) return labels;
    digits++;
  }
}

/** Identity is independent of the display label and uses one-based component indices. */
export function diagnosticComponentKey({ name, component }: DiagnosticComponent): string {
  return JSON.stringify([name, component]);
}

export class DiagnosticPanelSettings {
  private readonly hidden = new Map<string, DiagnosticComponent>();

  constructor(public scale: DiagnosticScale) {}

  isVisible(component: DiagnosticComponent): boolean {
    return !this.hidden.has(diagnosticComponentKey(component));
  }

  setVisible(component: DiagnosticComponent, visible: boolean): void {
    const key = diagnosticComponentKey(component);
    if (visible) this.hidden.delete(key);
    else this.hidden.set(key, { name: component.name, component: component.component });
  }

  showAll(): void { this.hidden.clear(); }
  hasHidden(): boolean { return this.hidden.size > 0; }

  hiddenFor(names: readonly string[]): DiagnosticComponent[] {
    const members = new Set(names);
    return [...this.hidden.values()].filter(component => members.has(component.name)).map(component => ({ ...component }));
  }
}

/** Settings outlive dock instances, while exported configurations include open instances only. */
export class DiagnosticPanelConfigurations {
  private readonly settings = new Map<string, DiagnosticPanelSettings>();
  private readonly active = new Set<string>();

  constructor(private readonly group: DiagnosticGroup) {}

  activate(panelId: string): DiagnosticPanelSettings {
    if (this.active.has(panelId)) throw new Error(`Diagnostic panel already exists: ${panelId}`);
    let settings = this.settings.get(panelId);
    if (!settings) {
      settings = new DiagnosticPanelSettings(this.group.scale === 'log' ? 'log' : 'linear');
      this.settings.set(panelId, settings);
    }
    this.active.add(panelId);
    return settings;
  }

  deactivate(panelId: string): void { this.active.delete(panelId); }

  getConfigs(): DiagnosticPanelConfig[] {
    return [...this.active].map(panelId => {
      const settings = this.settings.get(panelId)!;
      const names = [...this.group.names];
      return { panelId, names, scale: settings.scale, hidden: settings.hiddenFor(names) };
    });
  }
}

/** Sample indices exclude alignment artifacts but retain explicitly missing values. */
export function diagnosticSampleIndexes(values: readonly (number | null | undefined)[]): number[] {
  const indexes: number[] = [];
  values.forEach((value, index) => { if (value !== undefined) indexes.push(index); });
  return indexes;
}

/** Find the closest actual sample by time, preferring the earlier one on a tie. */
export function nearestDiagnosticSampleIndex(times: readonly number[], indexes: readonly number[], time: number): number | null {
  if (!indexes.length || !Number.isFinite(time)) return null;
  let low = 0, high = indexes.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (times[indexes[middle]] < time) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return indexes[0];
  if (low === indexes.length) return indexes[low - 1];
  const before = indexes[low - 1], after = indexes[low];
  return time - times[before] <= times[after] - time ? before : after;
}

/** Return points which have no line segment to make them visible. */
export function isolatedDiagnosticPoints(values: readonly (number | null | undefined)[], logarithmic: boolean): number[] {
  const points: number[] = [];
  let first = -1;
  let count = 0;
  const finish = () => {
    if (count === 1) points.push(first);
    first = -1;
    count = 0;
  };
  values.forEach((value, index) => {
    if (typeof value === 'number') {
      if (count++ === 0) first = index;
    } else if (logarithmic && value === null) finish();
  });
  finish();
  return points;
}

const superscript: Record<string, string> = {
  '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
};

export function dimensionText(dimension: DiagnosticDimension): string {
  const parts: string[] = [];
  for (const [symbol, exponent] of [['M', dimension.mass], ['L', dimension.length], ['T', dimension.time]] as const) {
    if (!exponent) continue;
    parts.push(symbol + (exponent === 1 ? '' : [...String(exponent)].map(digit => superscript[digit] ?? digit).join('')));
  }
  return parts.join(' ') || '1';
}

function groupTitle(info: DiagnosticMetadata): string {
  if (info.units) return info.units;
  if (!info.dimension) return info.label || info.name;
  const dimension = dimensionText(info.dimension);
  if (info.kind === 'count') return 'Count';
  if (info.kind === 'defect') return `${info.label || info.name} (${dimension})`;
  const quantity: Record<string, string> = {
    'M L T⁻²': 'Energy', 'M T⁻¹': 'Momentum', 'L²': 'Area', 'M L⁻¹': 'Mass',
    'L': 'Length', 'L T⁻¹': 'Velocity', 'T': 'Time',
  };
  return quantity[dimension] ? `${quantity[dimension]} (${dimension})` : dimension;
}

/** Match the core diagnostic_groups rule, including the unknown-dimension fallback. */
export function groupDiagnostics(infos: DiagnosticMetadata[]): DiagnosticGroup[] {
  const groups = new Map<string, DiagnosticGroup>();
  for (const info of infos) {
    const kind = info.kind ?? 'physical';
    const scale = info.scale ?? 'linear';
    const dimension = info.dimension ?? null;
    const key = dimension
      ? JSON.stringify(['dimension', kind, dimension.mass, dimension.length, dimension.time, scale])
      : info.units ? JSON.stringify(['units', kind, info.units, scale]) : JSON.stringify(['single', info.name]);
    const existing = groups.get(key);
    if (existing) existing.names.push(info.name);
    else groups.set(key, { key, title: groupTitle(info), kind, scale, dimension, names: [info.name] });
  }
  return [...groups.values()];
}


/** Align samples without fabricating them; hidden histories do not affect either axis. */
export function alignDiagnostics(histories: DiagnosticSeries[], logarithmic = false, settings?: DiagnosticPanelSettings): { times: number[]; series: AlignedSeries[] } {
  const selected = histories.map(history => ({
    history, count: history.values.reduce((maximum, row) => Math.max(maximum, row.length), history.components.length),
  })).filter(({ history, count }) => {
    if (!settings) return true;
    for (let component = 1; component <= count; component++) {
      if (settings.isVisible({ name: history.name, component })) return true;
    }
    return false;
  });
  const times = [...new Set(selected.flatMap(({ history }) => history.times.filter((time): time is number =>
    typeof time === 'number' && Number.isFinite(time))))].sort((a, b) => a - b);
  const indexes = new Map(times.map((time, index) => [time, index]));
  const series: AlignedSeries[] = [];
  for (const { history, count } of selected) {
    for (let component = 0; component < count; component++) {
      if (settings && !settings.isVisible({ name: history.name, component: component + 1 })) continue;
      const values = Array<number | null | undefined>(times.length).fill(undefined);
      history.times.forEach((time, row) => {
        if (typeof time !== 'number') return;
        const index = indexes.get(time);
        if (index === undefined) return;
        const value = history.values[row]?.[component];
        values[index] = typeof value === 'number' && Number.isFinite(value) && (!logarithmic || value > 0) ? value : null;
      });
      if (logarithmic) {
        // uPlot skips undefined alignment artifacts but clips at nulls. Extend each
        // gap to the next actual samples so other histories cannot bridge it.
        for (let index = 0; index < values.length; index++) {
          if (values[index] !== null) continue;
          for (let before = index - 1; before >= 0 && values[before] === undefined; before--) values[before] = null;
          let after = index + 1;
          while (after < values.length && values[after] === undefined) values[after++] = null;
          index = after - 1;
        }
      }
      const suffix = history.components[component] ?? (count > 1 ? String(component + 1) : '');
      series.push({
        name: history.name, component: component + 1,
        label: `${history.info.label || history.name}${suffix ? ` · ${suffix}` : ''}`, values,
      });
    }
  }
  return { times, series };
}
