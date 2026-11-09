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
export interface AlignedSeries { label: string; values: (number | null)[] }

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

/** Align independently sampled diagnostic components without fabricating samples. */
export function alignDiagnostics(histories: DiagnosticSeries[], logarithmic = false): { times: number[]; series: AlignedSeries[] } {
  const times = [...new Set(histories.flatMap(history => history.times.filter((time): time is number =>
    typeof time === 'number' && Number.isFinite(time))))].sort((a, b) => a - b);
  const indexes = new Map(times.map((time, index) => [time, index]));
  const series: AlignedSeries[] = [];
  for (const history of histories) {
    const count = history.values.reduce((maximum, row) => Math.max(maximum, row.length), history.components.length);
    for (let component = 0; component < count; component++) {
      const values = Array<number | null>(times.length).fill(null);
      history.times.forEach((time, row) => {
        if (typeof time !== 'number') return;
        const index = indexes.get(time);
        const value = history.values[row]?.[component];
        if (index !== undefined && typeof value === 'number' && Number.isFinite(value) && (!logarithmic || value > 0)) values[index] = value;
      });
      const suffix = history.components[component] ?? (count > 1 ? String(component + 1) : '');
      series.push({ label: `${history.info.label || history.name}${suffix ? ` · ${suffix}` : ''}`, values });
    }
  }
  return { times, series };
}
