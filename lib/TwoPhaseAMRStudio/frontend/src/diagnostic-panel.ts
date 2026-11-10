import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import './diagnostic-panel.css';
import {
  alignDiagnostics, diagnosticComponentKey, diagnosticSampleIndexes, DiagnosticPanelConfigurations,
  formatDiagnosticTicks, isolatedDiagnosticPoints, nearestDiagnosticSampleIndex,
  type AlignedSeries, type DiagnosticGroup, type DiagnosticPanelConfig, type DiagnosticScale, type DiagnosticSeries,
} from './diagnostic-data';
import { formatNumber, type PanelDefinition } from './panels';

export type { DiagnosticPanelConfig } from './diagnostic-data';

export interface DiagnosticPanelDefinition extends PanelDefinition {
  /** Snapshots of the currently open instances of this diagnostic group. */
  getConfigs(): DiagnosticPanelConfig[];
}

let scaleNoteId = 0;

/** One dockable chart for exactly one core diagnostic group. */
export function diagnosticPanel(group: DiagnosticGroup, id: string): DiagnosticPanelDefinition {
  const configurations = new DiagnosticPanelConfigurations(group);
  return {
    id, title: group.title,
    getConfigs() { return configurations.getConfigs(); },
    create({ state, client, panelId }) {
      const settings = configurations.activate(panelId);
      const element = document.createElement('section');
      element.className = 'diagnostic-panel';
      element.dataset.testid = 'diagnostic-panel';
      element.dataset.panelId = panelId;
      element.dataset.dimension = group.dimension ? JSON.stringify(group.dimension) : '';
      element.setAttribute('aria-label', group.title);

      const toolbar = document.createElement('div');
      toolbar.className = 'diagnostic-toolbar';
      const scaleLabel = document.createElement('label');
      scaleLabel.textContent = 'Y scale';
      const scale = document.createElement('select');
      scale.dataset.testid = 'diagnostic-scale';
      for (const [value, text] of [['linear', 'Linear'], ['log', 'Log']] as const) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = text;
        scale.append(option);
      }
      scale.value = settings.scale;
      scaleLabel.append(scale);
      const showAll = document.createElement('button');
      showAll.type = 'button';
      showAll.dataset.testid = 'diagnostic-show-all';
      showAll.textContent = 'Show all';
      toolbar.append(scaleLabel, showAll);

      const legend = document.createElement('div');
      legend.className = 'diagnostic-legend';
      legend.setAttribute('role', 'group');
      legend.setAttribute('aria-label', 'Diagnostic components');
      const note = document.createElement('p');
      note.className = 'diagnostic-scale-note';
      note.dataset.testid = 'diagnostic-scale-note';
      note.id = `diagnostic-scale-note-${++scaleNoteId}`;
      note.textContent = 'Log scale omits zero and negative values, leaving gaps.';
      scale.setAttribute('aria-describedby', note.id);
      const readout = document.createElement('div');
      readout.className = 'diagnostic-readout';
      readout.dataset.testid = 'diagnostic-readout';
      readout.setAttribute('aria-label', 'Nearest diagnostic samples');
      readout.hidden = true;
      const plot = document.createElement('div');
      plot.className = 'diagnostic-plot';
      plot.dataset.testid = 'diagnostic-plot';
      plot.dataset.xLabel = 't';
      plot.dataset.yLabel = group.title;
      plot.setAttribute('aria-label', `${group.title} versus time`);
      const status = document.createElement('div');
      status.className = 'diagnostic-status';
      status.dataset.testid = 'diagnostic-status';
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      element.append(toolbar, legend, note, status, readout, plot);

      let chart: uPlot | null = null;
      let histories: DiagnosticSeries[] = [];
      let loadError: string | null = null;
      let disposed = false;
      let loading = false;
      let refreshTimer: number | undefined;
      let requestedSignature = '';
      let wantedSignature = '';
      let cursorTime: number | null = null;
      const buttons = new Map<string, HTMLButtonElement>();

      function css(name: string): string {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      }

      function updateLegend(series: AlignedSeries[], colors: string[]): void {
        const nextButtons = series.map((item, index) => {
          const key = diagnosticComponentKey(item);
          let button = buttons.get(key);
          if (!button) {
            const component = { name: item.name, component: item.component };
            button = document.createElement('button');
            button.type = 'button';
            button.className = 'diagnostic-series';
            button.dataset.testid = 'diagnostic-toggle';
            button.dataset.name = component.name;
            button.dataset.component = String(component.component);
            const swatch = document.createElement('span');
            swatch.className = 'diagnostic-series-swatch';
            swatch.setAttribute('aria-hidden', 'true');
            swatch.textContent = '━';
            const label = document.createElement('span');
            label.className = 'diagnostic-series-label';
            const visibility = document.createElement('span');
            visibility.className = 'diagnostic-series-visibility';
            visibility.setAttribute('aria-hidden', 'true');
            button.append(swatch, label, visibility);
            button.addEventListener('click', () => {
              settings.setVisible(component, !settings.isVisible(component));
              draw();
            });
            buttons.set(key, button);
          }
          const visible = settings.isVisible(item);
          button.setAttribute('aria-pressed', String(visible));
          button.setAttribute('aria-label', `${item.label} (${item.name}, component ${item.component})`);
          (button.children[0] as HTMLElement).style.color = colors[index % colors.length];
          button.children[1].textContent = item.label;
          button.children[2].textContent = visible ? 'Visible' : 'Hidden';
          return button;
        });
        const currentKeys = new Set(series.map(diagnosticComponentKey));
        const focused = document.activeElement instanceof HTMLButtonElement && legend.contains(document.activeElement)
          ? document.activeElement : null;
        for (const [key, button] of buttons) {
          if (!currentKeys.has(key)) { legend.removeChild(button); buttons.delete(key); }
        }
        nextButtons.forEach((button, index) => {
          if (legend.children[index] !== button) legend.insertBefore(button, legend.children[index] ?? null);
        });
        if (focused && document.activeElement !== focused) {
          if (nextButtons.includes(focused)) focused.focus({ preventScroll: true });
          else scale.focus({ preventScroll: true });
        }
        showAll.setAttribute('aria-disabled', String(!settings.hasHidden()));
      }

      function draw(): void {
        if (disposed) return;
        const logarithmic = settings.scale === 'log';
        note.hidden = !logarithmic;
        const data = alignDiagnostics(histories, logarithmic, settings);
        const linearData = logarithmic ? alignDiagnostics(histories, false, settings) : data;
        const legendSeries = settings.hasHidden() ? alignDiagnostics(histories).series : linearData.series;
        const colors = [css('--diagnostic-series-1'), css('--diagnostic-series-2'), css('--diagnostic-series-3')];
        updateLegend(legendSeries, colors);
        const visible = data.series;
        const hasValues = visible.some(series => series.values.some(value => typeof value === 'number'));
        let message = loadError ?? '';
        if (!message && legendSeries.length && !visible.length) message = 'All components are hidden. Use Show all to restore them.';
        else if (!message && (!data.times.length || !legendSeries.length)) message = 'Waiting for diagnostics';
        else if (!message && !hasValues) message = logarithmic
          ? 'No positive diagnostic values among the visible components.'
          : 'No diagnostic values among the visible components.';
        status.textContent = message;
        status.hidden = !message;
        readout.hidden = !hasValues;
        readout.textContent = 'Hover over the chart to inspect the nearest sample.';

        chart?.destroy();
        chart = null;
        const width = Math.floor(plot.clientWidth);
        const height = Math.floor(plot.clientHeight);
        if (!hasValues) return;
        if (width < 120 || height < 100) {
          if (!message) status.textContent = 'Enlarge this panel to see the chart.';
          status.hidden = false;
          readout.hidden = true;
          return;
        }
        const series: uPlot.Series[] = [{}];
        const seriesIndexes = new Map(legendSeries.map((item, index) => [diagnosticComponentKey(item), index]));
        for (const item of visible) {
          const index = seriesIndexes.get(diagnosticComponentKey(item))!;
          const color = colors[index % colors.length];
          const points = isolatedDiagnosticPoints(item.values, logarithmic);
          series.push({
            label: item.label, stroke: color, width: 2, spanGaps: !logarithmic,
            points: { show: points.length > 0, filter: points.length ? points : null, fill: color },
            ...(index >= colors.length ? { dash: [6, 4] } : {}),
          });
        }
        // Use original sample locations for hover, rather than the nulls added to
        // the plotted log data to keep nonpositive values from bridging a gap.
        const sampleIndexes = linearData.series.map(item => diagnosticSampleIndexes(item.values));
        chart = new uPlot({
          width, height,
          scales: { x: { time: false }, y: logarithmic ? { distr: 3 } : {} },
          axes: [
            { label: 't', labelSize: 20, stroke: css('--diagnostic-axis'), grid: { stroke: css('--diagnostic-grid'), width: 0.5 }, size: 36,
              values: (_plot, splits, _axis, _space, increment) => formatDiagnosticTicks(splits, increment) },
            { label: group.title, labelSize: 20, stroke: css('--diagnostic-axis'), grid: { stroke: css('--diagnostic-grid'), width: 0.5 }, size: 58 },
          ],
          legend: { show: false },
          cursor: {
            drag: { setScale: false },
            dataIdx: (_current, seriesIndex, closestIndex, time) => seriesIndex === 0 ? closestIndex :
              nearestDiagnosticSampleIndex(data.times, sampleIndexes[seriesIndex - 1], time),
          },
          series,
          hooks: {
            setCursor: [current => {
              if (current.cursor.idx == null || current.cursor.left == null) {
                readout.textContent = 'Hover over the chart to inspect the nearest sample.';
                return;
              }
              const time = current.posToVal(current.cursor.left, 'x');
              readout.textContent = [
                `t = ${formatNumber(time)}`,
                ...linearData.series.map((item, seriesIndex) => {
                  const index = nearestDiagnosticSampleIndex(data.times, sampleIndexes[seriesIndex], time);
                  if (index === null) return `${item.label}: —`;
                  const value = item.values[index];
                  const omitted = logarithmic && typeof value === 'number' && value <= 0 ? ' (omitted)' : '';
                  return `${item.label}: ${formatNumber(value)}${omitted} (t = ${formatNumber(data.times[index])})`;
                }),
              ].join('  ·  ');
            }],
            draw: [current => {
              const time = state.timeline.current?.time;
              if (time === undefined || !Number.isFinite(time)) return;
              const x = current.valToPos(time, 'x', true);
              const { left, top, width: w, height: h } = current.bbox;
              if (x < left || x > left + w) return;
              current.ctx.save();
              current.ctx.strokeStyle = css('--timeline-regrid');
              current.ctx.lineWidth = Math.max(1, window.devicePixelRatio);
              current.ctx.setLineDash([5 * window.devicePixelRatio, 4 * window.devicePixelRatio]);
              current.ctx.beginPath();
              current.ctx.moveTo(x, top);
              current.ctx.lineTo(x, top + h);
              current.ctx.stroke();
              current.ctx.restore();
            }],
          },
        }, [data.times, ...visible.map(item => item.values)] as uPlot.AlignedData, plot);
        const plotted = chart;
        plotted.over.addEventListener('click', event => {
          const position = event.clientX - plotted.over.getBoundingClientRect().left;
          const time = plotted.posToVal(position, 'x');
          state.navigate(timeline => { timeline.playing = false; timeline.seekTime(time); });
        });
      }

      scale.addEventListener('change', () => {
        settings.scale = scale.value as DiagnosticScale;
        draw();
      });
      showAll.addEventListener('click', () => {
        if (!settings.hasHidden()) return;
        settings.showAll();
        draw();
      });

      async function fetchDiagnostics(): Promise<void> {
        refreshTimer = undefined;
        if (disposed || loading || state.connection !== 'connected' || !wantedSignature || wantedSignature === requestedSignature) return;
        loading = true;
        const signature = wantedSignature;
        try {
          const response = await client.getData({ what: 'diagnostics', names: [...group.names] });
          if (disposed) return;
          histories = Array.isArray(response.series) ? response.series as unknown as DiagnosticSeries[] : [];
          loadError = null;
          requestedSignature = signature;
          draw();
        } catch (error) {
          if (!disposed) {
            loadError = error instanceof Error ? error.message : 'Could not load diagnostics';
            requestedSignature = signature;
            draw();
          }
        } finally {
          loading = false;
          if (!disposed && wantedSignature !== requestedSignature) schedule();
        }
      }

      function schedule(): void {
        if (refreshTimer === undefined && !loading) refreshTimer = window.setTimeout(() => { void fetchDiagnostics(); }, 150);
      }
      function updateWantedSignature(): void {
        wantedSignature = state.connection === 'connected'
          ? JSON.stringify([state.timeline.frames.at(-1)?.id ?? '', state.step, group.names]) : '';
        if (wantedSignature && wantedSignature !== requestedSignature) schedule();
      }
      const unsubscribe = state.subscribe(() => {
        const current = state.timeline.current?.time ?? null;
        if (current !== cursorTime) { cursorTime = current; chart?.redraw(false); }
        updateWantedSignature();
      });
      const resize = new ResizeObserver(() => {
        if (chart && plot.clientWidth >= 120 && plot.clientHeight >= 100) {
          chart.setSize({ width: Math.floor(plot.clientWidth), height: Math.floor(plot.clientHeight) });
        } else draw();
      });
      resize.observe(plot);
      const theme = new MutationObserver(draw);
      theme.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
      updateWantedSignature();
      draw();
      return { element, dispose() {
        if (disposed) return;
        disposed = true;
        unsubscribe(); resize.disconnect(); theme.disconnect();
        if (refreshTimer !== undefined) window.clearTimeout(refreshTimer);
        chart?.destroy();
        configurations.deactivate(panelId);
      } };
    },
  };
}
