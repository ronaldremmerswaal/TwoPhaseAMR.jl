import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { alignDiagnostics, type DiagnosticGroup, type DiagnosticSeries } from './diagnostic-data';
import type { PanelDefinition } from './panels';

/** One dockable chart for exactly one core diagnostic group. */
export function diagnosticPanel(group: DiagnosticGroup, id: string): PanelDefinition {
  return {
    id, title: group.title,
    create({ state, client }) {
      const element = document.createElement('section');
      element.className = 'diagnostic-panel';
      element.dataset.testid = 'diagnostic-panel';
      element.dataset.dimension = group.dimension ? JSON.stringify(group.dimension) : '';
      element.setAttribute('aria-label', group.title);
      const legend = document.createElement('div');
      legend.className = 'diagnostic-legend';
      const plot = document.createElement('div');
      plot.className = 'diagnostic-plot';
      const status = document.createElement('div');
      status.className = 'diagnostic-status';
      status.textContent = 'Waiting for diagnostics';
      element.append(legend, plot, status);

      let chart: uPlot | null = null;
      let histories: DiagnosticSeries[] = [];
      let disposed = false;
      let loading = false;
      let refreshTimer: number | undefined;
      let requestedSignature = '';
      let wantedSignature = '';
      let cursorId: string | number | null = null;

      function css(name: string): string {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      }

      function draw(): void {
        if (disposed) return;
        const width = Math.floor(plot.clientWidth);
        const height = Math.floor(plot.clientHeight);
        if (width < 60 || height < 50) return;
        const data = alignDiagnostics(histories, group.scale === 'log');
        chart?.destroy();
        chart = null;
        legend.replaceChildren();
        if (!data.times.length || !data.series.some(series => series.values.some(value => value !== null))) {
          status.textContent = group.scale === 'log' && data.times.length ? 'No positive diagnostic values' : 'Waiting for diagnostics';
          status.hidden = false;
          return;
        }
        status.hidden = true;
        const colors = [css('--diagnostic-series-1'), css('--diagnostic-series-2'), css('--diagnostic-series-3')];
        const series: uPlot.Series[] = [{}];
        data.series.forEach((item, index) => {
          const color = colors[index % colors.length];
          const key = document.createElement('span');
          key.className = 'diagnostic-series';
          key.style.color = color;
          key.textContent = `━ ${item.label}`;
          legend.append(key);
          series.push({ label: item.label, stroke: color, width: 2, spanGaps: true, points: { show: false }, ...(index >= colors.length ? { dash: [6, 4] } : {}) });
        });
        chart = new uPlot({
          width, height,
          scales: { x: { time: false }, y: group.scale === 'log' ? { distr: 3 } : {} },
          axes: [
            { stroke: css('--diagnostic-axis'), grid: { stroke: css('--diagnostic-grid'), width: 0.5 }, size: 36 },
            { stroke: css('--diagnostic-axis'), grid: { stroke: css('--diagnostic-grid'), width: 0.5 }, size: 58 },
          ],
          legend: { show: false },
          cursor: { drag: { setScale: false } },
          series,
          hooks: { draw: [current => {
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
          }] },
        }, [data.times, ...data.series.map(item => item.values)] as uPlot.AlignedData, plot);
        chart.over.addEventListener('click', event => {
          if (!chart) return;
          const position = event.clientX - chart.over.getBoundingClientRect().left;
          const time = chart.posToVal(position, 'x');
          state.navigate(timeline => { timeline.playing = false; timeline.seekTime(time); });
        });
      }

      async function fetchDiagnostics(): Promise<void> {
        refreshTimer = undefined;
        if (disposed || loading || state.connection !== 'connected' || !wantedSignature || wantedSignature === requestedSignature) return;
        loading = true;
        const signature = wantedSignature;
        try {
          const response = await client.getData({ what: 'diagnostics', names: group.names });
          if (disposed) return;
          histories = Array.isArray(response.series) ? response.series as unknown as DiagnosticSeries[] : [];
          requestedSignature = signature;
          draw();
        } catch (error) {
          if (!disposed) {
            status.hidden = false;
            status.textContent = error instanceof Error ? error.message : 'Could not load diagnostics';
            requestedSignature = signature;
          }
        } finally {
          loading = false;
          if (!disposed && wantedSignature !== requestedSignature) schedule();
        }
      }

      function schedule(): void {
        if (refreshTimer === undefined && !loading) refreshTimer = window.setTimeout(() => { void fetchDiagnostics(); }, 150);
      }
      const unsubscribe = state.subscribe(() => {
        const current = state.timeline.current?.id ?? null;
        if (current !== cursorId) { cursorId = current; chart?.redraw(false); }
        const head = state.timeline.frames.at(-1)?.id ?? '';
        wantedSignature = state.connection === 'connected' ? `${head}:${state.step ?? ''}` : '';
        if (wantedSignature && wantedSignature !== requestedSignature) schedule();
      });
      const resize = new ResizeObserver(() => {
        if (chart && plot.clientWidth >= 60 && plot.clientHeight >= 50) {
          chart.setSize({ width: Math.floor(plot.clientWidth), height: Math.floor(plot.clientHeight) });
        } else draw();
      });
      resize.observe(plot);
      const theme = new MutationObserver(draw);
      theme.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
      wantedSignature = state.connection === 'connected' ? `${state.timeline.frames.at(-1)?.id ?? ''}:${state.step ?? ''}` : '';
      if (wantedSignature) schedule();
      return { element, dispose() {
        disposed = true;
        unsubscribe(); resize.disconnect(); theme.disconnect();
        if (refreshTimer !== undefined) window.clearTimeout(refreshTimer);
        chart?.destroy();
      } };
    },
  };
}
