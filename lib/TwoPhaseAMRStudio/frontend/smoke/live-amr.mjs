// Run from lib/TwoPhaseAMRStudio/frontend: node smoke/live-amr.mjs
// Requires the installed frontend/Playwright dependencies, Studio's built assets,
// and smoke/live_amr.jl: a paused, retained, two-velocity adaptive live run.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { chromium, expect as baseExpect } from '@playwright/test';

const expect = baseExpect.configure({ timeout: 60_000 });
const smokeDir = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(smokeDir, '../..');
const fixture = join(smokeDir, 'live_amr.jl');
const fields = ['velocity_vorticity', 'velocity_magnitude'];
// Even one delayed request exceeds the fixture's nominal 0.2 s step interval;
// loading a new two-phase frame needs several field/plane/topology requests.
const latencyMs = 300;
const child = spawn(process.env.JULIA || 'julia', [
  '--startup-file=no', '--heap-size-hint=3G', '-t', '1,1', `--project=${packageRoot}`, fixture,
], {
  cwd: packageRoot, stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env, OPENBLAS_NUM_THREADS: '1', JULIA_NUM_PRECOMPILE_TASKS: '1' },
});
let stderr = '';
child.stderr.setEncoding('utf8');
child.stderr.on('data', text => { stderr += text; });
child.stdin.on('error', () => {}); // An already-failed fixture may have closed its pipe.

function fixtureURL() {
  return new Promise((resolveURL, reject) => {
    const lines = createInterface({ input: child.stdout });
    const timer = setTimeout(() => finish(new Error('Live AMR fixture startup timed out')), 600_000);
    const exited = (code, signal) => finish(new Error(`Live AMR fixture exited before readiness (${code ?? signal})`));
    const failed = error => finish(error);
    function finish(error, value) {
      clearTimeout(timer);
      child.off('exit', exited);
      child.off('error', failed);
      lines.close();
      child.stdout.resume();
      if (error) reject(error); else resolveURL(value);
    }
    child.once('exit', exited);
    child.once('error', failed);
    lines.on('line', line => {
      if (line.startsWith('STUDIO_URL ')) finish(null, line.slice('STUDIO_URL '.length));
    });
  });
}

async function shutdownFixture() {
  if (child.pid === undefined) return;
  if (child.exitCode !== null || child.signalCode !== null) {
    assert.equal(child.exitCode, 0, `Live AMR fixture exited unexpectedly: ${stderr}`);
    return;
  }
  const exited = new Promise(resolveExit => child.once('exit', resolveExit));
  child.stdin.end('quit\n');
  let timer;
  const timeout = new Promise(resolveTimeout => {
    timer = setTimeout(() => { child.kill('SIGKILL'); resolveTimeout(false); }, 30_000);
  });
  const clean = await Promise.race([exited.then(() => true), timeout]);
  clearTimeout(timer);
  assert.ok(clean, 'Live AMR fixture did not stop after quit');
  assert.equal(child.exitCode, 0, `Live AMR fixture failed during teardown: ${stderr}`);
}

// Install only test-side instrumentation around the public client. Delaying
// responses, rather than WebSocket messages, keeps unsolicited live previews
// flowing while the real panel requests remain outstanding.
async function installMonitor(page, initial) {
  await page.evaluate(({ initial, delay, fields }) => {
    const api = window.twoPhaseAMRStudio;
    const { state, client } = api;
    const originalField = client.getField;
    const originalData = client.getData;
    const seen = new Set(state.timeline.frames.map(frame => frame.id));
    const monitor = {
      previews: [], requests: [], delivered: [], violations: [],
      samples: { light: [[], []], dark: [[], []] },
    };
    window.__liveAmrSmoke = monitor;
    const unsubscribe = state.subscribe(() => {
      for (const frame of state.timeline.frames) {
        if (seen.has(frame.id)) continue;
        seen.add(frame.id);
        if (frame.kind === 'preview' && frame.step > initial.step) {
          monitor.previews.push({ id: frame.id, step: frame.step, topology: frame.topology });
        }
      }
    });
    function delayed(kind, field, target, request) {
      const start = performance.now();
      const existingFrames = new Set(state.timeline.frames.map(frame => frame.id));
      const startState = state.runState;
      const theme = document.documentElement.dataset.theme;
      return Promise.resolve().then(request).then(value => new Promise(resolveValue => {
        setTimeout(() => {
          monitor.requests.push({
            kind, field, target, theme, startState, endState: state.runState,
            elapsed: performance.now() - start,
            previewsDuring: state.timeline.frames.filter(frame =>
              frame.kind === 'preview' && !existingFrames.has(frame.id)).length,
          });
          resolveValue(value);
        }, delay);
      }), error => {
        monitor.requests.push({ kind, field, target, theme, startState, error: String(error) });
        throw error;
      });
    }
    // Independently compute the registered robust range from both actual f32
    // phase responses. Matching a displayed range to these samples identifies
    // the drawn frame without treating the live playhead as the drawn frame.
    const phaseValues = new Map();
    client.getField = (...args) => delayed('field', args[1], args[0], () => originalField.apply(client, args)).then(result => {
      const [frameId, field, , phase] = args;
      if (!fields.includes(field) || !['liquid', 'gas'].includes(phase)) return result;
      const id = JSON.stringify([frameId, field]);
      const pair = phaseValues.get(id) ?? {};
      pair[phase] = result.values;
      phaseValues.set(id, pair);
      if (pair.liquid && pair.gas) {
        const frame = state.timeline.frames.find(item => item.id === frameId);
        const metadata = state.runs[0].fields.find(item => item.name === field);
        const finite = [...pair.liquid, ...pair.gas].filter(Number.isFinite);
        if (frame && metadata && finite.length) {
          const symmetric = metadata.symmetric === true || ['balance', 'coolwarm'].includes(metadata.colormap);
          const sorted = (symmetric ? finite.map(Math.abs) : finite).sort((a, b) => a - b);
          const percentile = sorted[Math.ceil(sorted.length * 0.995) - 1];
          const range = symmetric ? [-percentile, percentile] : [sorted[0], percentile];
          monitor.delivered.push({ field, frame: frame.id, step: frame.step, kind: frame.kind, topology: frame.topology, range });
        }
        phaseValues.delete(id);
      }
      return result;
    });
    client.getData = query => ['topology', 'planes'].includes(query.what)
      ? delayed(query.what, null, query.what === 'topology' ? query.topology : query.frame, () => originalData.call(client, query))
      : originalData.call(client, query);

    const different = (a, b) => a.some((value, index) =>
      Math.abs(value - b[index]) > 1e-5 * Math.max(1e-9, Math.abs(value), Math.abs(b[index])));
    let animation;
    function sample() {
      animation = requestAnimationFrame(sample);
      if (state.runState !== 'running') return;
      const theme = document.documentElement.dataset.theme;
      if (theme !== 'light' && theme !== 'dark') return;
      const head = state.timeline.current;
      const views = api.getFieldViews();
      const panels = document.querySelectorAll('[data-testid="field-panel"]');
      const fail = message => {
        if (!monitor.violations.length) {
          monitor.violations.push({ theme, step: head?.step, message });
        }
      };
      if (state.connection !== 'connected' || state.error) {
        fail(`Studio connection/error: ${state.connection}, ${state.error}`);
      }
      if (state.timeline.mode !== 'live') fail(`timeline left LIVE mode: ${state.timeline.mode}`);
      if (views.length !== 2 || panels.length !== 2) {
        fail(`expected two field views/panels, got ${views.length}/${panels.length}`);
        return;
      }
      views.forEach((view, index) => {
        const name = fields[index];
        if (view.field !== name) fail(`panel ${index} changed field to ${view.field}, expected ${name}`);
        if (view.colorPolicy !== 'robust') fail(`${name}: expected robust automatic colour policy, got ${view.colorPolicy}`);
        const range = view.colorRange;
        if (!Array.isArray(range) || range.length !== 2 || !range.every(Number.isFinite) || !(range[0] < range[1])) {
          fail(`${name}: drawn field has unavailable/nonfinite/degenerate range ${JSON.stringify(range)}`);
          return;
        }
        const bar = panels[index].querySelector('[data-testid="field-colorbar"]');
        const labels = [bar?.firstElementChild?.textContent?.trim(), bar?.lastElementChild?.textContent?.trim()];
        const numbers = labels.map(label => label ? Number(label) : NaN);
        if (!numbers.every(Number.isFinite) || numbers.some((value, i) =>
          Math.abs(value - range[i]) > 1e-5 * Math.max(1e-9, Math.abs(range[i])))) {
          fail(`${name}: visible colourbar ${JSON.stringify(labels)} does not show finite endpoints for ${JSON.stringify(range)}`);
        }
        const status = panels[index].querySelector('[data-testid="field-status"]');
        const loading = [...(status?.querySelectorAll('[role="status"]') ?? [])].some(element =>
          element.textContent.trim() === 'Loading field…' && element.getClientRects().length > 0);
        if (loading) fail(`${name}: Loading field… remains visible after a field has been drawn`);
        const delivered = monitor.delivered.findLast(item => item.field === name &&
          item.range.every((value, i) =>
            Math.abs(value - range[i]) <= 1e-6 * Math.max(1e-9, Math.abs(value), Math.abs(range[i]))));
        const history = monitor.samples[theme][index];
        const previous = history.at(-1);
        if (head && delivered && (!previous || (delivered.step > previous.dataStep && head.step > previous.step &&
            monitor.previews.length > previous.previews && different(range, previous.range)))) {
          history.push({ step: head.step, frame: head.id, dataFrame: delivered.frame, dataStep: delivered.step,
            kind: delivered.kind, topology: delivered.topology, previews: monitor.previews.length, range: [...range] });
        }
      });
    }
    animation = requestAnimationFrame(sample);
    monitor.stop = () => {
      cancelAnimationFrame(animation);
      unsubscribe();
      client.getField = originalField;
      client.getData = originalData;
    };
  }, { initial, delay: latencyMs, fields });
}

async function waitForLiveEvidence(page, theme, initial) {
  let timeoutError;
  try {
    await page.waitForFunction(({ theme, initial, fields, latencyMs }) => {
      const monitor = window.__liveAmrSmoke;
      const state = window.twoPhaseAMRStudio.state;
      const transition = state.timeline.frames.find(frame =>
        frame.step > initial.step && frame.topology !== initial.topology);
      const ranges = transition && monitor.samples[theme].every(history =>
        history.filter(sample => sample.kind === 'preview' && sample.dataStep >= transition.step &&
          sample.topology !== initial.topology).length >= 3);
      const overlapping = fields.every(field => monitor.requests.some(request =>
        request.kind === 'field' && request.field === field && request.theme === theme &&
        request.startState === 'running' && request.endState === 'running' &&
        request.elapsed >= latencyMs - 5 && request.previewsDuring > 0));
      const delayedLoad = transition && monitor.samples[theme].every((history, index) => history.some(sample =>
        sample.kind === 'preview' && sample.dataStep >= transition.step && sample.topology !== initial.topology &&
        ['topology', 'planes', 'field'].every(kind => monitor.requests.some(request =>
          request.kind === kind && (kind !== 'field' || request.field === fields[index]) &&
          String(request.target) === String(kind === 'topology' ? sample.topology : sample.dataFrame) &&
          request.elapsed >= latencyMs - 5))));
      return monitor.violations.length || ['completed', 'failed', 'stopping'].includes(state.runState) ||
        (transition && ranges && overlapping && delayedLoad && monitor.previews.length >= 6);
    }, { theme, initial, fields, latencyMs }, { timeout: 180_000 });
  } catch (error) { timeoutError = error; }
  const evidence = await page.evaluate(({ theme, initial }) => {
    const monitor = window.__liveAmrSmoke;
    const state = window.twoPhaseAMRStudio.state;
    return {
      runState: state.runState, connection: state.connection, error: state.error,
      transition: state.timeline.frames.find(frame => frame.step > initial.step && frame.topology !== initial.topology),
      previewCount: monitor.previews.length,
      ranges: monitor.samples[theme],
      requests: monitor.requests.slice(-20), violations: monitor.violations,
    };
  }, { theme, initial });
  assert.equal(evidence.violations.length, 0,
    `${theme}: invalid live field/colourbar state: ${JSON.stringify(evidence.violations)}`);
  assert.ok(!timeoutError,
    `${theme}: timed out waiting for three changing post-regrid preview ranges per field, delayed topology/planes/field loads, and preview arrivals during field requests: ${JSON.stringify(evidence)}`);
  assert.equal(evidence.runState, 'running', `${theme}: solver stopped before live evidence was collected: ${JSON.stringify(evidence)}`);
  return evidence;
}

// Inspect actual Playwright screenshots, not colours inferred from metadata.
// Project the physical domain into each plot and exclude the colourbar overlay.
// Classify only pixels matching that panel's actual CSS colormap, excluding
// background/transparent cells; demand high coverage so a blank plot cannot pass.
// Allow a modest fraction of genuine minima/maxima and percentile outliers.
async function screenshotColors(page, theme) {
  await page.evaluate(() => new Promise(resolveFrame => requestAnimationFrame(() => requestAnimationFrame(resolveFrame))));
  const geometry = await page.evaluate(() => {
    const api = window.twoPhaseAMRStudio;
    const domain = api.state.runs[0].domain;
    const panels = [...document.querySelectorAll('[data-testid="field-panel"]')];
    const rect = node => {
      const box = node.getBoundingClientRect();
      return { left: box.left, top: box.top, right: box.right, bottom: box.bottom, width: box.width, height: box.height };
    };
    return {
      runState: api.state.runState,
      views: api.getFieldViews().map((view, index) => {
        const plot = panels[index].querySelector('[data-testid="field-plot"]');
        const bar = panels[index].querySelector('[data-testid="field-colorbar"]');
        return { field: view.field, worldRect: view.worldRect, plot: rect(plot), bar: rect(bar),
          gradient: getComputedStyle(bar.children[1]).backgroundImage };
      }),
      origin: domain.origin, extent: domain.extent,
    };
  });
  assert.equal(geometry.runState, 'running', `${theme}: screenshot must be taken while the solver is running`);
  const image = await page.screenshot({ scale: 'css' });
  const result = await page.evaluate(async ({ image, geometry }) => {
    const bytes = Uint8Array.from(atob(image), character => character.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    return geometry.views.map(view => {
      const stops = [...view.gradient.matchAll(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/g)]
        .map(match => match.slice(1).map(Number));
      if (stops.length < 2 || !view.worldRect) throw new Error(`${view.field}: missing palette or world rectangle`);
      const palette = new Set();
      const packed = (r, g, b) => (r << 16) | (g << 8) | b;
      // Small tolerance covers GPU interpolation and screenshot quantization.
      for (const [r, g, b] of stops) {
        for (let dr = -2; dr <= 2; dr++) for (let dg = -2; dg <= 2; dg++) for (let db = -2; db <= 2; db++) {
          const rr = r + dr, gg = g + dg, bb = b + db;
          if (rr >= 0 && rr <= 255 && gg >= 0 && gg <= 255 && bb >= 0 && bb <= 255) palette.add(packed(rr, gg, bb));
        }
      }
      const [xmin, xmax, ymin, ymax] = view.worldRect;
      const [ox, oy] = geometry.origin, [ex, ey] = geometry.extent;
      const { plot, bar } = view;
      const left = Math.max(0, plot.left, plot.left + (ox - xmin) / (xmax - xmin) * plot.width);
      const right = Math.min(canvas.width, plot.right, plot.left + (ox + ex - xmin) / (xmax - xmin) * plot.width);
      const top = Math.max(0, plot.top, plot.top + (ymax - oy - ey) / (ymax - ymin) * plot.height);
      const bottom = Math.min(canvas.height, plot.bottom, plot.top + (ymax - oy) / (ymax - ymin) * plot.height);
      const low = stops[0], high = stops.at(-1);
      const near = (rgb, stop) => rgb.every((channel, index) => Math.abs(channel - stop[index]) <= 2);
      const colors = new Set();
      let domainPixels = 0, fieldPixels = 0, lowPixels = 0, highPixels = 0;
      for (let y = Math.ceil(top + 2); y < bottom - 2; y += 2) {
        for (let x = Math.ceil(left + 2); x < right - 2; x += 2) {
          if (x >= bar.left && x < bar.right && y >= bar.top && y < bar.bottom) continue;
          domainPixels++;
          const offset = (y * canvas.width + x) * 4;
          const rgb = [pixels[offset], pixels[offset + 1], pixels[offset + 2]];
          const key = packed(...rgb);
          if (!palette.has(key)) continue;
          fieldPixels++;
          colors.add(key);
          if (near(rgb, low)) lowPixels++;
          if (near(rgb, high)) highPixels++;
        }
      }
      return { field: view.field, domainPixels, fieldPixels, lowPixels, highPixels, colors: colors.size,
        coverage: fieldPixels / domainPixels, lowFraction: lowPixels / fieldPixels, highFraction: highPixels / fieldPixels };
    });
  }, { image: image.toString('base64'), geometry });
  assert.equal(await page.evaluate(() => window.twoPhaseAMRStudio.state.runState), 'running',
    `${theme}: solver stopped during screenshot capture`);
  for (const sample of result) {
    assert.ok(sample.domainPixels >= 500 && sample.fieldPixels >= 500 && sample.coverage >= 0.75 && sample.colors >= 5,
      `${theme}: ${sample.field} screenshot does not contain a sufficiently covered, nonconstant rendered field: ${JSON.stringify(sample)}`);
  }
  return result;
}

let browser;
let context;
let primaryError;
try {
  const address = await fixtureURL();
  browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  context = await browser.newContext({ viewport: { width: 1960, height: 1120 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.setDefaultTimeout(60_000);
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  const response = await page.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  assert.equal(response?.status(), 200, 'Studio page did not return HTTP 200');
  await expect(page.getByTestId('connection-status')).toHaveText('connected');
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED');
  await expect(page.getByTestId('timeline-mode')).toHaveText('LIVE');
  await expect(page.getByTestId('field-panel')).toHaveCount(2);
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews().map(view => view.field)),
    { message: 'two-phase defaults must be volume_fraction and velocity_magnitude' })
    .toEqual(['volume_fraction', 'velocity_magnitude']);
  const initial = await page.evaluate(() => {
    const { state } = window.twoPhaseAMRStudio;
    return { ...state.timeline.current };
  });
  assert.ok(initial.id != null && Number.isFinite(initial.step) && initial.topology != null,
    `initial paused frame is missing its id, step, or topology: ${JSON.stringify(initial)}`);
  const registered = await page.evaluate(() => window.twoPhaseAMRStudio.state.runs[0].fields);
  for (const field of fields) {
    const metadata = registered.find(entry => entry.name === field);
    assert.ok(metadata && metadata.interface > 0,
      `${field} must be a default-registered two-velocity field with an interface`);
  }
  await page.getByTestId('theme-select').selectOption('light');
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');
  await page.getByTestId('field-picker-trigger').first().click();
  await page.getByTestId('field-picker').first().selectOption('velocity_vorticity');
  // Remove interface/solid overlays from the image classification; the data,
  // phase selection, renderer, and colourbars are otherwise untouched.
  for (let index = 0; index < 2; index++) {
    await page.getByTestId('field-overlays').nth(index).click();
    await page.getByTestId('field-overlay-interface').nth(index).uncheck();
    await page.getByTestId('field-overlay-solid').nth(index).uncheck();
    await page.getByTestId('field-overlays').nth(index).click();
  }
  await expect.poll(() => page.evaluate(fields => {
    const views = window.twoPhaseAMRStudio.getFieldViews();
    return views.length === 2 && views.every((view, index) => view.field === fields[index] && view.ready &&
      view.worldRect && view.colorPolicy === 'robust' && Array.isArray(view.colorRange) &&
      view.colorRange.every(Number.isFinite) && view.colorRange[0] < view.colorRange[1]);
  }, fields), { timeout: 180_000, message: 'both selected fields must initially draw finite robust ranges at the paused frame' }).toBe(true);
  await installMonitor(page, initial);
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('RUNNING', { timeout: 180_000 });

  const screenshots = [];
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark') {
      await page.getByTestId('theme-select').selectOption(theme);
      await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(theme);
    }
    await waitForLiveEvidence(page, theme, initial);
    screenshots.push({ theme, fields: await screenshotColors(page, theme) });
  }
  // One suitable derived field must stay far below 20% at *each* palette end
  // in both themes. The allowance handles real extrema rather than rejecting
  // any endpoint pixel; comparing both fields avoids assuming one phase's
  // physically valid distribution cannot contain a large extremal plateau.
  const suitable = fields.find(field => screenshots.every(shot => {
    const sample = shot.fields.find(value => value.field === field);
    return sample && sample.lowFraction < 0.15 && sample.highFraction < 0.15;
  }));
  assert.ok(suitable, `neither derived field has unsaturated live colours in both themes (<15% at either endpoint): ${JSON.stringify(screenshots)}`);

  // Only now stop arrivals. A previously drawn frame may legitimately be
  // marked updating during backpressure, but the latest frame must settle.
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED', { timeout: 180_000 });
  const violations = await page.evaluate(() => window.__liveAmrSmoke.violations);
  assert.deepEqual(violations, [], `invalid live field/colourbar state during screenshot capture: ${JSON.stringify(violations)}`);
  await expect.poll(() => page.evaluate(fields => {
    const api = window.twoPhaseAMRStudio;
    const views = api.getFieldViews();
    return views.length === 2 && views.every((view, index) => view.field === fields[index] && view.ready && !view.loading &&
      view.frame === api.state.timeline.selectedId && Array.isArray(view.colorRange) && view.colorRange.every(Number.isFinite));
  }, fields), { timeout: 180_000, message: 'panels failed to settle on the latest frame after live previews stopped' }).toBe(true);
  // Retain the initial frame in the argument rather than relying on a Studio
  // test API or on any synthetic topology identifiers.
  const frames = await page.evaluate(initial => {
    const timeline = window.twoPhaseAMRStudio.state.timeline;
    const ids = new Set(timeline.frames.map(frame => frame.id));
    const previews = window.__liveAmrSmoke.previews;
    return { transition: timeline.frames.find(frame => frame.step > initial.step && frame.topology !== initial.topology),
      current: timeline.current, frameCount: timeline.frames.length, previewCount: previews.length,
      missingPreviews: previews.filter(frame => !ids.has(frame.id)) };
  }, initial);
  assert.ok(frames.transition && frames.transition.step > initial.step && frames.current?.step > frames.transition.step &&
      Number.isFinite(initial.time) && frames.current.time > initial.time,
    `retained live frames do not progress in step and time beyond a post-pause topology transition: ${JSON.stringify(frames)}`);
  assert.ok(frames.previewCount >= 6 && frames.missingPreviews.length === 0,
    `preview=:all did not retain every observed live preview: ${JSON.stringify(frames)}`);
  await expect(page.getByTestId('timeline-markers').locator('.timeline-marker.regrid').first()).toBeVisible();
  await page.evaluate(() => window.__liveAmrSmoke.stop());
  const actual = await page.evaluate(async ({ initial, frames, fields }) => {
    const client = window.twoPhaseAMRStudio.client;
    const [before, after, ...values] = await Promise.all([
      client.getData({ what: 'topology', topology: Number(initial.topology) }),
      client.getData({ what: 'topology', topology: Number(frames.transition.topology) }),
      ...fields.flatMap(field => ['liquid', 'gas'].map(phase => client.getField(frames.current.id, field, 'f32', phase))),
    ]);
    const blocks = topology => topology.blocks.map(block =>
      JSON.stringify([block.level, block.i, block.j, block.origin, block.spacing])).sort();
    const arrays = values.map(result => {
      const finite = Array.from(result.values).filter(Number.isFinite);
      return { count: finite.length, min: Math.min(...finite), max: Math.max(...finite) };
    });
    return { before: blocks(before), after: blocks(after), arrays };
  }, { initial, frames: { transition: frames.transition, current: frames.current }, fields });
  assert.notDeepEqual(actual.before, actual.after, 'topology IDs changed but stored AMR block geometry did not');
  fields.forEach((field, index) => {
    const phases = actual.arrays.slice(index * 2, index * 2 + 2);
    assert.ok(phases.every(phase => phase.count > 0 && Number.isFinite(phase.min) && Number.isFinite(phase.max)) &&
      Math.max(...phases.map(phase => phase.max)) > Math.min(...phases.map(phase => phase.min)),
    `${field}: real liquid/gas field samples must be finite and spatially nonconstant: ${JSON.stringify(phases)}`);
  });
  assert.deepEqual(pageErrors, [], `Studio page errors: ${pageErrors.join('; ')}`);
} catch (error) {
  primaryError = error;
} finally {
  for (const cleanup of [() => context?.close(), () => browser?.close(), () => shutdownFixture()]) {
    try { await cleanup(); } catch (error) { primaryError ??= error; }
  }
}
if (primaryError && stderr) primaryError.message += `\nJulia stderr:\n${stderr.replace(/token=[^&\s]+/g, 'token=[redacted]')}`;
if (primaryError) throw primaryError;
console.log('Studio live AMR browser smoke passed (two-phase frames, regrid, delayed fields, colours, light/dark)');
