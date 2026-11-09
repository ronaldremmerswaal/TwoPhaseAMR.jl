import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { chromium, expect as baseExpect } from '@playwright/test';

const expect = baseExpect.configure({ timeout: 30_000 });
const smokeDir = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(smokeDir, '../..');
const fixture = join(smokeDir, 'fixture.jl');
const temp = await mkdtemp(join(tmpdir(), 'twophase-studio-smoke-'));
const output = join(temp, 'stored.gif');
const child = spawn(process.env.JULIA || 'julia', [
  '--startup-file=no', '--heap-size-hint=3G', '-t', '1', `--project=${packageRoot}`, fixture,
], {
  cwd: packageRoot, stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env, OPENBLAS_NUM_THREADS: '1', JULIA_NUM_PRECOMPILE_TASKS: '1' },
});
let stderr = '';
child.stderr.setEncoding('utf8');
child.stderr.on('data', text => { stderr += text; });
child.stdin.on('error', () => {}); // A failed fixture may already have closed its pipe.

function fixtureURL() {
  return new Promise((resolveURL, reject) => {
    const lines = createInterface({ input: child.stdout });
    const timer = setTimeout(() => finish(new Error('Julia fixture startup timed out')), 600_000);
    const exited = (code, signal) => finish(new Error(`Julia fixture exited before readiness (${code ?? signal})`));
    const failed = error => finish(error);
    function finish(error, value) {
      clearTimeout(timer);
      child.off('exit', exited);
      child.off('error', failed);
      lines.close();
      child.stdout.resume(); // Keep draining incidental output after the URL.
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
    assert.equal(child.exitCode, 0, `Julia fixture exited unexpectedly: ${stderr}`);
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
  assert.ok(clean, 'Julia fixture failed to shut down');
  assert.equal(child.exitCode, 0, `Julia fixture failed during teardown: ${stderr}`);
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
  const contextOptions = { viewport: { width: 1960, height: 1120 } };
  context = await browser.newContext(contextOptions);
  const page = await context.newPage();
  page.setDefaultTimeout(30_000);
  const pageErrors = [];
  const sent = [];
  const assetRequests = [];
  const socketURLs = [];
  page.on('request', request => {
    const url = new URL(request.url());
    if (url.origin === new URL(address).origin && ['/studio.js', '/studio.css'].includes(url.pathname)) {
      assetRequests.push(url);
    }
  });
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('websocket', socket => {
    socketURLs.push(new URL(socket.url()));
    socket.on('framesent', ({ payload }) => {
      if (typeof payload === 'string') {
        try { sent.push(JSON.parse(payload)); } catch { /* protocol assertions below */ }
      }
    });
  });
  const response = await page.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  assert.equal(response?.status(), 200);
  await expect(page.getByTestId('connection-status')).toHaveText('connected');
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED');
  await expect(page.getByTestId('timeline-mode')).toHaveText('LIVE');
  await expect(page.getByTestId('timeline-current')).not.toHaveText('No frame');
  const token = new URL(address).searchParams.get('token');
  assert.ok(token && ['/studio.js', '/studio.css'].every(path =>
    assetRequests.some(url => url.pathname === path && url.searchParams.get('token') === token)),
  'authenticated asset requests must preserve the page token');
  assert.ok(socketURLs.some(url => url.pathname === '/ws' && url.searchParams.get('token') === token),
    'the WebSocket must preserve the page token');

  const theme = page.getByTestId('theme-select');
  await expect(theme).toHaveCount(1);
  await expect(page.getByTestId('theme-toggle')).toHaveCount(0);
  await expect(theme).toHaveValue('system');
  // Measure the rendered status text against its effective background, not
  // just its token: a transparent badge can inherit a different surface.
  async function checkRunStatusContrast(themeName, label) {
    await expect(page.getByTestId('run-state')).toHaveText(label);
    await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(themeName);
    const ratio = await page.evaluate(() => {
      const element = document.querySelector('[data-testid="run-state"]');
      const rgba = value => {
        const channels = value.match(/[\d.]+/g);
        if (!channels || channels.length < 3) throw new Error(`Unrecognized computed color: ${value}`);
        return [Number(channels[0]) / 255, Number(channels[1]) / 255, Number(channels[2]) / 255,
          channels.length > 3 ? Number(channels[3]) : 1];
      };
      const over = (front, back) => {
        const alpha = front[3] + back[3] * (1 - front[3]);
        if (!alpha) return [0, 0, 0, 0];
        return [
          ...front.slice(0, 3).map((channel, i) =>
            (channel * front[3] + back[i] * back[3] * (1 - front[3])) / alpha),
          alpha,
        ];
      };
      let background = [0, 0, 0, 0];
      for (let node = element; node && background[3] < 1; node = node.parentElement) {
        background = over(background, rgba(getComputedStyle(node).backgroundColor));
      }
      background = over(background, [1, 1, 1, 1]);
      const foreground = over(rgba(getComputedStyle(element).color), background);
      const luminance = color => {
        const linear = color.slice(0, 3).map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
      };
      const a = luminance(foreground), b = luminance(background);
      return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    });
    assert.ok(ratio >= 4.5, `${themeName}: ${label} status contrast ${ratio} is below WCAG AA`);
  }
  await page.emulateMedia({ colorScheme: 'light' });
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');

  // Advance the actual paused solver through the WebSocket controls. The
  // fixture activates its AMR criterion after ten steps; every fifth step is
  // a scheduled keyframe. Pause before taking either screenshot.
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.runs[0]?.name))
    .toBe('Adaptive flow');
  await expect(page.getByTestId('run-name')).toHaveText('Adaptive flow');
  const liveLegend = page.getByTestId('diagnostic-toggle').first();
  await expect(liveLegend).toBeVisible();
  await liveLegend.click();
  await expect(liveLegend).toHaveAttribute('aria-pressed', 'false');
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('RUNNING');
  await checkRunStatusContrast('light', 'RUNNING');
  await page.emulateMedia({ colorScheme: 'dark' });
  await checkRunStatusContrast('dark', 'RUNNING');
  await page.emulateMedia({ colorScheme: 'light' });
  await page.waitForFunction(() => {
    const state = window.twoPhaseAMRStudio.state;
    const frames = state.timeline.frames;
    return state.step >= 25 &&
      frames.filter(frame => frame.kind === 'keyframe').length >= 2 &&
      new Set(frames.map(frame => frame.topology)).size >= 2;
  }, null, { timeout: 180_000 });
  await expect(liveLegend).toHaveAttribute('aria-pressed', 'false');
  await liveLegend.click();
  await expect(liveLegend).toHaveAttribute('aria-pressed', 'true');
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED');
  await expect(page.getByTestId('timeline-mode')).toHaveText('LIVE');
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.timeline.current?.step))
    .toBeGreaterThanOrEqual(25);
  const pausedStep = await page.evaluate(() => window.twoPhaseAMRStudio.state.step);
  assert.ok(pausedStep >= 25);
  await expect(page.getByTestId('run-clock')).toContainText(`step ${pausedStep}`);
  const markers = page.getByTestId('timeline-markers');
  await expect.poll(() => markers.locator('.timeline-marker.keyframe').count())
    .toBeGreaterThanOrEqual(2);
  await expect(markers.locator('.timeline-marker.keyframe').nth(0)).toBeVisible();
  await expect(markers.locator('.timeline-marker.keyframe').nth(1)).toBeVisible();
  const keyframeBoxes = await markers.locator('.timeline-marker.keyframe').evaluateAll(nodes =>
    nodes.map(node => ({ x: node.getBoundingClientRect().x, width: node.getBoundingClientRect().width })));
  assert.ok(keyframeBoxes.some(box => Math.abs(box.x - keyframeBoxes[0].x) > Math.max(box.width, keyframeBoxes[0].width)),
    'keyframe dots must be visibly separated on the timeline');
  await expect(markers.locator('.timeline-marker.regrid').first()).toBeVisible();

  // A tick alone is not proof of AMR: compare the stored topology snapshots
  // and the solver's step log at the first transition in the frame index.
  const regrid = await page.evaluate(async () => {
    const api = window.twoPhaseAMRStudio;
    const frames = api.state.timeline.frames;
    const index = frames.findIndex((frame, i) => i > 0 && frame.topology !== frames[i - 1].topology);
    if (index < 1) throw new Error('No topology transition in recorded frames');
    const [before, after, steplog, beforeVelocity, afterVelocity] = await Promise.all([
      api.client.getData({ what: 'topology', topology: frames[index - 1].topology }),
      api.client.getData({ what: 'topology', topology: frames[index].topology }),
      api.client.getData({ what: 'steplog' }),
      api.client.getField(frames[index - 1].id, 'velocity_x', 'f32'),
      api.client.getField(frames[index].id, 'velocity_x', 'f32'),
    ]);
    return {
      before, after, step: frames[index].step, time: frames[index].time, steplog,
      beforeVelocity: Array.from(beforeVelocity.values), afterVelocity: Array.from(afterVelocity.values),
    };
  });
  assert.equal(regrid.before.blocks.length, 1);
  assert.equal(regrid.after.blocks.length, 4);
  assert.ok(regrid.after.blocks.every(block => block.level === 1));
  assert.notEqual(regrid.before.generation, regrid.after.generation);
  assert.ok(regrid.afterVelocity.length > regrid.beforeVelocity.length);
  assert.ok(regrid.afterVelocity.every(Number.isFinite) && new Set(regrid.afterVelocity).size > 1,
    'the refined frame must contain transferred, nonconstant velocity data');
  assert.ok(regrid.steplog.step.some((step, i) => step === regrid.step && regrid.steplog.topology_changed[i]),
    'the recorded topology change must match a solver regrid');
  // The fixture's timeline spans [0, 1]; the percentage round-trip costs
  // only a few Float64 ulps.
  const tickTime = await markers.locator('.timeline-marker.regrid').first()
    .evaluate(tick => Number.parseFloat(tick.style.left) / 100);
  assert.ok(Math.abs(tickTime - regrid.time) <= 16 * Number.EPSILON,
    'the visible regrid tick must mark the recorded topology transition');

  // System preference is the default; the single selector persists per
  // browser. Check computed token contrast in each theme and keep both
  // screenshots of the advanced paused run for review.
  await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--text').trim())).not.toBe('');
  const recordingControls = page.getByTestId('recording-controls');
  await expect.poll(() => recordingControls.evaluate(details => details.open)).toBe(false);
  await expect(theme).toHaveValue('system');
  await page.emulateMedia({ colorScheme: 'dark' });
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
  await page.emulateMedia({ colorScheme: 'light' });
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');
  const shots = '/tmp/studio-shots';
  await mkdir(shots, { recursive: true });
  async function checkTheme(name) {
    await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(name);
    await checkRunStatusContrast(name, 'PAUSED');
    // Inspect the actual visible controls, then collapse the drawer so it
    // does not obscure the workspace in either screenshot.
    await recordingControls.locator('summary').click();
    await expect(page.getByTestId('recording-path')).toBeVisible();
    const contrasts = await page.evaluate(() => {
      const pairs = [
        ...['--text', '--muted', '--record', '--error'].flatMap(fg =>
          ['--surface', '--surface-raised'].map(bg => [fg, bg])),
        ['--field-foreground', '--field-background'], ['--colorbar-text', '--field-background'],
        ['--diagnostic-text', '--diagnostic-background'], ['--diagnostic-axis', '--diagnostic-background'],
      ];
      const probe = document.createElement('span');
      document.body.appendChild(probe);
      const linear = value => {
        const numbers = value.match(/[\d.]+/g).slice(0, 3).map(Number);
        return numbers.map(number => {
          const c = number / 255;
          return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
        });
      };
      const rgb = variable => {
        if (!getComputedStyle(document.documentElement).getPropertyValue(variable).trim()) throw new Error(`Missing theme token ${variable}`);
        probe.style.color = `var(${variable})`;
        return linear(getComputedStyle(probe).color);
      };
      const luminance = color => 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
      const result = pairs.map(([fg, bg]) => {
        const a = luminance(rgb(fg)), b = luminance(rgb(bg));
        return { fg, bg, ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) };
      });
      for (const [selector, pseudo] of [
        ['[data-testid="recording-path"]', '::placeholder'],
        ['[data-testid="recording-stop"]', null],
        ['.dv-tab', null],
      ]) {
        const element = document.querySelector(selector);
        const a = luminance(linear(getComputedStyle(element, pseudo).color));
        const b = luminance(linear(getComputedStyle(element).backgroundColor));
        result.push({ fg: selector, bg: 'control background', ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) });
      }
      probe.remove();
      return result;
    });
    for (const { fg, bg, ratio } of contrasts) {
      assert.ok(ratio >= 4.5, `${name}: ${fg} on ${bg} contrast ${ratio} is below WCAG AA`);
    }
    await recordingControls.locator('summary').click();
    await expect.poll(() => recordingControls.evaluate(details => details.open)).toBe(false);
  }
  await theme.selectOption('light');
  await checkTheme('light');
  await theme.selectOption('dark');
  await expect(theme).toHaveValue('dark');
  await checkTheme('dark');
  // Start a real stored recording before the screenshots, so the REC
  // indicator reflects the server-owned session rather than a mock.
  await recordingControls.locator('summary').click();
  await page.getByTestId('recording-source').selectOption('stored');
  await page.getByTestId('recording-field').selectOption('velocity_x');
  await page.getByTestId('recording-path').fill(output);
  await page.getByTestId('recording-start').click();
  await expect(page.getByTestId('recording-status')).toContainText('REC');
  await expect(page.getByTestId('recording-status')).toBeVisible();
  await recordingControls.locator('summary').click();
  await expect.poll(() => recordingControls.evaluate(details => details.open)).toBe(false);
  await theme.selectOption('light');
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');
  await page.screenshot({ path: join(shots, 'light.png'), fullPage: true });
  await theme.selectOption('dark');
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
  await page.screenshot({ path: join(shots, 'dark.png'), fullPage: true });
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 60_000 });
  await expect(page.getByTestId('connection-status')).toHaveText('connected');
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED');
  await expect(page.getByTestId('recording-status')).toContainText('REC');
  await expect(theme).toHaveValue('dark');
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
  await theme.selectOption('system');
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');

  // Exercise the real browser decoder and server on a captured simulation field.
  const fields = await page.evaluate(async () => {
    const api = window.twoPhaseAMRStudio;
    const id = api.state.timeline.frames[0].id;
    const f32 = await api.client.getField(id, 'velocity_x', 'f32');
    const u16 = await api.client.getField(id, 'velocity_x', 'u16');
    return { f32: Array.from(f32.values), u16: Array.from(u16.values), scale: u16.scale };
  });
  assert.ok(fields.f32.length > 1 && new Set(fields.f32).size > 1);
  assert.equal(fields.f32.length, fields.u16.length);
  // Round-to-nearest quantization: half a bin plus Float32 header rounding.
  const errorBound = fields.scale / 2 + 8 * 2 ** -23 * Math.max(...fields.f32.map(Math.abs));
  fields.f32.forEach((value, index) => assert.ok(Math.abs(value - fields.u16[index]) <= errorBound));
  const metadata = await page.evaluate(async () => {
    const api = window.twoPhaseAMRStudio;
    const frame = api.state.timeline.frames[0];
    const [topology, planes, diagnostics, steplog] = await Promise.all([
      api.client.getData({ what: 'topology', topology: frame.topology }),
      api.client.getData({ what: 'planes', frame: frame.id }),
      api.client.getData({ what: 'diagnostics' }),
      api.client.getData({ what: 'steplog' }),
    ]);
    return { topology, planes, diagnostics, steplog };
  });
  assert.ok(Array.isArray(metadata.topology.blocks) && metadata.topology.blocks.length > 0);
  assert.ok(Array.isArray(metadata.planes.phases));
  assert.ok(Array.isArray(metadata.diagnostics.series));
  assert.ok(Array.isArray(metadata.steplog.step));
  const rejected = await page.evaluate(async () => {
    const api = window.twoPhaseAMRStudio;
    try {
      await api.client.getField(api.state.timeline.frames[0].id, 'no_such_field', 'f32');
      return false;
    } catch (error) {
      return error instanceof Error && error.message.length > 0;
    }
  });
  assert.ok(rejected, 'a failed field request must reject its browser promise');
  await expect(page.locator('#error[role="alert"]')).toBeVisible();
  await page.getByRole('button', { name: 'Dismiss error' }).click();
  await expect(page.getByTestId('connection-status')).toHaveText('connected');

  // Stored recording remains a server-owned session while two clients look at
  // different times. Stopping it below performs the actual animation export.
  await expect.poll(() => recordingControls.evaluate(details => details.open)).toBe(false);
  await recordingControls.locator('summary').click();
  await expect(page.getByTestId('recording-status')).toContainText('REC');
  const second = await context.newPage();
  await second.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await expect(second.getByTestId('connection-status')).toHaveText('connected');
  await expect(second.getByTestId('recording-status')).toContainText('REC');

  const beforeRecordingRun = await page.evaluate(() => window.twoPhaseAMRStudio.state.step);
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('RUNNING');
  await page.waitForFunction(start => {
    const state = window.twoPhaseAMRStudio.state;
    return state.step >= start + 5 && state.timeline.frames.some(frame => frame.step >= start + 5);
  }, beforeRecordingRun, { timeout: 120_000 });
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED');
  const selected = () => page.evaluate(() => window.twoPhaseAMRStudio.state.timeline.selectedId);
  const latest = await selected();
  const controlsBefore = sent.length;
  await page.getByTestId('timeline-prev').click();
  await expect(page.getByTestId('timeline-mode')).toHaveText('HISTORY');
  assert.notEqual(await selected(), latest);
  await page.getByTestId('timeline-next').click();
  assert.equal(await selected(), latest);
  await page.getByTestId('timeline-prev-keyframe').click();
  const selectedKind = await page.evaluate(() => window.twoPhaseAMRStudio.state.timeline.current?.kind);
  assert.ok(['initial', 'keyframe', 'manual', 'pause', 'final'].includes(selectedKind));
  await page.getByTestId('timeline-next-keyframe').click();
  await page.getByTestId('timeline-scrub').evaluate(slider => {
    slider.value = '0';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  });
  const first = await selected();
  await page.getByTestId('timeline-play').click();
  await expect(page.getByTestId('timeline-play')).toHaveAttribute('aria-label', 'Pause playback');
  await expect.poll(selected, { timeout: 10_000 }).not.toBe(first);
  if (await page.getByTestId('timeline-play').getAttribute('aria-label') === 'Pause playback') {
    await page.getByTestId('timeline-play').click();
  }
  await page.getByTestId('timeline-follow').click();
  await expect(page.getByTestId('timeline-mode')).toHaveText('LIVE');
  assert.ok(!sent.slice(controlsBefore).some(message => ['control', 'recording'].includes(message.type)),
    'timeline navigation sent a solver or recording command');
  await expect(second.getByTestId('timeline-mode')).toHaveText('LIVE');
  await expect(page.getByTestId('recording-status')).toContainText('REC');

  // Leave the first browser in history while the solver makes new frames; a
  // second browser continues following live and the recording stays active.
  await page.getByTestId('timeline-prev').click();
  const pinned = await selected();
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('RUNNING');
  await expect(page.getByTestId('timeline-new-count')).toHaveText(/\+[1-9]\d* new/);
  assert.equal(await selected(), pinned);
  await expect(page.getByTestId('timeline-mode')).toHaveText('HISTORY');
  await expect(second.getByTestId('timeline-mode')).toHaveText('LIVE');
  await expect(second.getByTestId('recording-status')).toContainText('REC');
  await expect.poll(() => second.evaluate(() => window.twoPhaseAMRStudio.state.timeline.selectedId))
    .not.toBe(pinned);
  await page.getByTestId('timeline-follow').click();
  await expect(page.getByTestId('timeline-mode')).toHaveText('LIVE');

  const beforeManual = await page.evaluate(() => window.twoPhaseAMRStudio.state.step);
  await page.getByTestId('control-keyframe').click();
  await page.waitForFunction(step => window.twoPhaseAMRStudio.state.step >= step + 2, beforeManual,
    { timeout: 45_000 });
  await page.getByTitle('Refresh the stored frame index').click();
  await page.waitForFunction(() => window.twoPhaseAMRStudio.state.timeline.frames.some(frame => frame.kind === 'manual'),
    null, { timeout: 30_000 });
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED');
  await expect(page.getByTestId('recording-status')).toContainText('REC');
  await page.getByTestId('control-stop').click();
  await expect(page.getByTestId('run-state')).toHaveText('STOPPED');
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.runState)).toBe('completed');
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.stopped)).toBe(true);
  await expect(page.getByTestId('recording-status')).toContainText('REC');
  await page.getByTestId('recording-stop').click();
  await expect(page.locator('#recording-detail')).toContainText('completed', { timeout: 180_000 });
  await expect(page.getByTestId('recording-state')).not.toContainText('REC');
  const gif = await readFile(output);
  assert.ok(gif.length > 6 && ['GIF87a', 'GIF89a'].includes(gif.subarray(0, 6).toString('ascii')),
    'stored recording lacks a GIF header');
  assert.equal(gif.at(-1), 0x3b, 'stored recording lacks a GIF trailer');
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.recording.encoded)).toBeGreaterThan(0);
  assert.deepEqual(pageErrors, []);
  await second.close();
} catch (error) {
  primaryError = error;
} finally {
  for (const cleanup of [
    () => context?.close(),
    () => browser?.close(),
    () => shutdownFixture(),
    () => rm(temp, { recursive: true, force: true }),
  ]) {
    try { await cleanup(); } catch (error) { primaryError ??= error; }
  }
}
if (primaryError && stderr) primaryError.message += `\nJulia stderr:\n${stderr.replace(/token=[^&\s]+/g, 'token=[redacted]')}`;
if (primaryError) throw primaryError;
console.log('Studio browser smoke passed (real flow, recording and WebSocket)');
