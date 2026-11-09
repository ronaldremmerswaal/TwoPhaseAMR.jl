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
  context = await browser.newContext();
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

  // System preference is the default; explicit selection and the status-bar
  // toggle persist per browser. Check actual computed token contrast in each
  // theme and keep both screenshots for review.
  await expect.poll(() => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--text').trim())).not.toBe('');
  const theme = page.getByTestId('theme-select');
  await expect(theme).toHaveValue('system');
  await context.emulateMedia({ colorScheme: 'dark' });
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
  await context.emulateMedia({ colorScheme: 'light' });
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');
  const shots = '/tmp/studio-shots';
  await mkdir(shots, { recursive: true });
  async function checkTheme(name) {
    await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(name);
    const contrasts = await page.evaluate(() => {
      const pairs = [
        ...['--text', '--muted', '--record', '--error'].flatMap(fg =>
          ['--surface', '--surface-raised'].map(bg => [fg, bg])),
        ['--field-foreground', '--field-background'], ['--colorbar-text', '--field-background'],
        ['--diagnostic-text', '--diagnostic-background'], ['--diagnostic-axis', '--diagnostic-background'],
      ];
      const probe = document.createElement('span');
      document.body.appendChild(probe);
      const rgb = variable => {
        if (!getComputedStyle(document.documentElement).getPropertyValue(variable).trim()) throw new Error(`Missing theme token ${variable}`);
        probe.style.color = `var(${variable})`;
        const numbers = getComputedStyle(probe).color.match(/[\d.]+/g).slice(0, 3).map(Number);
        return numbers.map(value => {
          const c = value / 255;
          return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
        });
      };
      const luminance = color => 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
      const result = pairs.map(([fg, bg]) => {
        const a = luminance(rgb(fg)), b = luminance(rgb(bg));
        return { fg, bg, ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) };
      });
      probe.remove();
      return result;
    });
    for (const { fg, bg, ratio } of contrasts) {
      assert.ok(ratio >= 4.5, `${name}: ${fg} on ${bg} contrast ${ratio} is below WCAG AA`);
    }
    await page.screenshot({ path: join(shots, `${name}.png`), fullPage: true });
  }
  await theme.selectOption('light');
  await checkTheme('light');
  await page.getByTestId('theme-toggle').click();
  await expect(theme).toHaveValue('dark');
  await checkTheme('dark');
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 60_000 });
  await expect(page.getByTestId('connection-status')).toHaveText('connected');
  await expect(theme).toHaveValue('dark');
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
  await theme.selectOption('system');
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');

  // Exercise the real browser decoder and server on a captured simulation
  // field. No renderer is assumed or installed by this shell-only smoke.
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
  await expect(page.getByRole('alert')).toBeVisible();
  await page.getByRole('button', { name: 'Dismiss error' }).click();
  await expect(page.getByTestId('connection-status')).toHaveText('connected');

  // Stored recording remains a server-owned session while two clients look at
  // different times. Stopping it below performs the actual animation export.
  await page.getByTestId('recording-source').selectOption('stored');
  await page.getByTestId('recording-field').selectOption('velocity_x');
  await page.getByTestId('recording-path').fill(output);
  await page.getByTestId('recording-start').click();
  await expect(page.getByTestId('recording-status')).toContainText('REC');
  const second = await context.newPage();
  await second.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await expect(second.getByTestId('connection-status')).toHaveText('connected');
  await expect(second.getByTestId('recording-status')).toContainText('REC');

  await page.getByTestId('control-pause').click(); // Resume the initial pause.
  await expect(page.getByTestId('run-state')).toHaveText('RUNNING');
  await page.waitForFunction(() => window.twoPhaseAMRStudio.state.timeline.frames.length >= 6, null, { timeout: 45_000 });
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
  await expect(page.getByTestId('run-state')).toHaveText('COMPLETED');
  await expect(page.getByTestId('recording-status')).toContainText('REC');
  await page.getByTestId('recording-stop').click();
  await expect(page.getByTestId('recording-status')).toContainText('completed', { timeout: 180_000 });
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
