// Run after building Studio's assets: npm run test:continue
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { chromium, expect as baseExpect } from '@playwright/test';

const expect = baseExpect.configure({ timeout: 180_000 });
const smokeDir = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(smokeDir, '../..');
const shots = '/tmp/studio-shots';
await mkdir(shots, { recursive: true });
const child = spawn(process.env.JULIA || 'julia', [
  '--startup-file=no', '--heap-size-hint=3G', '-t', '2,1', `--project=${packageRoot}`,
  join(smokeDir, 'continue.jl'),
], {
  cwd: packageRoot, stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env, OPENBLAS_NUM_THREADS: '1', JULIA_NUM_PRECOMPILE_TASKS: '1' },
});
let stderr = '';
child.stderr.setEncoding('utf8');
child.stderr.on('data', text => { stderr += text; });
child.stdin.on('error', () => {});

function fixtureURL() {
  return new Promise((resolveURL, reject) => {
    const lines = createInterface({ input: child.stdout });
    const timer = setTimeout(() => finish(new Error('Continuation fixture startup timed out')), 1_200_000);
    const exited = (code, signal) => finish(new Error(`Continuation fixture exited before readiness (${code ?? signal})`));
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
    assert.equal(child.exitCode, 0, `Continuation fixture exited unexpectedly: ${stderr}`);
    return;
  }
  const exited = new Promise(resolveExit => child.once('exit', resolveExit));
  child.stdin.end('quit\n');
  let timer;
  const timeout = new Promise(resolveTimeout => {
    timer = setTimeout(() => { child.kill('SIGKILL'); resolveTimeout(false); }, 90_000);
  });
  const clean = await Promise.race([exited.then(() => true), timeout]);
  clearTimeout(timer);
  assert.ok(clean, 'Continuation fixture did not stop after quit');
  assert.equal(child.exitCode, 0, `Continuation fixture failed during teardown: ${stderr}`);
}

function hasSamplesAfter(messages, time) {
  return messages.some(message => Array.isArray(message.series) && message.series.some(series =>
    series.name === 'amplitude' && Array.isArray(series.times) &&
    series.times.some(sample => typeof sample === 'number' && sample > time + 1e-12)));
}

let browser;
let context;
let otherContext;
let primaryError;
try {
  const address = await fixtureURL();
  browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  context = await browser.newContext({ viewport: { width: 1960, height: 1120 } });
  const page = await context.newPage();
  const sent = [];
  const diagnosticResponses = [];
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('websocket', socket => {
    socket.on('framesent', ({ payload }) => {
      if (typeof payload !== 'string') return;
      try { sent.push(JSON.parse(payload)); } catch { /* Not a JSON control message. */ }
    });
    socket.on('framereceived', ({ payload }) => {
      if (typeof payload !== 'string') return;
      try {
        const message = JSON.parse(payload);
        if (message.type === 'diagnostics') diagnosticResponses.push(message);
      } catch { /* Not a JSON response. */ }
    });
  });
  await page.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio?.state.connection)).toBe('connected');
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.runState), { timeout: 300_000 }).toBe('paused');
  await expect(page.getByTestId('control-continue')).toBeDisabled();
  await page.getByTestId('control-pause').click();
  await expect.poll(() => page.evaluate(() => {
    const state = window.twoPhaseAMRStudio.state;
    return state.runState === 'completed' && state.stopped === false;
  }), { timeout: 300_000 }).toBe(true);
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.timeline.frames.length)).toBeGreaterThan(2);
  const initial = await page.evaluate(() => {
    const state = window.twoPhaseAMRStudio.state;
    return { time: state.time, finalTime: state.finalTime, suggested: state.suggestedFinalTime, resumable: state.resumable };
  });
  assert.equal(initial.resumable, true);
  assert.ok(Number.isFinite(initial.finalTime) && initial.finalTime > 0);
  assert.ok(Math.abs(initial.time - initial.finalTime) < 1e-10);
  const target = 2 * initial.finalTime;
  assert.ok(Math.abs(initial.suggested - target) < 1e-10, 'server should suggest the original interval after the previous end');
  await expect.poll(() => page.getByTestId('continue-target').evaluate(input => input.valueAsNumber)).toBeCloseTo(initial.suggested, 12);
  await expect(page.getByTestId('control-continue')).toBeEnabled();
  await expect.poll(() => hasSamplesAfter(diagnosticResponses, initial.finalTime - 1e-10)).toBe(true);

  // Reject empty and non-later targets without sending a control command.
  const input = page.getByTestId('continue-target');
  await input.fill('');
  await expect(page.getByTestId('control-continue')).toBeDisabled();
  await expect(input).toHaveAttribute('aria-invalid', 'true');
  await input.press('Enter');
  await input.fill(String(initial.time));
  await expect(page.getByTestId('control-continue')).toBeDisabled();
  assert.equal(sent.filter(message => message.type === 'control' && message.action === 'continue').length, 0);
  await input.fill(String(target));
  await expect(page.getByTestId('control-continue')).toBeEnabled();

  // Keep the same historical playhead, field configuration and diagnostic
  // controls across continuation rather than rebuilding the dock layout.
  await page.locator('[data-tab-panel-id="diagnostic-0"]').click();
  const diagnostic = page.locator('[data-testid="diagnostic-panel"][data-panel-id="diagnostic-0"]');
  await expect(diagnostic.locator('.uplot')).toBeVisible();
  await diagnostic.getByTestId('diagnostic-scale').selectOption('log');
  await expect.poll(() => diagnostic.getByTestId('diagnostic-toggle').count()).toBeGreaterThan(1);
  await diagnostic.getByTestId('diagnostic-toggle').first().click();
  await expect(diagnostic.getByTestId('diagnostic-toggle').first()).toHaveAttribute('aria-pressed', 'false');
  await page.getByTestId('timeline-prev').click();
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.timeline.mode)).toBe('history');
  await expect.poll(() => page.evaluate(() => {
    const api = window.twoPhaseAMRStudio;
    return api.getFieldViews().every(view => view.ready && view.frame === api.state.timeline.selectedId);
  })).toBe(true);
  await page.evaluate(() => {
    window.__continueSnapshot = () => {
      const api = window.twoPhaseAMRStudio;
      return {
        selectedId: api.state.timeline.selectedId,
        panelIds: [...new Set([...document.querySelectorAll('[data-tab-panel-id]')].map(node => node.getAttribute('data-tab-panel-id')))].sort(),
        views: api.getFieldViews().map(({ panelId, field, mode, pinnedFrame, camera, fitMode, linked, colormap, colorPolicy, overlays }) =>
          ({ panelId, field, mode, pinnedFrame, camera, fitMode, linked, colormap, colorPolicy, overlays })),
        diagnostics: [...document.querySelectorAll('[data-testid="diagnostic-panel"]')].map(panel => ({
          id: panel.getAttribute('data-panel-id'),
          scale: panel.querySelector('[data-testid="diagnostic-scale"]')?.value,
          components: [...panel.querySelectorAll('[data-testid="diagnostic-toggle"]')].map(button =>
            [button.getAttribute('data-name'), button.getAttribute('data-component'), button.getAttribute('aria-pressed')]),
        })).sort((a, b) => a.id.localeCompare(b.id)),
      };
    };
  });
  const before = await page.evaluate(() => window.__continueSnapshot());
  assert.ok(before.views.length > 0);
  assert.ok(before.panelIds.includes('diagnostic-0') && before.diagnostics.length > 0);
  for (const theme of ['light', 'dark']) {
    await page.getByTestId('theme-select').selectOption(theme);
    await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(theme);
    await page.screenshot({ path: join(shots, `fb1-${theme}.png`) });
  }

  // A second client must see the shared target and continued store as well.
  otherContext = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const other = await otherContext.newPage();
  other.on('pageerror', error => pageErrors.push(error.message));
  await other.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await expect.poll(() => other.evaluate(() => window.twoPhaseAMRStudio?.state.connection)).toBe('connected');
  await expect.poll(() => other.evaluate(() => {
    const state = window.twoPhaseAMRStudio.state;
    return state.runState === 'completed' && state.stopped === false;
  })).toBe(true);
  await expect.poll(() => other.evaluate(() => window.twoPhaseAMRStudio.state.finalTime)).toBe(initial.finalTime);

  await page.evaluate(() => {
    const state = window.twoPhaseAMRStudio.state;
    window.__continueStates = [];
    state.subscribe(() => {
      if (window.__continueStates.at(-1) !== state.runState) window.__continueStates.push(state.runState);
    });
  });
  await page.getByTestId('control-continue').click();
  await expect.poll(() => page.evaluate(() => window.__continueStates.includes('running')), { timeout: 300_000 }).toBe(true);
  await expect.poll(() => other.evaluate(() => window.twoPhaseAMRStudio.state.finalTime), { timeout: 300_000 }).toBe(target);
  for (const clientPage of [page, other]) {
    await expect.poll(() => clientPage.evaluate(() => {
      const state = window.twoPhaseAMRStudio.state;
      return state.runState === 'completed' && state.stopped === false;
    }), { timeout: 300_000 }).toBe(true);
  }
  await expect.poll(async () => Number(await input.inputValue())).toBeCloseTo(target + initial.finalTime, 12);
  await expect(page.getByTestId('control-continue')).toBeEnabled();
  for (const clientPage of [page, other]) {
    await expect.poll(async () => Number(await clientPage.locator('#timeline-end').textContent())).toBeCloseTo(target, 6);
  }
  for (const clientPage of [page, other]) {
    await expect.poll(() => clientPage.evaluate(oldEnd => window.twoPhaseAMRStudio.state.timeline.frames.some(frame => frame.time > oldEnd + 1e-12), initial.finalTime)).toBe(true);
  }
  await expect.poll(() => hasSamplesAfter(diagnosticResponses, initial.finalTime)).toBe(true);
  await expect(diagnostic.getByTestId('diagnostic-scale')).toHaveValue('log');
  await expect(diagnostic.locator('.uplot')).toBeVisible();
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.timeline.mode)).toBe('history');
  const after = await page.evaluate(() => window.__continueSnapshot());
  assert.deepEqual(after, before);
  // Clicking the right edge of the rendered chart must navigate to a new
  // sample, proving that the plot itself refreshed beyond the old interval.
  await expect.poll(async () => {
    const overlay = diagnostic.locator('.u-over');
    const box = await overlay.boundingBox();
    if (!box || box.width < 4 || box.height < 4) return -1;
    await overlay.click({ position: { x: box.width - 2, y: box.height / 2 } });
    return page.evaluate(() => window.twoPhaseAMRStudio.state.timeline.current?.time ?? -1);
  }).toBeGreaterThan(initial.finalTime);
  assert.deepEqual(sent.filter(message => message.type === 'control' && message.action === 'continue'), [
    { type: 'control', action: 'continue', final_time: target },
  ]);
  assert.deepEqual(pageErrors, []);
} catch (error) {
  primaryError = error;
} finally {
  try { await otherContext?.close(); } catch (error) { primaryError ??= error; }
  try { await context?.close(); } catch (error) { primaryError ??= error; }
  try { await browser?.close(); } catch (error) { primaryError ??= error; }
  try { await shutdownFixture(); } catch (error) { primaryError ??= error; }
}
if (primaryError && stderr) primaryError.message += `\nJulia stderr:\n${stderr.replace(/token=[^&\s]+/g, 'token=[redacted]')}`;
if (primaryError) throw primaryError;
console.log('Studio continuation browser smoke passed (capillary example, shared state, frames, diagnostics, light/dark)');
