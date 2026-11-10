// Run from lib/TwoPhaseAMRStudio/frontend after building Studio's assets.
// Uses the real, initially paused live AMR fixture; no synthetic state updates.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { chromium, expect as baseExpect } from '@playwright/test';

const expect = baseExpect.configure({ timeout: 60_000 });
const smokeDir = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(smokeDir, '../..');
const child = spawn(process.env.JULIA || 'julia', [
  '--startup-file=no', '--heap-size-hint=3G', '-t', '1,1', `--project=${packageRoot}`,
  join(smokeDir, 'live_amr.jl'),
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
    const timer = setTimeout(() => finish(new Error('Live AMR fixture startup timed out')), 600_000);
    const exited = (code, signal) => finish(new Error(`Live AMR fixture exited before readiness (${code ?? signal})`));
    const failed = error => finish(error);
    function finish(error, value) {
      clearTimeout(timer);
      child.off('exit', exited); child.off('error', failed);
      lines.close(); child.stdout.resume();
      if (error) reject(error); else resolveURL(value);
    }
    child.once('exit', exited); child.once('error', failed);
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

// Track the actual elements, not just their selectors: replacing a control
// between animation frames must fail even if its replacement has the same box.
async function sampleHeaders(page, theme, size) {
  const result = await page.evaluate(({ duration, theme }) => new Promise((resolveSample, reject) => {
    const state = window.twoPhaseAMRStudio.state;
    const selector = [
      '.studio-header :is(button, input, select, textarea, summary, a, [role="button"], [tabindex])',
      '.field-toolbar :is(button, input, select, textarea, summary, a, [role="button"], [tabindex])',
      '.diagnostic-toolbar :is(button, input, select, textarea, summary, a, [role="button"], [tabindex])',
      '.diagnostic-legend button', '#dock .dv-tab', '#dock .dv-tab .dv-default-tab-action',
      '#dock .dv-tabs-and-actions-container :is(button, [role="button"], [tabindex])',
    ].join(', ');
    const visible = node => {
      const box = node.getBoundingClientRect();
      const style = getComputedStyle(node);
      return node.isConnected && node.getClientRects().length > 0 && box.width > 0 && box.height > 0 &&
        style.display !== 'none' && style.visibility === 'visible';
    };
    const controls = [...document.querySelectorAll(selector)].filter(visible);
    const boxes = new Map(controls.map(node => {
      const rect = node.getBoundingClientRect();
      return [node, { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom }];
    }));
    const describe = node => node.dataset.testid || node.getAttribute('data-tab-panel-id') ||
      node.closest('.dv-tab')?.getAttribute('data-tab-panel-id') || node.className || node.tagName;
    const errors = [];
    const fail = message => { if (errors.length < 10 && !errors.includes(message)) errors.push(message); };
    const counts = {
      global: controls.filter(node => node.closest('.studio-header')).length,
      fields: controls.filter(node => node.closest('.field-toolbar')).length,
      diagnostic: controls.filter(node => node.closest('.diagnostic-toolbar, .diagnostic-legend')).length,
      tabs: controls.filter(node => node.matches('#dock .dv-tab')).length,
    };
    const started = performance.now();
    const firstStep = state.step, firstTime = state.time;
    const frameIds = new Set();
    const steps = new Set();
    const progress = [];
    let lastProgressStep = firstStep;
    function observeState() {
      if (state.connection !== 'connected' || state.error || state.runState !== 'running' || state.timeline.mode !== 'live') {
        fail(`live state changed: ${state.connection}/${state.runState}/${state.timeline.mode}/${state.error}`);
      }
      if (state.timeline.current?.id != null) frameIds.add(String(state.timeline.current.id));
      if (Number.isFinite(state.step)) {
        steps.add(state.step);
        if (state.step !== lastProgressStep) {
          progress.push({ elapsed: performance.now() - started, step: state.step });
          lastProgressStep = state.step;
        }
      }
    }
    const unsubscribe = state.subscribe(observeState);
    function observeMutations(records) {
      for (const record of records) {
        if (record.type === 'attributes') {
          // Catch a control being hidden and shown between animation frames,
          // including one that happened to be hidden in the initial sample.
          if (record.target.matches(selector) &&
              (record.oldValue === null || !record.target.hasAttribute('hidden'))) {
            fail(`header control visibility changed: ${describe(record.target)}`);
          }
          if (record.oldValue === null) {
            for (const node of controls) if (record.target === node || record.target.contains(node)) {
              fail(`control was hidden: ${describe(node)}`);
            }
          }
        } else {
          for (const removed of record.removedNodes) {
            if (removed.nodeType !== Node.ELEMENT_NODE) continue;
            for (const node of controls) if (removed === node || removed.contains(node)) {
              fail(`control was removed or remounted: ${describe(node)}`);
            }
          }
        }
      }
    }
    const observer = new MutationObserver(observeMutations);
    observer.observe(document.getElementById('app'), {
      subtree: true, childList: true, attributes: true, attributeFilter: ['hidden'], attributeOldValue: true,
    });
    let previous = started, maxGap = 0, samples = 0, maxMovement = 0;
    let animation;
    const timeout = setTimeout(() => {
      cancelAnimationFrame(animation); observer.disconnect(); unsubscribe();
      reject(new Error('Animation-frame sampling did not finish'));
    }, 20_000);
    function sample() {
      const now = performance.now();
      maxGap = Math.max(maxGap, now - previous); previous = now; samples++;
      observeState();
      if (document.documentElement.dataset.theme !== theme) fail('theme changed during sampling');
      const current = new Set([...document.querySelectorAll(selector)].filter(visible));
      if (current.size !== controls.length || [...current].some(node => !boxes.has(node))) {
        fail(`visible header control set changed: ${controls.length} to ${current.size}`);
      }
      for (const node of controls) {
        if (!current.has(node)) { fail(`control disappeared: ${describe(node)}`); continue; }
        const rect = node.getBoundingClientRect(), initial = boxes.get(node);
        const movement = Math.max(...['left', 'top', 'right', 'bottom'].map(edge => Math.abs(rect[edge] - initial[edge])));
        maxMovement = Math.max(maxMovement, movement);
        if (movement > 0.5) fail(`control moved ${movement.toFixed(3)}px: ${describe(node)}`);
      }
      if (now - started < duration) { animation = requestAnimationFrame(sample); return; }
      clearTimeout(timeout); observeMutations(observer.takeRecords()); observer.disconnect(); unsubscribe();
      resolveSample({ counts, errors, elapsed: now - started, samples, maxGap, maxMovement,
        firstStep, lastStep: state.step, firstTime, lastTime: state.time,
        steps: [...steps], frames: [...frameIds], progress });
    }
    sample();
  }), { duration: 3200, theme });
  const detail = `${theme} ${size}: ${JSON.stringify(result)}`;
  assert.ok(result.counts.global >= 6 && result.counts.fields >= 16 && result.counts.diagnostic >= 3 && result.counts.tabs >= 3,
    `did not sample all expected header control classes: ${detail}`);
  assert.ok(result.elapsed >= 3000 && result.samples >= 10 && result.maxGap <= 750,
    `animation samples did not cover the full three seconds: ${detail}`);
  assert.ok(Number.isFinite(result.firstStep) && result.lastStep > result.firstStep && result.steps.length >= 3 &&
    Number.isFinite(result.firstTime) && result.lastTime > result.firstTime && result.frames.length >= 3 &&
    result.progress.some(item => item.elapsed < result.elapsed / 2 && item.step > result.firstStep) &&
    result.progress.some(item => item.elapsed >= result.elapsed / 2 && item.step > result.firstStep),
  `the real simulation and frame stream did not advance throughout sampling: ${detail}`);
  assert.deepEqual(result.errors, [], `header controls were not stable: ${detail}`);
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
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  const response = await page.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  assert.equal(response?.status(), 200, 'Studio page did not return HTTP 200');
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio?.state.connection)).toBe('connected');
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio?.state.runState)).toBe('paused');
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio?.state.timeline.mode)).toBe('live');
  await expect(page.getByTestId('field-panel')).toHaveCount(2);
  // Readiness is meaningful while paused, but not while new frames keep arriving.
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews().filter(view => view.ready).length),
    { timeout: 180_000 }).toBe(2);
  await page.getByTestId('control-pause').click();
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.runState), { timeout: 180_000 }).toBe('running');
  await page.waitForFunction(() => {
    const state = window.twoPhaseAMRStudio.state;
    return state.runState === 'running' && state.step >= 10 &&
      document.querySelectorAll('.diagnostic-legend button').length > 0;
  }, null, { timeout: 180_000 });

  const results = [];
  for (const [size, viewport] of [
    ['wide', { width: 1960, height: 1120 }],
    ['narrow', { width: 1200, height: 1000 }],
  ]) {
    await page.setViewportSize(viewport);
    await expect(page.getByTestId('field-panel')).toHaveCount(2);
    // Responsive reflow recreates panels. Wait for their diagnostic controls,
    // not for field `ready`, which may legitimately stay false during live input.
    await page.evaluate(() => new Promise(resolveFrame => requestAnimationFrame(() => requestAnimationFrame(resolveFrame))));
    await expect.poll(() => page.evaluate(() => {
      const panels = [...document.querySelectorAll('[data-testid="diagnostic-panel"]')]
        .filter(node => node.getClientRects().length > 0);
      return panels.length > 0 && panels.every(node => node.querySelector('.diagnostic-legend button'));
    }), { timeout: 180_000, message: 'diagnostic controls must be present after reflow' }).toBe(true);
    for (const theme of ['light', 'dark']) {
      await page.getByTestId('theme-select').selectOption(theme);
      await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(theme);
      await page.mouse.move(1, 1);
      await page.evaluate(() => new Promise(resolveFrame => requestAnimationFrame(() => requestAnimationFrame(resolveFrame))));
      results.push({ theme, size, ...await sampleHeaders(page, theme, size) });
    }
  }
  assert.deepEqual(pageErrors, [], `Studio page errors: ${pageErrors.join('; ')}`);
  console.log(`Studio header stability passed: ${results.map(({ theme, size, samples, maxMovement }) =>
    `${theme}/${size} ${samples} samples, ${maxMovement.toFixed(3)}px`).join('; ')}`);
} catch (error) {
  primaryError = error;
} finally {
  for (const cleanup of [() => context?.close(), () => browser?.close(), () => shutdownFixture()]) {
    try { await cleanup(); } catch (error) { primaryError ??= error; }
  }
}
if (primaryError && stderr) primaryError.message += `\nJulia stderr:\n${stderr.replace(/token=[^&\s]+/g, 'token=[redacted]')}`;
if (primaryError) throw primaryError;
