import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { chromium, expect as baseExpect } from '@playwright/test';

const expect = baseExpect.configure({ timeout: 30_000 });
const smokeDir = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(smokeDir, '../..');
const fixture = join(smokeDir, 'time_linear.jl');
const temp = await mkdtemp(join(tmpdir(), 'twophase-studio-time-'));
const storedOutput = join(temp, 'stored.gif');
const liveOutput = join(temp, 'live.gif');
const child = spawn(process.env.JULIA || 'julia', [
  '--startup-file=no', '--heap-size-hint=3G', '-t', '1', `--project=${packageRoot}`, fixture,
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
    const timer = setTimeout(() => finish(new Error('Julia fixture startup timed out')), 600_000);
    const exited = (code, signal) => finish(new Error(`Julia fixture exited before readiness (${code ?? signal})`));
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

// Count actual GIF image descriptors; a rendered video must have one frame per
// output time, not one per stored frame.
function gifFrameCount(bytes) {
  assert.ok(bytes.length > 13 && ['GIF87a', 'GIF89a'].includes(bytes.subarray(0, 6).toString('ascii')),
    'not a GIF');
  let offset = 13;
  const globalPacked = bytes[10];
  if (globalPacked & 0x80) offset += 3 * (1 << ((globalPacked & 7) + 1));
  let frames = 0;
  function skipSubBlocks() {
    for (;;) {
      const size = bytes[offset++];
      if (size === 0) return;
      offset += size;
    }
  }
  while (offset < bytes.length) {
    const block = bytes[offset++];
    if (block === 0x3b) break;
    if (block === 0x21) {
      offset++; // extension label
      skipSubBlocks();
    } else if (block === 0x2c) {
      frames++;
      const packed = bytes[offset + 8];
      offset += 9;
      if (packed & 0x80) offset += 3 * (1 << ((packed & 7) + 1));
      offset++; // LZW minimum code size
      skipSubBlocks();
    } else {
      throw new Error(`unexpected GIF block 0x${block.toString(16)} at ${offset - 1}`);
    }
  }
  return frames;
}

function scheduleAttributes(text) {
  return Object.fromEntries(text.split(/\s+/).filter(Boolean).map(pair => pair.split('=')));
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
  context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await context.newPage();
  page.setDefaultTimeout(30_000);
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await expect(page.getByTestId('connection-status')).toHaveText('connected');
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED');

  const frameTimes = await page.evaluate(() => {
    const times = [...new Set(window.twoPhaseAMRStudio.state.timeline.frames.map(frame => frame.time))];
    return times.sort((a, b) => a - b);
  });
  assert.ok(frameTimes.length > 4, 'the fixture must record several frames');
  const gaps = frameTimes.slice(1).map((time, index) => Number((time - frameTimes[index]).toFixed(9)));
  assert.ok(new Set(gaps).size > 1, `recorded frame times must be irregular, got gaps ${gaps}`);

  // Stored export: the dialog reports the same uniform time line the export
  // uses, and the GIF has exactly that many frames.
  await page.getByTestId('open-export').click();
  await page.getByTestId('export-source').selectOption('stored');
  await page.getByTestId('export-path').fill(storedOutput);
  await page.getByTestId('export-frames').selectOption('all');
  await page.locator('[name="width"]').fill('320');
  await page.locator('[name="height"]').fill('240');
  await page.locator('[name="duration"]').fill('1.0');
  await page.locator('[name="fps"]').fill('10');
  const scheduleBox = page.getByTestId('export-schedule');
  await expect.poll(() => scheduleBox.getAttribute('data-frames'), { timeout: 30_000 }).toBe('11');
  const storedPlan = scheduleAttributes(await scheduleBox.evaluate(node => node.getAttributeNames()
    .filter(name => name.startsWith('data-')).map(name => `${name}=${node.getAttribute(name)}`).join(' ')));
  const storedFrames = Number(storedPlan['data-frames']);
  const storedSpacing = Number(storedPlan['data-spacing']);
  const storedStart = Number(storedPlan['data-window-start']);
  const storedEnd = Number(storedPlan['data-window-end']);
  assert.equal(storedFrames, 11, 'duration 1 s at 10 fps has 11 output frames');
  assert.ok(storedSpacing > 0, 'stored output spacing must be positive');
  assert.ok(Math.abs(Number(storedPlan['data-sim-per-second']) - storedSpacing * 10) < 1e-12);
  assert.ok(Math.abs(storedStart - frameTimes[0]) < 1e-9, 'default window starts at the first frame');
  assert.ok(Math.abs(storedEnd - frameTimes.at(-1)) < 1e-9, 'default window ends at the last frame');
  const storedTargets = Array.from({ length: storedFrames }, (_, index) => storedStart + index * storedSpacing);
  assert.ok(storedTargets.every((time, index) => index === 0 || time > storedTargets[index - 1]),
    'stored output times must be strictly increasing');
  assert.ok(Math.abs(storedTargets.at(-1) - storedEnd) < 1e-12, 'stored output ends at the window end');
  await page.getByTestId('export-submit').click();
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.recording.status), { timeout: 180_000 })
    .toBe('completed');
  await page.locator('.export-close').click();
  const storedGif = await readFile(storedOutput);
  assert.equal(gifFrameCount(storedGif), storedFrames, 'stored GIF must have the scheduled frame count');

  // Live export: Δt = sim_per_second / fps, so the output count follows the
  // simulation time elapsed while recording, not the number of stored frames.
  const liveStart = await page.evaluate(() => window.twoPhaseAMRStudio.state.time);
  await page.getByTestId('open-export').click();
  await page.getByTestId('export-source').selectOption('live');
  await page.getByTestId('export-path').fill(liveOutput);
  await page.locator('[name="width"]').fill('320');
  await page.locator('[name="height"]').fill('240');
  await page.locator('[name="sim_per_second"]').fill('0.05');
  await page.locator('[name="fps"]').fill('10');
  const livePlan = await page.getByTestId('export-schedule').textContent();
  assert.match(livePlan, /0\.005 simulation s per frame/);
  await page.getByTestId('export-submit').click();
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.recording.active)).toBe(true);
  await page.locator('.export-close').click();
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('RUNNING');
  await page.waitForTimeout(4000);
  await page.getByTestId('control-pause').click();
  await expect(page.getByTestId('run-state')).toHaveText('PAUSED');
  const liveEnd = await page.evaluate(() => window.twoPhaseAMRStudio.state.time);
  await page.getByTestId('open-export').click();
  await page.getByTestId('export-stop').click();
  await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.recording.status), { timeout: 180_000 })
    .toBe('completed');
  const live = await page.evaluate(() => {
    const recording = window.twoPhaseAMRStudio.state.recording;
    return { encoded: recording.encoded, times: recording.times };
  });
  await page.locator('.export-close').click();
  assert.ok(Array.isArray(live.times) && live.times.length === live.encoded,
    'live state must report one time per encoded frame');
  assert.ok(live.times.every((time, index) => index === 0 || time >= live.times[index - 1]),
    'live frame time labels must be monotone');
  assert.ok(Math.abs(live.times[0] - liveStart) <= 0.02, 'live output starts at the recording start time');
  assert.ok(live.times.at(-1) <= liveEnd + 1e-9, 'live output never passes the stopped time');
  const expectedLive = Math.floor((liveEnd - liveStart) / 0.005) + 1;
  assert.ok(Math.abs(live.encoded - expectedLive) <= 2,
    `live output count ${live.encoded} must follow Δt (expected about ${expectedLive})`);
  const liveGif = await readFile(liveOutput);
  assert.ok(liveGif.length > 6 && ['GIF87a', 'GIF89a'].includes(liveGif.subarray(0, 6).toString('ascii')),
    'live export lacks a GIF header');
  assert.equal(liveGif.at(-1), 0x3b, 'live export lacks a GIF trailer');
  assert.deepEqual(pageErrors, []);
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
console.log('Time-linear stored and live export smoke passed.');
