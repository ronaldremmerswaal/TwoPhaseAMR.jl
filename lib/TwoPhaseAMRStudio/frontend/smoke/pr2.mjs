import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { chromium, expect as baseExpect } from '@playwright/test';

const expect = baseExpect.configure({ timeout: 45_000 });
const smokeDir = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(smokeDir, '../..');
const shots = '/tmp/studio-shots';
await mkdir(shots, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});

async function runExample(example) {
  const child = spawn(process.env.JULIA || 'julia', [
    '--startup-file=no', '--heap-size-hint=3G', '-t', '1', `--project=${packageRoot}`,
    join(smokeDir, 'examples.jl'), example,
  ], { cwd: packageRoot, stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, OPENBLAS_NUM_THREADS: '1', JULIA_NUM_PRECOMPILE_TASKS: '1' } });
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', text => { stderr += text; });
  child.stdin.on('error', () => {});
  let context;
  let failure;
  try {
    const address = await new Promise((resolveURL, reject) => {
      const lines = createInterface({ input: child.stdout });
      const timer = setTimeout(() => finish(new Error('Example startup timed out')), 1_200_000);
      const exited = (code, signal) => finish(new Error(`Example exited before readiness (${code ?? signal})`));
      function finish(error, value) {
        clearTimeout(timer); child.off('exit', exited); lines.close(); child.stdout.resume();
        if (error) reject(error); else resolveURL(value);
      }
      child.once('exit', exited);
      child.once('error', finish);
      lines.on('line', line => { if (line.startsWith('STUDIO_URL ')) finish(null, line.slice(11)); });
    });
    context = await browser.newContext({ viewport: { width: 1960, height: 1120 } });
    const page = await context.newPage();
    const errors = [];
    const sent = [];
    page.on('websocket', socket => socket.on('framesent', ({ payload }) => {
      if (typeof payload === 'string') {
        try { sent.push(JSON.parse(payload)); } catch { /* Ignore malformed unrelated frames. */ }
      }
    }));
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(address, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await expect(page.getByTestId('connection-status')).toHaveText('connected');
    await expect(page.getByTestId('field-panel')).toHaveCount(2);
    await expect(page.locator('canvas[aria-hidden="true"]')).toHaveCount(1);
    await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews().filter(view => view.ready && view.worldRect).length)).toBe(2);
    await expect(page.getByTestId('run-state')).toHaveText('FINISHED');
    for (const control of ['control-pause', 'control-stop', 'control-keyframe']) await expect(page.getByTestId(control)).toBeDisabled();
    await expect(page.getByTestId('run-clock')).toContainText('last dt');
    await page.getByTestId('run-name').click();
    await expect(page.getByTestId('run-panel')).toBeVisible();
    await page.locator('[data-tab-panel-id="diagnostic-0"]').click();
    const views = await page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews());
    const expected = example === 'capillary-wave' ? ['volume_fraction', 'velocity_magnitude'] : ['velocity_vorticity', 'pressure'];
    assert.deepEqual(views.map(view => view.field), expected);
    await expect(page.getByTestId('field-retry').first()).toBeHidden();
    await expect(page.getByTestId('field-frame-badge').first()).toBeHidden();
    await page.getByTestId('field-picker-trigger').first().click();
    await page.getByTestId('field-search').first().fill('velocity');
    await expect.poll(() => page.getByTestId('field-picker').first().locator('option').count()).toBeGreaterThan(0);
    await page.getByTestId('field-search').first().fill('');
    await page.getByTestId('field-picker-trigger').first().click();
    if (example === 'capillary-wave') {
      await expect.poll(() => new Set(sent.filter(message => message.type === 'get' && message.what === 'field' &&
        message.field === 'velocity_magnitude').map(message => message.phase))).toEqual(new Set(['liquid', 'gas']));
      await page.getByTestId('field-settings').nth(1).click();
      await page.getByTestId('field-mode').nth(1).selectOption('plic');
      await page.getByTestId('field-settings').nth(1).click();
    } else {
      assert.ok(sent.some(message => message.type === 'get' && message.what === 'field' &&
        message.field === 'velocity_vorticity' && message.phase === undefined),
      'single-phase fields must be requested without a phase selector');
    }
    await page.evaluate(() => {
      const { state } = window.twoPhaseAMRStudio;
      if (state.timeline.frames.length < 2) throw new Error('Example must have at least two stored frames');
      state.navigate(timeline => timeline.edge('first'));
      state.navigate(timeline => timeline.edge('last'));
      state.navigate(timeline => timeline.edge('first'));
    });
    await expect.poll(() => page.evaluate(() => {
      const api = window.twoPhaseAMRStudio;
      return api.getFieldViews().every(view => view.ready && view.frame === api.state.timeline.selectedId);
    })).toBe(true);
    await page.getByTestId('timeline-follow').click();
    await expect.poll(() => page.evaluate(() => {
      const api = window.twoPhaseAMRStudio;
      return api.getFieldViews().every(view => view.ready && view.frame === api.state.timeline.selectedId);
    })).toBe(true);
    await page.getByTestId('timeline-scrub').hover();
    await expect(page.getByTestId('timeline-scrub')).toHaveAttribute('title', /Frame #.*t=.* · /);
    const diagnostics = page.getByTestId('diagnostic-panel');
    assert.ok(await diagnostics.count() > 0, 'diagnostic dimension groups were not docked');
    await expect.poll(() => page.locator('.diagnostic-panel .uplot').count()).toBeGreaterThan(0);
    if (example === 'decaying-turbulence') {
      const dimensions = await diagnostics.evaluateAll(nodes => nodes.map(node => node.dataset.dimension));
      assert.ok(new Set(dimensions).size > 1, 'different physical dimensions must be in different panels');
      await expect(page.locator('.diagnostic-legend').first()).toContainText(/energy|momentum|cells|blocks/i);
    }
    await expect.poll(async () => {
      const counts = await diagnostics.evaluateAll(nodes => nodes.map(node => node.querySelectorAll('[data-testid="diagnostic-toggle"]').length));
      return Math.max(0, ...counts);
    }).toBeGreaterThan(1);
    const multiIndex = await diagnostics.evaluateAll(nodes => nodes.findIndex(node => node.querySelectorAll('[data-testid="diagnostic-toggle"]').length > 1));
    const multiDiagnostic = diagnostics.nth(multiIndex);
    const diagnosticId = await multiDiagnostic.getAttribute('data-panel-id');
    const hiddenToggle = multiDiagnostic.getByTestId('diagnostic-toggle').first();
    const hiddenName = await hiddenToggle.getAttribute('data-name');
    const hiddenComponent = Number(await hiddenToggle.getAttribute('data-component'));
    await hiddenToggle.focus();
    await page.keyboard.press('Space');
    await expect(hiddenToggle).toHaveAttribute('aria-pressed', 'false');
    await multiDiagnostic.getByTestId('diagnostic-scale').selectOption('log');
    await expect(multiDiagnostic.getByTestId('diagnostic-scale-note')).toContainText(/zero and negative/i);
    await page.getByTestId('timeline-prev').click();
    await page.getByTestId('timeline-follow').click();
    await expect(hiddenToggle).toHaveAttribute('aria-pressed', 'false');
    await expect(multiDiagnostic.getByTestId('diagnostic-scale')).toHaveValue('log');
    const diagnosticPlot = multiDiagnostic.getByTestId('diagnostic-plot');
    await expect(diagnosticPlot).toHaveAttribute('data-x-label', 't');
    await diagnosticPlot.hover();
    await expect(multiDiagnostic.getByTestId('diagnostic-readout')).toBeVisible();
    await diagnosticPlot.click();
    await expect(page.getByTestId('timeline-mode')).toHaveText('HISTORY');
    await page.getByTestId('timeline-follow').click();
    const plots = page.getByTestId('field-plot');
    const first = plots.nth(0);
    const before = await first.boundingBox();
    assert.ok(before && before.width > 100 && before.height > 100, 'the field viewport should occupy the panel');
    const firstScale = Number(await first.getAttribute('data-camera-scale'));
    await page.mouse.move(before.x + before.width / 2, before.y + before.height / 2);
    await expect(page.getByTestId('field-hover').first()).toContainText(/block|level/i);
    await first.click({ position: { x: before.width / 2, y: before.height / 2 } });
    await expect(page.getByTestId('field-pinned').first()).toContainText(/block|level/i);
    const anchor = [0.33, 0.4];
    const initial = await page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews()[0]);
    await page.mouse.move(before.x + anchor[0] * before.width, before.y + anchor[1] * before.height);
    await page.mouse.wheel(0, -400);
    await expect.poll(async () => Number(await first.getAttribute('data-camera-scale'))).toBeLessThan(firstScale);
    const inward = await page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews()[0]);
    const point = view => [
      view.worldRect[0] + anchor[0] * (view.worldRect[1] - view.worldRect[0]),
      view.worldRect[3] - anchor[1] * (view.worldRect[3] - view.worldRect[2]),
    ];
    assert.ok(point(initial).every((value, index) => Math.abs(value - point(inward)[index]) < 1e-8), 'wheel zoom must stay anchored at the cursor');
    await page.mouse.wheel(0, 400);
    await expect.poll(async () => Number(await first.getAttribute('data-camera-scale'))).toBeGreaterThan(inward.camera.scale);
    await page.getByTestId('field-fit').first().click();
    await expect.poll(() => page.evaluate(() => {
      const api = window.twoPhaseAMRStudio;
      const view = api.getFieldViews()[0];
      const extent = api.state.runs[0].domain.extent;
      const ideal = Math.max(extent[0] / view.width, extent[1] / view.height);
      return Math.abs(view.camera.scale / ideal - 1);
    })).toBeLessThan(1e-6);
    await page.mouse.move(before.x + anchor[0] * before.width, before.y + anchor[1] * before.height);
    await page.mouse.wheel(0, -400);
    await expect.poll(async () => Number(await first.getAttribute('data-camera-scale'))).toBeLessThan(firstScale);
    const after = await first.boundingBox();
    assert.ok(after && Math.abs(after.width - before.width) < 1 && Math.abs(after.height - before.height) < 1,
      'zoom must fill the existing panel instead of shrinking it');
    const zoomed = await page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews()[0]);
    assert.ok(zoomed.worldRect);
    const [xmin, xmax, ymin, ymax] = zoomed.worldRect;
    assert.ok(Math.abs((xmax - xmin) / (ymax - ymin) - after.width / after.height) < 0.02,
      'camera limits must have the viewport aspect');
    await expect.poll(async () => {
      const scales = await plots.evaluateAll(nodes => nodes.map(node => Number(node.dataset.cameraScale)));
      return Math.abs(scales[0] - scales[1]);
    }).toBeLessThan(1e-10);
    await page.getByTestId('timeline-prev').click();
    await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews()[1].ready)).toBe(true);
    await page.getByTestId('field-pin').nth(1).click();
    await expect(page.getByTestId('field-frame-badge').nth(1)).toBeVisible();
    await page.getByTestId('timeline-follow').click();
    await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews().every(view => view.ready))).toBe(true);
    await page.getByTestId('field-settings').nth(1).click();
    await page.getByTestId('field-color-lock').nth(1).click();
    await page.getByTestId('field-settings').nth(1).click();
    const savedViews = await page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews());
    await page.setViewportSize({ width: 1200, height: 1000 });
    await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews().filter(view => view.ready).length)).toBe(2);
    const restoredViews = await page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews());
    for (const saved of savedViews) {
      const restored = restoredViews.find(view => view.panelId === saved.panelId);
      assert.deepEqual(
        [restored.field, restored.pinnedFrame, restored.linkGroup, restored.colorPolicy, restored.fixedColorRange, restored.fitMode, restored.camera],
        [saved.field, saved.pinnedFrame, saved.linkGroup, saved.colorPolicy, saved.fixedColorRange, saved.fitMode, saved.camera],
        'responsive layout must preserve each field view and camera',
      );
    }
    const narrowFields = await Promise.all([plots.nth(0).boundingBox(), plots.nth(1).boundingBox()]);
    const narrowDiagnostic = await diagnostics.first().boundingBox();
    assert.ok(narrowFields[0] && narrowFields[1] && narrowDiagnostic &&
      narrowDiagnostic.y >= Math.max(narrowFields[0].y + narrowFields[0].height, narrowFields[1].y + narrowFields[1].height) &&
      narrowDiagnostic.width > 1.5 * Math.max(narrowFields[0].width, narrowFields[1].width),
    'automatic narrow layout must dock diagnostics in a row below both fields');
    await expect.poll(async () => Math.abs(Number(await plots.first().getAttribute('data-camera-scale')) - zoomed.camera.scale)).toBeLessThan(1e-10);
    await page.setViewportSize({ width: 1960, height: 1120 });
    await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.getFieldViews().filter(view => view.ready).length)).toBe(2);
    const restoredDiagnostic = page.locator(`[data-testid="diagnostic-panel"][data-panel-id="${diagnosticId}"]`);
    await expect(restoredDiagnostic.getByTestId('diagnostic-scale')).toHaveValue('log');
    await expect(restoredDiagnostic.locator(`[data-testid="diagnostic-toggle"][data-name="${hiddenName}"][data-component="${hiddenComponent}"]`)).toHaveAttribute('aria-pressed', 'false');
    const sash = page.locator('.dv-split-view-container.dv-horizontal > .dv-sash-container > .dv-sash.dv-enabled').first();
    const sashBox = await sash.boundingBox();
    assert.ok(sashBox, 'dock layout must expose a resizable divider');
    await page.mouse.move(sashBox.x + sashBox.width / 2, sashBox.y + sashBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(sashBox.x + sashBox.width / 2 + 70, sashBox.y + sashBox.height / 2, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => Math.abs((await first.boundingBox()).width - after.width)).toBeGreaterThan(10);
    const resized = await first.boundingBox();
    const beforeBoxScale = Number(await first.getAttribute('data-camera-scale'));
    await page.getByTestId('field-box-zoom').first().click();
    await page.mouse.move(resized.x + .25 * resized.width, resized.y + .25 * resized.height);
    await page.mouse.down();
    await page.mouse.move(resized.x + .75 * resized.width, resized.y + .75 * resized.height, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => Number(await first.getAttribute('data-camera-scale'))).toBeLessThan(beforeBoxScale);
    await page.getByTestId('field-fit').first().click();
    if (example === 'capillary-wave') {
      const domainFit = Number(await first.getAttribute('data-camera-scale'));
      await expect(page.getByTestId('field-interface-fit').first()).toBeVisible();
      await page.getByTestId('field-interface-fit').first().click();
      await expect.poll(async () => Number(await first.getAttribute('data-camera-scale'))).toBeLessThan(domainFit);
      await page.getByTestId('field-fit').first().click();
    }

    for (const theme of ['light', 'dark']) {
      await page.getByTestId('theme-select').selectOption(theme);
      await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(theme);
      const ratios = await page.evaluate(() => {
        const canvas = document.createElement('canvas').getContext('2d');
        const css = getComputedStyle(document.documentElement);
        function rgb(variable) {
          canvas.fillStyle = css.getPropertyValue(variable).trim();
          const hex = canvas.fillStyle.slice(1);
          return [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16) / 255).map(c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4);
        }
        function luminance(c) { return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]; }
        return [['--text', '--surface'], ['--muted', '--surface'], ['--field-foreground', '--field-background'],
          ['--diagnostic-text', '--diagnostic-background'], ['--diagnostic-axis', '--diagnostic-background']].map(([a, b]) => {
          const x = luminance(rgb(a)), y = luminance(rgb(b));
          return (Math.max(x, y) + .05) / (Math.min(x, y) + .05);
        });
      });
      assert.ok(ratios.every(ratio => ratio >= 4.5), `${theme} contrast below WCAG AA: ${ratios}`);
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.screenshot({ path: join(shots, `pr2-${example}-${theme}.png`), fullPage: true });
      await page.getByTestId('field-panel').first().screenshot({ path: join(shots, `pr2-${example}-field-${theme}.png`) });
    }
    await page.screenshot({ path: join(shots, `pr2-${example}-diagnostic-log.png`), fullPage: true });

    await page.getByTestId('open-export').click();
    const dialog = page.getByTestId('export-dialog');
    await expect(dialog).toBeVisible();
    await expect(page.getByTestId('export-path')).toHaveValue(`${example}-${expected[0]}.mp4`);
    const output = join(shots, `pr2-${example}.gif`);
    await page.getByTestId('export-path').fill(output);
    await dialog.locator('[name="width"]').fill('640');
    await dialog.locator('[name="height"]').fill('360');
    const arrangement = example === 'capillary-wave' ? 'right' : 'below';
    await page.getByTestId('export-arrangement').selectOption(arrangement);
    const diagnosticInputs = page.getByTestId('export-diagnostic');
    let selectedDiagnostic = false;
    for (let index = 0; index < await diagnosticInputs.count(); index++) {
      const input = diagnosticInputs.nth(index);
      if (JSON.parse(await input.inputValue()).includes(hiddenName)) {
        await input.check();
        selectedDiagnostic = true;
        break;
      }
    }
    assert.ok(selectedDiagnostic, 'the configured diagnostic group must be exportable');
    await page.getByTestId('export-preview').click();
    await expect(page.getByTestId('export-preview-image')).toBeVisible({ timeout: 240_000 });
    await expect.poll(() => page.getByTestId('export-preview-image').evaluate(image => image.naturalWidth), { timeout: 240_000 }).toBeGreaterThan(0);
    await page.screenshot({ path: join(shots, `pr2-${example}-export-preview.png`), fullPage: true });
    await page.getByTestId('export-submit').click();
    await expect.poll(() => sent.some(message => message.type === 'recording' && message.action === 'export' &&
      message.field === expected[0] && message.diagnostics.length > 0 && message.arrangement === arrangement &&
      message.diagnostic_options.some(option => option.scale === 'log' && option.hidden.some(item => item.name === hiddenName && item.component === hiddenComponent)))).toBe(true);
    await expect.poll(() => page.evaluate(() => window.twoPhaseAMRStudio.state.recording.status), { timeout: 240_000 }).toBe('completed');
    const gif = await readFile(output);
    assert.ok(gif.length > 7 && ['GIF87a', 'GIF89a'].includes(gif.subarray(0, 6).toString('ascii')) && gif.at(-1) === 0x3b,
      'dialog export must produce a complete GIF');
    await expect(page.getByTestId('export-download')).toBeVisible();
    const downloadURL = await page.getByTestId('export-download').getAttribute('href');
    const downloaded = await page.request.get(downloadURL);
    assert.equal(downloaded.status(), 200, 'completed export must be downloadable with the Studio token');
    assert.deepEqual(await downloaded.body(), gif);
    await dialog.locator('.export-close').click();
    const firstTab = page.locator('[data-tab-panel-id="field-primary"]');
    const secondTab = page.locator('[data-tab-panel-id="field-secondary"]');
    const sameGroup = () => page.evaluate(() => {
      const a = document.querySelector('[data-tab-panel-id="field-primary"]');
      const b = document.querySelector('[data-tab-panel-id="field-secondary"]');
      return !!a && !!b && a.closest('.dv-tabs-container') === b.closest('.dv-tabs-container');
    });
    const source = await firstTab.boundingBox();
    const destination = await secondTab.boundingBox();
    await page.mouse.move(source.x + 20, source.y + source.height / 2);
    await page.mouse.down();
    await page.mouse.move(source.x + 50, source.y + source.height / 2, { steps: 5 });
    await page.mouse.move(destination.x + 20, destination.y + destination.height / 2, { steps: 20 });
    await page.mouse.move(destination.x + 21, destination.y + destination.height / 2);
    await page.mouse.up();
    await expect.poll(sameGroup).toBe(true);
    await page.screenshot({ path: join(shots, `pr2-${example}-drag-tab.png`) });

    const joined = await firstTab.boundingBox();
    const targetPlot = await page.locator('[data-testid="field-plot"]:visible').first().boundingBox();
    await page.mouse.move(joined.x + 20, joined.y + joined.height / 2);
    await page.mouse.down();
    await page.mouse.move(joined.x + 50, joined.y + joined.height / 2, { steps: 5 });
    await page.mouse.move(targetPlot.x + targetPlot.width - 30, targetPlot.y + targetPlot.height / 2, { steps: 25 });
    await page.mouse.move(targetPlot.x + targetPlot.width - 29, targetPlot.y + targetPlot.height / 2);
    await expect(page.locator('.dv-drop-target-selection:visible').first()).toBeVisible();
    await page.screenshot({ path: join(shots, `pr2-${example}-drag-split-target.png`) });
    await page.mouse.up();
    await expect.poll(sameGroup).toBe(false);
    await page.screenshot({ path: join(shots, `pr2-${example}-drag-split.png`) });

    const split = await firstTab.boundingBox();
    await page.keyboard.down('Shift');
    await page.mouse.move(split.x + 20, split.y + split.height / 2);
    await page.mouse.down();
    await page.mouse.move(split.x + 160, split.y + 120, { steps: 20 });
    await page.mouse.up();
    await page.keyboard.up('Shift');
    await expect(page.locator('.dv-resize-container [data-tab-panel-id="field-primary"]')).toBeVisible();
    await expect.poll(async () => (await page.locator('.dv-resize-container [data-testid="field-plot"]').boundingBox())?.height ?? 0).toBeGreaterThan(40);
    await page.screenshot({ path: join(shots, `pr2-${example}-drag-floating.png`) });
    await firstTab.locator('.dv-default-tab-action').click();
    await expect(page.getByTestId('field-panel')).toHaveCount(1);
    await page.locator('[data-tab-panel-id="diagnostic-0"] .dv-default-tab-action').click();
    await expect(page.locator('[data-tab-panel-id="diagnostic-0"]')).toHaveCount(0);
    await page.getByTestId('add-panel-menu').click();
    await expect(page.getByTestId('panel-option-run')).toBeVisible();
    await expect(page.getByTestId('panel-option-frames')).toBeVisible();
    await expect(page.getByTestId('panel-option-diagnostic-0')).toBeVisible();
    await page.getByTestId('panel-option-field').click();
    await expect(page.getByTestId('field-panel')).toHaveCount(2);
    await page.getByTestId('add-panel-menu').click();
    await page.getByTestId('panel-option-diagnostic-0').click();
    await expect(page.locator('[data-tab-panel-id^="diagnostic-0-added-"]')).toBeVisible();
    await page.locator('.layout-menu summary').click();
    await page.getByTestId('layout-auto').click();
    await expect(page.getByTestId('field-panel')).toHaveCount(2);
    assert.deepEqual(errors, []);
    console.log(`${example}: fields, zoom, ${await diagnostics.count()} diagnostic groups, stored export, light/dark screenshots passed`);
  } catch (error) {
    error.message += `\nJulia stderr:\n${stderr.replace(/token=[^&\s]+/g, 'token=[redacted]')}`;
    failure = error;
  } finally {
    await context?.close();
    if (child.exitCode === null && child.signalCode === null) {
      const exited = new Promise(resolveExit => child.once('exit', resolveExit));
      child.stdin.end('quit\n');
      const timer = setTimeout(() => child.kill('SIGKILL'), 30_000);
      await exited;
      clearTimeout(timer);
    }
    if (!failure) assert.equal(child.exitCode, 0, `Fixture ${example} failed: ${stderr}`);
  }
  if (failure) throw failure;
}

try {
  await runExample('capillary-wave');
  await runExample('decaying-turbulence');
} finally {
  await browser.close();
}
