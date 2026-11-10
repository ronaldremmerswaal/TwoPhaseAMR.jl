import { build } from 'esbuild';
import { chromium } from '@playwright/test';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const result = await build({ entryPoints: [resolve(frontend, 'src/renderer.ts')], bundle: true, format: 'iife',
  globalName: 'StudioRendererBenchmark', write: false });
const browser = await chromium.launch({ headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  await page.setContent('<div id="plot" style="position:absolute;inset:0"></div>');
  await page.addScriptTag({ content: result.outputFiles[0].text });
  const measurement = await page.evaluate(() => {
    const renderer = new window.StudioRendererBenchmark.SharedRenderer();
    const plot = document.querySelector('#plot');
    const blocks = [];
    for (let j = 0; j < 20; j++) for (let i = 0; i < 20; i++) {
      blocks.push({ level: 0, i, j, origin: [i / 20, j / 20], spacing: [1 / 320, 1 / 320] });
    }
    const values = new Float32Array(102400);
    for (let i = 0; i < values.length; i++) values[i] = (Math.sin(i * 0.01) + 1) / 2;
    const topology = { id: 1, blocks, block_size: [16, 16], domain: { origin: [0, 0], extent: [1, 1] }, face_tiles: [[], []], solids: [] };
    renderer.registerView('bench', plot);
    renderer.updateView('bench', { field: { topology, placement: 'cell', values }, camera: { center: [0.5, 0.5], scale: 1 / 720 }, range: [0, 1] });
    const gl = renderer.canvas.getContext('webgl2');
    for (let i = 0; i < 5; i++) { renderer.renderNow(); gl.finish(); }
    const pixel = new Uint8Array(4);
    gl.readPixels(640, 360, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    if (pixel[3] === 0) throw new Error('Benchmark field was not rendered');
    const samples = [];
    for (let i = 0; i < 20; i++) {
      const start = performance.now();
      for (let frame = 0; frame < 10; frame++) { renderer.renderNow(); gl.finish(); }
      samples.push((performance.now() - start) / 10);
    }
    renderer.dispose();
    samples.sort((a, b) => a - b);
    return { cells: values.length, blocks: blocks.length, median_ms: samples[Math.floor(samples.length / 2)], p95_ms: samples[Math.floor(samples.length * .95)] };
  });
  console.log(JSON.stringify(measurement));
} finally {
  await browser.close();
}
