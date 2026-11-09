import { createHash } from 'node:crypto';
import { readdir, readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.resolve(root, '../assets');
const checking = process.argv.includes('--check');

async function sourceFiles(directory) {
  const entries = await readdir(path.join(root, directory), { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const name = `${directory}/${entry.name}`;
    if (entry.isDirectory()) files.push(...await sourceFiles(name));
    else if (entry.isFile()) files.push(name);
  }
  return files;
}

const files = ['.nvmrc', 'package.json', 'package-lock.json', 'tsconfig.json', 'index.html',
  ...await sourceFiles('src'), ...await sourceFiles('scripts')].sort();
const hash = createHash('sha256');
for (const name of files) {
  hash.update(name).update('\0').update(await readFile(path.join(root, name))).update('\0');
}
const digest = hash.digest('hex');
const dockviewLicense = (await readFile(path.join(root, 'node_modules/dockview/LICENCE.md'), 'utf8')).trim();
const uplotLicense = (await readFile(path.join(root, 'node_modules/uplot/LICENSE'), 'utf8')).trim();
const banner = `/* source-sha256: ${digest}\n${dockviewLicense}\n\nuPlot:\n${uplotLicense}\n*/`;
const result = await esbuild.build({
  absWorkingDir: root,
  entryPoints: { studio: 'src/main.ts' },
  outdir: output,
  bundle: true,
  format: 'esm',
  target: 'es2022',
  minify: true,
  charset: 'utf8',
  legalComments: 'inline',
  banner: { js: banner, css: banner },
  write: false,
});
const outputs = new Map(result.outputFiles.map(file => [path.basename(file.path), file.contents]));
outputs.set('index.html', Buffer.from(`<!-- source-sha256: ${digest} -->\n${await readFile(path.join(root, 'index.html'), 'utf8')}`));
for (const name of ['studio.js', 'studio.css', 'index.html']) {
  const expected = outputs.get(name);
  if (!expected) throw new Error(`Build did not generate ${name}`);
  const destination = path.join(output, name);
  if (checking) {
    let actual;
    try { actual = await readFile(destination); } catch { throw new Error(`Missing ${name}; run npm run build`); }
    if (!Buffer.from(expected).equals(actual)) throw new Error(`Stale ${name}; run npm run build`);
  } else {
    await mkdir(output, { recursive: true });
    await writeFile(destination, expected);
  }
}
console.log(checking ? 'Studio assets are up to date.' : 'Studio assets generated.');
