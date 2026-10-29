import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const base = dirname(fileURLToPath(import.meta.url));
const documents = {
  'data-cleaning': await readFile(join(base, 'data-cleaning.md'), 'utf8'),
  'reward-design': await readFile(join(base, 'reward-design.md'), 'utf8'),
};

await writeFile(
  join(base, 'embedded-docs.js'),
  `window.__DOC_PREVIEW_MARKDOWN__ = ${JSON.stringify(documents)};\n`,
  'utf8',
);
