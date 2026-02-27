import { defineConfig } from 'tsup'

import pkg from './package.json'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  sourcemap: true,
  external: [],
  define: {
    __VERSION__: JSON.stringify(pkg.version)
  }
})
