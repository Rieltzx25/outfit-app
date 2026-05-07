import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { copyFileSync, mkdirSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))

// ORT needs both the .wasm binary and a paired .mjs (pthread loader) at the same
// URL prefix. Vite hashes the .wasm into /assets/ but doesn't pick up the .mjs.
// We copy the pair into dist/ort/ at build time and serve them under public/ort/
// in dev so ort.env.wasm.wasmPaths='/ort/' resolves both sides identically.
const ORT_FILES = [
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
]

function copyOrtRuntime(): Plugin {
  const ortDistDir = resolve(__dirname, 'node_modules/onnxruntime-web/dist')
  const publicOrtDir = resolve(__dirname, 'public/ort')
  return {
    name: 'copy-ort-runtime',
    apply: () => true,
    buildStart() {
      // Place into public/ort so dev server (vite serve) and prod build both see them.
      if (!existsSync(publicOrtDir)) mkdirSync(publicOrtDir, { recursive: true })
      for (const f of ORT_FILES) {
        const src = resolve(ortDistDir, f)
        const dst = resolve(publicOrtDir, f)
        if (existsSync(src)) copyFileSync(src, dst)
      }
    },
  }
}

// Cross-origin isolation enables SharedArrayBuffer so onnxruntime-web WASM can run
// multi-threaded (4x speedup on this hardware). Required for both dev and preview.
const crossOriginIsolation: Plugin = {
  name: 'configure-coop-coep',
  configureServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
      next()
    })
  },
  configurePreviewServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
      next()
    })
  },
}

export default defineConfig({
  plugins: [react(), copyOrtRuntime(), crossOriginIsolation],
  worker: {
    format: 'es',
  },
})
