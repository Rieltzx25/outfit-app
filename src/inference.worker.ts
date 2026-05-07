/// <reference lib="webworker" />
// Standard onnxruntime-web entry — uses jsep multi-threaded WASM, fastest for our
// hardware. We tried /webgpu earlier but the user's laptop GPU couldn't init
// WebGPU (timed out) and the asyncify WASM is ~13% slower than jsep, so net loss.
import * as ort from 'onnxruntime-web'
import { OUTFIT_LABELS } from './classes'

// ORT needs both .wasm and the paired .mjs (pthread loader) at one URL prefix.
// We mirror onnxruntime-web/dist into public/ort/ at build time (see vite.config.ts)
// so they're same-origin — necessary because COOP/COEP=require-corp would block
// jsdelivr otherwise. Same-origin + COOP/COEP unlocks SharedArrayBuffer, which
// unlocks multi-thread WASM.
ort.env.wasm.wasmPaths = '/ort/'
// More threads = lower latency until we hit physical core count. 8 is safe on
// any laptop with >=8 logical cores; user has 24.
ort.env.wasm.numThreads = Math.min(8, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 1)
// SIMD is auto-detected but explicit for clarity.
ort.env.wasm.simd = true

let mainSession: ort.InferenceSession | null = null
let specialistSession: ort.InferenceSession | null = null
let activeBackend: 'webgpu' | 'wasm' = 'wasm'

// FP32 weights. Tried FP16 — file is half the size but WASM SIMD has no native
// FP16 ops, so runtime latency is unchanged (sometimes slightly worse due to
// cast overhead). Keep FP32 since speed beats download size for repeat users.
const MAIN_URL = '/models/outfit-v4-35cls-yolov8s.onnx'
const SPECIALIST_URL = '/models/shoe-specialist-v1.onnx'
const MAIN_BYTES = 45_000_000
const SPECIALIST_BYTES = 12_500_000
const TOTAL_BYTES = MAIN_BYTES + SPECIALIST_BYTES

const SHOE_IDX = 16
const HEELS_IDX = 19
const SANDAL_IDX = 17
const BOOT_IDX = 18
const TANKTOP_IDX = 2
const TSHIRT_IDX = 1
const FOOTWEAR_CLASSES = new Set([SHOE_IDX, HEELS_IDX, SANDAL_IDX, BOOT_IDX])

const CACHE_NAME = 'outfit-models-v7-fp32-final'

async function getOrFetchModel(url: string, expectedBytes: number, onPct: (pct: number) => void): Promise<ArrayBuffer> {
  try {
    const cache = await caches.open(CACHE_NAME)
    const hit = await cache.match(url)
    if (hit) {
      onPct(100)
      return await hit.arrayBuffer()
    }
    const buf = await streamDownload(url, expectedBytes, onPct)
    cache.put(url, new Response(buf.slice(0), { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.byteLength) } }))
      .catch(e => console.warn('cache.put failed:', e))
    return buf
  } catch {
    return await streamDownload(url, expectedBytes, onPct)
  }
}

async function streamDownload(url: string, expectedBytes: number, onPct: (pct: number) => void): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`failed to fetch ${url}: ${res.status}`)
  const total = Number(res.headers.get('content-length')) || expectedBytes
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value); received += value.byteLength
    onPct(Math.min(99, Math.round((received / total) * 100)))
  }
  const buf = new Uint8Array(received)
  let off = 0
  for (const c of chunks) { buf.set(c, off); off += c.byteLength }
  return buf.buffer
}

async function loadModel(post: (msg: string, pct?: number) => void) {
  if (mainSession && specialistSession) { post('ready', 100); return }
  post('checking cache…', 0)

  let mainPct = 0, specPct = 0
  const updateOverall = () => {
    const overall = Math.round(((mainPct * MAIN_BYTES) + (specPct * SPECIALIST_BYTES)) / TOTAL_BYTES)
    const fromCache = mainPct === 100 && specPct === 100
    post(fromCache ? 'loading from cache' : `downloading — ${overall}% (main ${mainPct}%, specialist ${specPct}%)`, overall)
  }

  const [mainBuf, specBuf] = await Promise.all([
    getOrFetchModel(MAIN_URL, MAIN_BYTES, p => { mainPct = p; updateOverall() }),
    getOrFetchModel(SPECIALIST_URL, SPECIALIST_BYTES, p => { specPct = p; updateOverall() }),
  ])

  const create = async (buf: ArrayBuffer, label: string) => {
    post(`initializing ${label} (wasm)…`, 99)
    return await ort.InferenceSession.create(buf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      executionMode: 'parallel',
      enableCpuMemArena: true,
      enableMemPattern: true,
      freeDimensionOverrides: { batch: 1 },
    })
  }

  activeBackend = 'wasm'
  mainSession = await create(mainBuf, 'main model')
  specialistSession = await create(specBuf, 'shoe specialist')
  post('ready', 100)
}

export interface Box {
  x1: number; y1: number; x2: number; y2: number
  score: number
  cls: number
}

const IMG_SIZE = 640
const NMS_IOU = 0.45
const SPECIALIST_CONF = 0.40
const SPECIALIST_OVERRIDE_IOU = 0.4

let _scratch: OffscreenCanvas | null = null
function scratch() {
  if (!_scratch) _scratch = new OffscreenCanvas(IMG_SIZE, IMG_SIZE)
  return _scratch
}

function letterbox(bitmap: ImageBitmap, srcW: number, srcH: number) {
  const r = Math.min(IMG_SIZE / srcW, IMG_SIZE / srcH)
  const newW = Math.round(srcW * r)
  const newH = Math.round(srcH * r)
  const padX = (IMG_SIZE - newW) / 2
  const padY = (IMG_SIZE - newH) / 2
  const c = scratch()
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.fillStyle = '#727272'
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE)
  ctx.drawImage(bitmap, 0, 0, srcW, srcH, padX, padY, newW, newH)
  const px = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data
  const plane = IMG_SIZE * IMG_SIZE
  const inp = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    inp[i] = px[i * 4] / 255
    inp[plane + i] = px[i * 4 + 1] / 255
    inp[2 * plane + i] = px[i * 4 + 2] / 255
  }
  return { tensor: new ort.Tensor('float32', inp, [1, 3, IMG_SIZE, IMG_SIZE]), r, padX, padY }
}

function decode(out: Record<string, ort.Tensor>, r: number, padX: number, padY: number, nc: number, classOffset: number, conf: number): Box[] {
  const key = Object.keys(out)[0]
  const data = out[key].data as Float32Array
  const dims = out[key].dims as number[]
  const N = dims[2]
  const candidates: Box[] = []
  for (let i = 0; i < N; i++) {
    let bestS = 0, bestC = -1
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * N + i]
      if (s > bestS) { bestS = s; bestC = c }
    }
    if (bestS < conf) continue
    const cx = data[0 * N + i]
    const cy = data[1 * N + i]
    const w = data[2 * N + i]
    const h = data[3 * N + i]
    let x1 = cx - w / 2, y1 = cy - h / 2
    let x2 = cx + w / 2, y2 = cy + h / 2
    x1 = (x1 - padX) / r; y1 = (y1 - padY) / r
    x2 = (x2 - padX) / r; y2 = (y2 - padY) / r
    candidates.push({ x1, y1, x2, y2, score: bestS, cls: bestC + classOffset })
  }
  return nms(candidates, NMS_IOU)
}

// Profile flag — set true to log step timings to console. Off by default in prod.
let profile = false
function tick(): number { return performance.now() }

async function detect(bitmap: ImageBitmap, srcW: number, srcH: number, conf: number): Promise<Box[]> {
  if (!mainSession || !specialistSession) throw new Error('Models not loaded')

  const t0 = tick()
  const { tensor, r, padX, padY } = letterbox(bitmap, srcW, srcH)
  bitmap.close()
  const t1 = tick()

  const mainOut = await mainSession.run({ images: tensor })
  const t2 = tick()
  const mainBoxes = decode(mainOut, r, padX, padY, OUTFIT_LABELS.length, 0, conf)

  const mainHasFootwear = mainBoxes.some(b => FOOTWEAR_CLASSES.has(b.cls))
  let specBoxes: Box[] = []
  if (mainHasFootwear) {
    const specOut = await specialistSession.run({ images: tensor })
    specBoxes = decode(specOut, r, padX, padY, 1, SHOE_IDX, SPECIALIST_CONF)
  }

  const survivingMain: Box[] = []
  for (const m of mainBoxes) {
    if (FOOTWEAR_CLASSES.has(m.cls)) {
      let overridden = false
      for (const s of specBoxes) {
        if (iou(m, s) > SPECIALIST_OVERRIDE_IOU) { overridden = true; break }
      }
      if (!overridden) survivingMain.push(m)
    } else {
      survivingMain.push(m)
    }
  }

  const finalBoxes: Box[] = []
  for (const b of survivingMain) {
    if (b.cls !== TSHIRT_IDX) { finalBoxes.push(b); continue }
    let isTankTop = false
    for (const o of survivingMain) {
      if (o.cls === TANKTOP_IDX && iou(b, o) > 0.5 && o.score > 0.20) {
        isTankTop = true; break
      }
    }
    if (isTankTop) {
      finalBoxes.push({ ...b, cls: TANKTOP_IDX })
    } else {
      finalBoxes.push(b)
    }
  }

  const dedup = nms(finalBoxes, 0.6)
  const t3 = tick()
  lastBreakdown = { letterbox: t1-t0, main_run: t2-t1, post: t3-t2, total: t3-t0 }
  if (profile) {
    console.log(`[detect] letterbox=${(t1-t0).toFixed(1)}ms main_run=${(t2-t1).toFixed(1)}ms post=${(t3-t2).toFixed(1)}ms total=${(t3-t0).toFixed(1)}ms`)
  }
  return [...dedup, ...specBoxes]
}

let lastBreakdown: { letterbox: number; main_run: number; post: number; total: number } | null = null

function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
  return ua > 0 ? inter / ua : 0
}

function nms(boxes: Box[], thr: number): Box[] {
  boxes.sort((a, b) => b.score - a.score)
  const keep: Box[] = []
  const taken = new Array<boolean>(boxes.length).fill(false)
  for (let i = 0; i < boxes.length; i++) {
    if (taken[i]) continue
    keep.push(boxes[i])
    for (let j = i + 1; j < boxes.length; j++) {
      if (taken[j]) continue
      if (boxes[j].cls === boxes[i].cls && iou(boxes[i], boxes[j]) >= thr) taken[j] = true
    }
  }
  return keep
}

type InMsg =
  | { type: 'load'; id: number }
  | { type: 'detect'; id: number; bitmap: ImageBitmap; srcW: number; srcH: number; conf: number }
  | { type: 'profile'; id: number; on: boolean }

self.onmessage = async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data
  try {
    if (msg.type === 'load') {
      await loadModel((m, pct) => self.postMessage({ type: 'progress', id: msg.id, msg: m, pct }))
      self.postMessage({ type: 'loaded', id: msg.id, backend: activeBackend })
    } else if (msg.type === 'detect') {
      const boxes = await detect(msg.bitmap, msg.srcW, msg.srcH, msg.conf)
      self.postMessage({ type: 'detected', id: msg.id, boxes, backend: activeBackend, breakdown: lastBreakdown })
    } else if (msg.type === 'profile') {
      profile = msg.on
      self.postMessage({ type: 'loaded', id: msg.id, backend: activeBackend })
    }
  } catch (e) {
    self.postMessage({ type: 'error', id: msg.id, message: e instanceof Error ? e.message : String(e) })
  }
}
