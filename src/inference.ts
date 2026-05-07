import * as ort from 'onnxruntime-web/webgpu'
export { OUTFIT_LABELS, OUTFIT_COLORS, OUTFIT_GROUPS, type OutfitLabel } from './classes'
import { OUTFIT_LABELS } from './classes'

// Let Vite bundle the .wasm asset (needed for the asyncify variant used by WebGPU);
// fall back to CDN only if needed in a non-bundled context.
// Multi-threaded WASM (used as fallback when WebGPU unavailable)
ort.env.wasm.numThreads = Math.min(4, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 1)

// Detect WebGPU support once. We probe `navigator.gpu` here, but actual init
// may still fail on some laptops (old drivers, iGPU). loadModel() handles the
// fallback to WASM if WebGPU session creation throws.
// Allow forcing WASM via ?backend=wasm or localStorage('outfit-backend')='wasm'
// to work around WebGPU stalls on certain Chrome+laptop combos.
function backendOverride(): 'webgpu' | 'wasm' | null {
  if (typeof window === 'undefined') return null
  try {
    const url = new URL(window.location.href)
    const q = url.searchParams.get('backend')
    if (q === 'wasm' || q === 'webgpu') return q
    const ls = localStorage.getItem('outfit-backend')
    if (ls === 'wasm' || ls === 'webgpu') return ls
  } catch { /* ignore */ }
  return null
}
const _override = backendOverride()
const _hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator
const hasWebGPU = _override === 'wasm' ? false : _hasWebGPU
export let ACTIVE_BACKEND: 'webgpu' | 'wasm' = hasWebGPU ? 'webgpu' : 'wasm'

let mainSession: ort.InferenceSession | null = null
let specialistSession: ort.InferenceSession | null = null

const MAIN_URL = '/models/outfit-v4-35cls-yolov8s.onnx'
const SPECIALIST_URL = '/models/shoe-specialist-v1.onnx'
const MAIN_BYTES = 45_000_000
const SPECIALIST_BYTES = 12_500_000
const TOTAL_BYTES = MAIN_BYTES + SPECIALIST_BYTES

// Class indices (must match classes.ts)
const SHOE_IDX = 16
const HEELS_IDX = 19
const SANDAL_IDX = 17
const BOOT_IDX = 18
const TANKTOP_IDX = 2
const TSHIRT_IDX = 1
const FOOTWEAR_CLASSES = new Set([SHOE_IDX, HEELS_IDX, SANDAL_IDX, BOOT_IDX])

// Cache models in the Cache API so they survive hard-refresh and offline.
// Bump CACHE_NAME if model files change to invalidate stale entries.
const CACHE_NAME = 'outfit-models-v4'

async function getOrFetchModel(url: string, expectedBytes: number, onPct: (pct: number) => void): Promise<ArrayBuffer> {
  // Try Cache API first (persists across reloads — even hard-refresh)
  try {
    const cache = await caches.open(CACHE_NAME)
    const hit = await cache.match(url)
    if (hit) {
      onPct(100)
      return await hit.arrayBuffer()
    }
    // Cache miss — stream download with progress
    const buf = await streamDownload(url, expectedBytes, onPct)
    // Store in cache for next load (don't await — let it happen in background)
    cache.put(url, new Response(buf.slice(0), { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.byteLength) } }))
      .catch(e => console.warn('cache.put failed:', e))
    return buf
  } catch {
    // Cache API unavailable (e.g. private mode iOS) → fall back to plain fetch
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

export async function loadModel(onProgress?: (msg: string, pct?: number) => void) {
  if (mainSession && specialistSession) { onProgress?.('ready', 100); return }
  onProgress?.('checking cache…', 0)

  let mainPct = 0, specPct = 0
  const updateOverall = () => {
    const overall = Math.round(((mainPct * MAIN_BYTES) + (specPct * SPECIALIST_BYTES)) / TOTAL_BYTES)
    const fromCache = mainPct === 100 && specPct === 100
    onProgress?.(fromCache ? 'loading from cache' : `downloading — ${overall}% (main ${mainPct}%, specialist ${specPct}%)`, overall)
  }

  const [mainBuf, specBuf] = await Promise.all([
    getOrFetchModel(MAIN_URL, MAIN_BYTES, p => { mainPct = p; updateOverall() }),
    getOrFetchModel(SPECIALIST_URL, SPECIALIST_BYTES, p => { specPct = p; updateOverall() }),
  ])

  const tryCreate = async (buf: ArrayBuffer, label: string) => {
    if (hasWebGPU) {
      try {
        onProgress?.(`initializing ${label} (webgpu)…`, 99)
        return await ort.InferenceSession.create(buf, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
        })
      } catch (e) {
        console.warn(`webgpu init failed for ${label}, falling back to wasm:`, e)
        ACTIVE_BACKEND = 'wasm'
      }
    }
    onProgress?.(`initializing ${label} (wasm)…`, 99)
    return await ort.InferenceSession.create(buf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  }
  mainSession = await tryCreate(mainBuf, 'main model')
  specialistSession = await tryCreate(specBuf, 'shoe specialist')
  onProgress?.('ready', 100)
}

export interface Box {
  x1: number; y1: number; x2: number; y2: number
  score: number
  cls: number
}

const IMG_SIZE = 640
const NMS_IOU = 0.45
const SPECIALIST_CONF = 0.40   // shoe specialist threshold
const SPECIALIST_OVERRIDE_IOU = 0.4  // overlap to override main's footwear

function letterbox(source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement, srcW: number, srcH: number) {
  const r = Math.min(IMG_SIZE / srcW, IMG_SIZE / srcH)
  const newW = Math.round(srcW * r)
  const newH = Math.round(srcH * r)
  const padX = (IMG_SIZE - newW) / 2
  const padY = (IMG_SIZE - newH) / 2
  const tmp = document.createElement('canvas')
  tmp.width = IMG_SIZE; tmp.height = IMG_SIZE
  const ctx = tmp.getContext('2d', { willReadFrequently: true })!
  ctx.fillStyle = '#727272'
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE)
  ctx.drawImage(source, 0, 0, srcW, srcH, padX, padY, newW, newH)
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

export async function detect(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  srcW: number,
  srcH: number,
  options: { confThreshold?: number } = {},
): Promise<Box[]> {
  if (!mainSession || !specialistSession) throw new Error('Models not loaded')
  const CONF = options.confThreshold ?? 0.35

  const { tensor, r, padX, padY } = letterbox(source, srcW, srcH)

  // ORT WASM uses a single global WASM instance — concurrent run() across sessions
  // throws "Session already started". Must serialize.
  const mainOut = await runWithTimeout(mainSession.run({ images: tensor }), 8000, 'main')
  const mainBoxes = decode(mainOut, r, padX, padY, OUTFIT_LABELS.length, 0, CONF)

  // Skip specialist when main has no footwear candidates — saves ~half the inference time per frame.
  // Run it whenever main fires anything in the footwear range (incl. mistaken heels-on-sneakers).
  const mainHasFootwear = mainBoxes.some(b => FOOTWEAR_CLASSES.has(b.cls))
  let specBoxes: Box[] = []
  if (mainHasFootwear) {
    const specOut = await runWithTimeout(specialistSession.run({ images: tensor }), 8000, 'specialist')
    // Specialist outputs class 0 = shoe; remap to SHOE_IDX (16)
    specBoxes = decode(specOut, r, padX, padY, 1, SHOE_IDX, SPECIALIST_CONF)
  }

  // === Merge logic ===
  // 1. For each specialist box, find overlapping footwear in main; remove them.
  // 2. Keep specialist boxes (they replace main's broken shoe predictions).
  // 3. Tank-top post-process: if t-shirt and tank-top both present in same area, prefer tank-top.

  const survivingMain: Box[] = []
  for (const m of mainBoxes) {
    if (FOOTWEAR_CLASSES.has(m.cls)) {
      // Check if any specialist box covers this area
      let overridden = false
      for (const s of specBoxes) {
        if (iou(m, s) > SPECIALIST_OVERRIDE_IOU) {
          overridden = true; break
        }
      }
      if (!overridden) survivingMain.push(m)
    } else {
      survivingMain.push(m)
    }
  }

  // Tank-top post-process: convert overlapping t-shirt → tank-top if model also fired tank-top in same area
  // (our v4 model has weak tank-top class; reading both increases recall for sleeveless tops)
  const finalBoxes: Box[] = []
  for (const b of survivingMain) {
    if (b.cls !== TSHIRT_IDX) { finalBoxes.push(b); continue }
    // Check if there's a tank-top box that overlaps this t-shirt
    let isTankTop = false
    for (const o of survivingMain) {
      if (o.cls === TANKTOP_IDX && iou(b, o) > 0.5 && o.score > 0.20) {
        isTankTop = true; break
      }
    }
    if (isTankTop) {
      // Replace this t-shirt with tank-top label (keep box)
      finalBoxes.push({ ...b, cls: TANKTOP_IDX })
    } else {
      finalBoxes.push(b)
    }
  }

  // Also dedupe: if both tank-top and t-shirt-replaced-as-tanktop exist for same area, keep highest score
  const dedup = nms(finalBoxes, 0.6)

  // Add specialist's shoe boxes
  return [...dedup, ...specBoxes]
}

function runWithTimeout<T>(p: Promise<T>, ms: number, tag: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const id = setTimeout(() => reject(new Error(`inference timeout (${tag} > ${ms}ms)`)), ms)
    p.then(v => { clearTimeout(id); resolve(v) }, e => { clearTimeout(id); reject(e) })
  })
}

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
      // class-aware NMS: only suppress same-class overlaps
      if (boxes[j].cls === boxes[i].cls && iou(boxes[i], boxes[j]) >= thr) taken[j] = true
    }
  }
  return keep
}
