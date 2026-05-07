// Thin client for the inference worker. All ORT/ONNX work happens off the main thread
// in `inference.worker.ts` so the video element and rAF rendering stay smooth even when
// inference is heavy (which it is — YOLOv8s 640×640 on WASM).
import InferenceWorker from './inference.worker?worker'
export { OUTFIT_LABELS, OUTFIT_COLORS, OUTFIT_GROUPS, type OutfitLabel } from './classes'

export interface Box {
  x1: number; y1: number; x2: number; y2: number
  score: number
  cls: number
}

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: (msg: string, pct?: number) => void }>()

// Mutable so we can reflect the worker's chosen backend (webgpu/wasm) once load completes.
export let ACTIVE_BACKEND: 'webgpu' | 'wasm' = 'wasm'

function ensureWorker(): Worker {
  if (worker) return worker
  worker = new InferenceWorker()
  worker.onmessage = (ev: MessageEvent) => {
    const data = ev.data
    const entry = pending.get(data.id)
    if (!entry) return
    if (data.type === 'progress') {
      entry.onProgress?.(data.msg, data.pct)
    } else if (data.type === 'loaded') {
      ACTIVE_BACKEND = data.backend
      pending.delete(data.id)
      entry.resolve(undefined)
    } else if (data.type === 'detected') {
      ACTIVE_BACKEND = data.backend
      if (data.breakdown) {
        const b = data.breakdown as { letterbox: number; main_run: number; post: number; total: number }
        console.log(`[detect] letterbox=${b.letterbox.toFixed(1)} main_run=${b.main_run.toFixed(1)} post=${b.post.toFixed(1)} total=${b.total.toFixed(1)} ms`)
      }
      pending.delete(data.id)
      entry.resolve(data.boxes)
    } else if (data.type === 'error') {
      pending.delete(data.id)
      entry.reject(new Error(data.message))
    }
  }
  worker.onerror = (e) => {
    console.error('inference worker error:', e)
  }
  return worker
}

export async function loadModel(onProgress?: (msg: string, pct?: number) => void): Promise<'webgpu' | 'wasm'> {
  const w = ensureWorker()
  const id = nextId++
  return new Promise<'webgpu' | 'wasm'>((resolve, reject) => {
    pending.set(id, {
      resolve: () => resolve(ACTIVE_BACKEND),
      reject,
      onProgress: (msg, pct) => onProgress?.(msg, pct),
    })
    w.postMessage({ type: 'load', id })
  })
}

export async function detect(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  srcW: number,
  srcH: number,
  options: { confThreshold?: number } = {},
): Promise<Box[]> {
  const w = ensureWorker()
  const conf = options.confThreshold ?? 0.35

  // createImageBitmap snapshots the current frame; transferring it to the worker hands
  // off ownership so there's no copy. This is the cheap part — the model run inside the
  // worker is what takes ~1s. The main thread is free during that wait.
  const bitmap = await createImageBitmap(source)
  const id = nextId++

  return new Promise<Box[]>((resolve, reject) => {
    pending.set(id, {
      resolve: (v) => resolve(v as Box[]),
      reject,
    })
    w.postMessage({ type: 'detect', id, bitmap, srcW, srcH, conf }, [bitmap])
  })
}
