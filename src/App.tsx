import { useEffect, useRef, useState } from 'react'
import { loadModel, detect, OUTFIT_LABELS, OUTFIT_COLORS, OUTFIT_GROUPS, type Box } from './inference'

type Status = 'idle' | 'loading' | 'live' | 'paused' | 'error'

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const lastImgRef = useRef<HTMLImageElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const aliveRef = useRef<(() => void) | null>(null)
  const fpsRef = useRef<{ t: number; n: number }>({ t: performance.now(), n: 0 })

  const [status, setStatus] = useState<Status>('idle')
  const [statusMsg, setStatusMsg] = useState('initialize')
  const [boxes, setBoxes] = useState<Box[]>([])
  const [fps, setFps] = useState(0)
  const [latency, setLatency] = useState(0)
  const [confThreshold, setConfThreshold] = useState(0.35)
  const [facing, setFacing] = useState<'user' | 'environment'>('environment')
  const [backend, setBackend] = useState<'webgpu' | 'wasm'>('wasm')
  const confRef = useRef(0.35)
  useEffect(() => { confRef.current = confThreshold }, [confThreshold])

  useEffect(() => () => stopAll(), [])

  function stopAll() {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    aliveRef.current?.()
    aliveRef.current = null
    const v = videoRef.current
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach(t => t.stop())
      v.srcObject = null
    }
  }

  async function start() {
    try {
      setStatus('loading')
      setStatusMsg('loading model')
      const be = await loadModel(m => setStatusMsg(m.toLowerCase()))
      setBackend(be)

      setStatusMsg('requesting camera')
      // Mark as 'live' before awaiting metadata so the <video> becomes visible —
      // some mobile browsers won't fire loadedmetadata on display:none videos.
      setStatus('live')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      }).catch(async (err) => {
        // Retry without facingMode constraint on devices that don't support it
        if ((err as { name?: string })?.name === 'OverconstrainedError') {
          return navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        }
        throw err
      })
      const v = videoRef.current!
      v.muted = true
      v.playsInline = true
      v.setAttribute('playsinline', 'true')   // iOS Safari quirk
      v.setAttribute('webkit-playsinline', 'true')
      v.srcObject = stream
      // Wait for metadata with a 6s safety timeout so we never hang forever
      await Promise.race([
        new Promise<void>((res) => {
          if (v.readyState >= 1) return res()
          const onmd = () => { v.removeEventListener('loadedmetadata', onmd); res() }
          v.addEventListener('loadedmetadata', onmd)
        }),
        new Promise<void>((res) => setTimeout(res, 6000)),
      ])
      try { await v.play() } catch (err) {
        console.warn('video.play() rejected:', err)
      }

      const c = canvasRef.current!
      c.width = v.videoWidth || 1280
      c.height = v.videoHeight || 720
      lastImgRef.current = null

      setStatusMsg('live')
      runLoop()
    } catch (e) {
      console.error('start() failed:', e)
      setStatus('error')
      const name = (e as { name?: string })?.name
      const msg = e instanceof Error ? e.message : String(e)
      let friendly = (msg || 'unknown error').toLowerCase()
      if (name === 'NotAllowedError') friendly = 'camera permission denied — click the lock icon and allow'
      else if (name === 'NotFoundError') friendly = 'no camera detected'
      else if (name === 'NotReadableError') friendly = 'camera in use by another app'
      else if (name === 'OverconstrainedError') friendly = 'camera does not support 30fps@720p'
      setStatusMsg(friendly)
    }
  }

  function stop() {
    stopAll()
    setStatus('paused')
    setStatusMsg('stopped')
    setBoxes([])
    const c = canvasRef.current
    if (c) c.getContext('2d')?.clearRect(0, 0, c.width, c.height)
  }

  function runLoop() {
    // rAF-driven loop: render every frame (60 FPS), inference fires-and-forgets in
    // background — no nested async while-loop, which can stall on WebGPU.
    let latestBoxes: Box[] = []
    let inferring = false
    let inferStartedAt = 0
    let alive = true
    aliveRef.current = () => { alive = false }
    // Watchdog: if a single inference takes longer than this, assume the backend
    // (typically WebGPU on certain laptop GPU/driver combos) has stalled and
    // release the lock so subsequent frames can try again. Without this the page
    // would stay frozen until refresh.
    const STALL_LIMIT_MS = 9000

    const tick = () => {
      if (!alive) return
      const v = videoRef.current
      if (v && v.readyState >= 2 && v.videoWidth > 0) {
        const now = performance.now()
        const f = fpsRef.current
        f.n++
        if (now - f.t >= 1000) {
          setFps(Math.round((f.n * 1000) / (now - f.t)))
          f.n = 0
          f.t = now
        }
        drawOverlay(latestBoxes, v.videoWidth, v.videoHeight)

        if (inferring && now - inferStartedAt > STALL_LIMIT_MS) {
          console.warn(`inference stalled >${STALL_LIMIT_MS}ms; releasing watchdog`)
          inferring = false
        }

        // Kick inference (non-blocking). If still running from previous frame, skip.
        if (!inferring) {
          inferring = true
          inferStartedAt = now
          const t0 = performance.now()
          detect(v, v.videoWidth, v.videoHeight, { confThreshold: confRef.current })
            .then(det => {
              if (!alive) return
              latestBoxes = det
              setBoxes(det)
              setLatency(Math.round(performance.now() - t0))
            })
            .catch(e => console.error('detect failed:', e))
            .finally(() => { inferring = false })
        }
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  function drawOverlay(det: Box[], w: number, h: number, redrawImg = false) {
    const c = canvasRef.current
    if (!c) return
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h }
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, c.width, c.height)
    if (redrawImg && lastImgRef.current) ctx.drawImage(lastImgRef.current, 0, 0)
    const lw = Math.max(2, c.width / 350)
    const fontSize = Math.max(13, Math.round(c.width / 60))
    ctx.lineWidth = lw
    ctx.font = `500 ${fontSize}px 'JetBrains Mono', monospace`
    ctx.textBaseline = 'bottom'
    for (const b of det) {
      const color = OUTFIT_COLORS[b.cls]
      ctx.strokeStyle = color
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1)
      // corner accents
      const cornerLen = Math.max(8, c.width / 80)
      ctx.beginPath()
      ctx.moveTo(b.x1, b.y1 + cornerLen); ctx.lineTo(b.x1, b.y1); ctx.lineTo(b.x1 + cornerLen, b.y1)
      ctx.moveTo(b.x2 - cornerLen, b.y1); ctx.lineTo(b.x2, b.y1); ctx.lineTo(b.x2, b.y1 + cornerLen)
      ctx.moveTo(b.x1, b.y2 - cornerLen); ctx.lineTo(b.x1, b.y2); ctx.lineTo(b.x1 + cornerLen, b.y2)
      ctx.moveTo(b.x2 - cornerLen, b.y2); ctx.lineTo(b.x2, b.y2); ctx.lineTo(b.x2 - cornerLen, b.y2)
      ctx.lineWidth = lw * 1.6
      ctx.stroke()
      ctx.lineWidth = lw

      const label = `${OUTFIT_LABELS[b.cls]} ${(b.score * 100).toFixed(0)}%`
      const padX = 8, padY = 5
      const tw = ctx.measureText(label).width + padX * 2
      const th = fontSize + padY * 2
      ctx.fillStyle = color
      ctx.fillRect(b.x1, Math.max(0, b.y1 - th), tw, th)
      ctx.fillStyle = '#0a0a0a'
      ctx.fillText(label, b.x1 + padX, Math.max(th, b.y1) - padY)
    }
  }

  async function processBlob(f: Blob) {
    try {
      setStatus('loading')
      setStatusMsg('loading model')
      const be = await loadModel(m => setStatusMsg(m.toLowerCase()))
      setBackend(be)
      stopAll()
      const url = URL.createObjectURL(f)
      const img = new Image()
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url })
      lastImgRef.current = img
      const c = canvasRef.current!
      c.width = img.naturalWidth
      c.height = img.naturalHeight
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      setStatusMsg('detecting')
      const t0 = performance.now()
      const det = await detect(img, img.naturalWidth, img.naturalHeight, { confThreshold: confRef.current })
      setBoxes(det)
      drawOverlay(det, img.naturalWidth, img.naturalHeight, true)
      setLatency(Math.round(performance.now() - t0))
      setFps(0)
      setStatus('paused')
      setStatusMsg(det.length ? `found ${det.length} item${det.length > 1 ? 's' : ''}` : 'no items detected')
    } catch (err) {
      console.error(err)
      setStatus('error')
      setStatusMsg(err instanceof Error ? err.message : 'unknown error')
    }
  }

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    try { await processBlob(f) } finally { e.target.value = '' }
  }

  // Ctrl/Cmd+V paste an image from clipboard. Skips when user is typing in an input.
  useEffect(() => {
    const onPaste = (ev: ClipboardEvent) => {
      const t = ev.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      const items = ev.clipboardData?.items
      if (!items) return
      for (const it of items) {
        if (it.kind === 'file' && it.type.startsWith('image/')) {
          const blob = it.getAsFile()
          if (blob) {
            ev.preventDefault()
            processBlob(blob)
            return
          }
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [])

  const activeClasses = new Set(boxes.map(b => b.cls))

  return (
    <div className="shell">
      <div className="topbar">
        <div className="brand">
          <span className="glyph">Y</span>
          OUTFIT&nbsp;DETECTOR <span className="v mono">v8n</span>
        </div>
        <nav>
          <a href="https://github.com/Rieltzx25/outfit-detection-yolov8" target="_blank" rel="noreferrer">github</a>
          <a href="#info">about</a>
        </nav>
      </div>

      <div className="hero">
        <h1>Detect what people <span className="accent">wear</span><br />in real time.</h1>
        <div className="tagline">
          <span>YOLOv8n / 7 classes</span>
          <span>browser-only · no upload</span>
        </div>
      </div>
      <p className="lede">
        A YOLOv8 model trained on 7 clothing categories, running fully on-device. Point your
        camera, upload a photo — bounding boxes are drawn in your browser using ONNX Runtime Web.
        Nothing is sent anywhere.
      </p>

      <div className="layout">
        <div className={`card ${status === 'live' ? 'live' : ''}`}>
          <div className={`stage ${status === 'live' ? 'live' : ''}`}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ visibility: status === 'live' ? 'visible' : 'hidden' }}
            />
            <canvas ref={canvasRef} />
            <div className="scanline" />
            {status !== 'live' && status !== 'paused' && (
              <div className="empty">
                <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4">
                  <path d="M4 7h16M4 12h16M4 17h10" />
                </svg>
                <span className="mono" style={{ fontSize: 12, letterSpacing: '0.04em' }}>
                  {status === 'loading' ? statusMsg.toUpperCase() : 'STREAM OFFLINE'}
                </span>
              </div>
            )}
          </div>
          <div className="toolbar">
            {status !== 'live' ? (
              <button className="btn primary" onClick={start} disabled={status === 'loading'}>
                {status === 'loading' ? '⋯ loading' : '▶ start camera'}
              </button>
            ) : (
              <button className="btn" onClick={stop}>■ stop</button>
            )}
            <label className="btn" style={{ cursor: 'pointer' }}>
              ↑ upload
              <input type="file" accept="image/*" onChange={handleFile} />
            </label>
            <button
              className="btn"
              type="button"
              onClick={async () => {
                const next = facing === 'user' ? 'environment' : 'user'
                setFacing(next)
                if (status === 'live') { stop(); setTimeout(start, 60) }
              }}
              title="switch camera"
            >
              ⇄ {facing === 'user' ? 'front' : 'back'}
            </button>
            <div className="status-line">
              <span className={`dot ${status === 'live' ? 'live' : status === 'loading' ? 'loading' : status === 'error' ? 'error' : ''}`} />
              <span>{statusMsg}</span>
            </div>
          </div>
        </div>

        <aside className="side">
          <div className="card panel">
            <h3>Tuning</h3>
            <div className="setting">
              <label>
                <span>min confidence</span>
                <span className="mono" style={{ color: 'var(--muted)' }}>{confThreshold.toFixed(2)}</span>
              </label>
              <input
                type="range" min="0.1" max="0.9" step="0.05"
                value={confThreshold}
                onChange={e => setConfThreshold(parseFloat(e.target.value))}
              />
            </div>
          </div>

          <div className="card panel">
            <h3>Telemetry</h3>
            <div className="metrics">
              <div className="metric">
                <div className="label">FPS</div>
                <div className="value">{fps}<span className="unit">/s</span></div>
              </div>
              <div className="metric">
                <div className="label">Latency</div>
                <div className="value">{latency}<span className="unit">ms</span></div>
              </div>
              <div className="metric">
                <div className="label">Items</div>
                <div className="value">{boxes.length}</div>
              </div>
              <div className="metric">
                <div className="label">Backend</div>
                <div className="value" style={{ fontSize: 14 }}>{backend.toUpperCase()}</div>
              </div>
            </div>
          </div>

          <div className="card panel">
            <h3>Classes <span style={{ color: 'var(--muted)', fontWeight: 400 }}>· {OUTFIT_LABELS.length}</span></h3>
            <div className="legend-groups">
              {OUTFIT_GROUPS.map(group => (
                <div key={group.name} className="legend-group">
                  <div className="legend-group-name">{group.name}</div>
                  <div className="legend">
                    {group.indices.map(i => (
                      <span key={i} className={`chip ${activeClasses.has(i) ? 'active' : ''}`}>
                        <span className="swatch" style={{ background: OUTFIT_COLORS[i] }} />
                        {OUTFIT_LABELS[i]}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card panel">
            <h3>Detections</h3>
            {boxes.length === 0 ? (
              <div className="empty-side">No items in frame.</div>
            ) : (
              <div className="detlist">
                {boxes.map((b, i) => (
                  <div className="det" key={i}>
                    <span className="swatch" style={{ background: OUTFIT_COLORS[b.cls] }} />
                    <span className="name">{OUTFIT_LABELS[b.cls]}</span>
                    <span className="conf">{(b.score * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>

      <footer id="info">
        <span>YOLOv8n · 640×640 · ONNX FP32 · 12.3 MB</span>
        <span>
          model: <a href="https://github.com/Rieltzx25/outfit-detection-yolov8" target="_blank" rel="noreferrer">Rieltzx25/outfit-detection-yolov8</a>
        </span>
      </footer>
    </div>
  )
}
