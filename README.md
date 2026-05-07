# outfit detector

Real-time clothing detection running fully in your browser. Point your webcam, paste a photo, or upload one — bounding boxes are drawn locally, no frames leave your device.

🔗 **[outfit-detection-deploy-jade.vercel.app](https://outfit-detection-deploy-jade.vercel.app)**

## what it does

35-class YOLOv8s detector for clothing items (tops, outerwear, bottoms, footwear, etc.) plus a separate shoe specialist that overrides the main model on footwear regions for better accuracy.

Runs entirely in-browser via ONNX Runtime Web. Inference happens in a Web Worker so the video stays smooth while the model is thinking.

## stack

- React + TypeScript + Vite
- onnxruntime-web (multi-threaded WASM, 8 threads)
- YOLOv8s — 35 classes — ONNX FP32 — 45 MB
- Shoe specialist — single class — 12 MB
- Hosted on Vercel with COOP/COEP headers for SharedArrayBuffer

## run locally

```bash
npm install
npm run dev
```

## controls

- **Start camera** — webcam preview + live detection
- **Upload** — pick an image from disk
- **Ctrl/Cmd + V** — paste an image from clipboard
- **Confidence slider** — minimum score for a box to render
- **Backend** — auto WebGPU on supported hardware, otherwise multi-threaded WASM

## links

- Source: [Rieltzx25/outfit-app](https://github.com/Rieltzx25/outfit-app)
- Model: [Rieltzx25/outfit-detection-yolov8](https://github.com/Rieltzx25/outfit-detection-yolov8)
