// Copies the MediaPipe Tasks wasm runtime into public/ and downloads the hand
// landmarker model once, so the app serves them itself instead of depending on a
// CDN at runtime. Runs automatically after `npm install` (postinstall).
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const wasmSrc = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const wasmDst = join(root, "public", "mediapipe", "wasm");
const modelDst = join(root, "public", "models", "hand_landmarker.task");
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

if (existsSync(wasmSrc)) {
  mkdirSync(wasmDst, { recursive: true });
  for (const f of readdirSync(wasmSrc)) copyFileSync(join(wasmSrc, f), join(wasmDst, f));
  console.log(`[mediapipe] copied wasm runtime -> ${wasmDst}`);
} else {
  console.warn("[mediapipe] @mediapipe/tasks-vision not installed; the app will fall back to the CDN");
}

if (existsSync(modelDst) && statSync(modelDst).size > 1_000_000) {
  console.log("[mediapipe] hand_landmarker.task already present");
} else {
  try {
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    mkdirSync(dirname(modelDst), { recursive: true });
    writeFileSync(modelDst, Buffer.from(await res.arrayBuffer()));
    console.log(`[mediapipe] downloaded hand landmarker model -> ${modelDst}`);
  } catch (err) {
    console.warn(`[mediapipe] could not download the model (${err.message}); the app will load it from ${MODEL_URL}`);
  }
}
