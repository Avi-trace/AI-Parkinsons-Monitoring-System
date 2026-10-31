/**
 * Shared GPU hand tracking (MediaPipe Tasks HandLandmarker) + webcam.
 *
 * Replaces the deprecated @mediapipe/hands CDN bundle that every test loaded on
 * its own. One camera stream and one landmarker per hand-count are shared by all
 * tests and released when no test is using them.
 *
 *  - GPU delegate (WebGL) first, automatic CPU fallback.
 *  - Frames are driven by requestVideoFrameCallback: one inference per real
 *    camera frame, with that frame's capture timestamp (not "when the callback
 *    happened to run"), so the backend gets accurate sample times.
 *  - Dynamic fps: the camera is asked for 60 fps; an adaptive controller steps
 *    the capture resolution down when inference can't keep up and back up when
 *    there is headroom. The measured fps is reported with every recording.
 */
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// Pinned to the installed package version. `npm install` copies the wasm files and
// downloads the model into /public (scripts/copy-mediapipe-assets.mjs); the CDN
// URLs are only a fallback if those local files are missing.
const TASKS_VERSION = "0.10.35";
const LOCAL_WASM = "/mediapipe/wasm";
const LOCAL_MODEL = "/models/hand_landmarker.task";
const CDN_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/wasm`;
const CDN_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Capture resolutions the adaptive controller moves between (highest first).
// More pixels = less landmark jitter, which matters for small tremors.
export const RESOLUTION_LADDER = [
  { width: 1280, height: 720 },
  { width: 960, height: 540 },
  { width: 640, height: 480 },
  { width: 480, height: 360 },
];
const START_LEVEL = 0;
const TARGET_FPS = 60;

async function exists(url) {
  try {
    const r = await fetch(url, { method: "HEAD" });
    // Vite's dev server answers unknown paths with index.html (200), so a 200
    // with an HTML content type means "missing".
    return r.ok && !(r.headers.get("content-type") || "").includes("text/html");
  } catch {
    return false;
  }
}

// ── Landmarker (cached per numHands) ─────────────────────────────────────────
let visionPromise = null;
const landmarkers = new Map(); // numHands -> Promise<{ landmarker, delegate }>

async function getVision() {
  if (!visionPromise) {
    visionPromise = (async () => {
      const local = await exists(`${LOCAL_WASM}/vision_wasm_internal.js`);
      return FilesetResolver.forVisionTasks(local ? LOCAL_WASM : CDN_WASM);
    })();
  }
  return visionPromise;
}

export function getHandLandmarker(numHands = 1) {
  if (!landmarkers.has(numHands)) {
    landmarkers.set(numHands, (async () => {
      const vision = await getVision();
      const modelAssetPath = (await exists(LOCAL_MODEL)) ? LOCAL_MODEL : CDN_MODEL;
      const options = (delegate) => ({
        baseOptions: { modelAssetPath, delegate },
        runningMode: "VIDEO",
        numHands,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      try {
        return { landmarker: await HandLandmarker.createFromOptions(vision, options("GPU")), delegate: "GPU" };
      } catch (err) {
        console.warn("[handTracker] GPU delegate unavailable, falling back to CPU:", err);
        return { landmarker: await HandLandmarker.createFromOptions(vision, options("CPU")), delegate: "CPU" };
      }
    })());
    // Don't cache failures: a later attempt (e.g. after a reload of assets) can retry.
    landmarkers.get(numHands).catch(() => landmarkers.delete(numHands));
  }
  return landmarkers.get(numHands);
}

// ── Shared camera (reference counted) ────────────────────────────────────────
let camera = null; // { stream, users, level }

export async function acquireCamera() {
  if (camera) {
    camera.users += 1;
    return camera;
  }
  const { width, height } = RESOLUTION_LADDER[START_LEVEL];
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: width },
      height: { ideal: height },
      frameRate: { ideal: TARGET_FPS, min: 15 },
      resizeMode: "none",
    },
  });
  camera = { stream, users: 1, level: START_LEVEL };
  return camera;
}

export function releaseCamera() {
  if (!camera) return;
  camera.users -= 1;
  if (camera.users <= 0) {
    camera.stream.getTracks().forEach((t) => t.stop());
    camera = null;
  }
}

async function setResolution(level) {
  if (!camera) return false;
  const track = camera.stream.getVideoTracks()[0];
  const { width, height } = RESOLUTION_LADDER[level];
  try {
    await track.applyConstraints({ width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: TARGET_FPS } });
    camera.level = level;
    return true;
  } catch (err) {
    console.warn("[handTracker] could not change resolution:", err);
    return false;
  }
}

export function cameraInfo() {
  if (!camera) return null;
  const s = camera.stream.getVideoTracks()[0]?.getSettings?.() || {};
  return { width: s.width, height: s.height, frameRate: s.frameRate, label: camera.stream.getVideoTracks()[0]?.label };
}

// ── Frame loop with adaptive resolution ──────────────────────────────────────
/**
 * Starts the per-frame loop. `onFrame(result, frameTimeMs, video)` is called once
 * per processed camera frame; `onStats(stats)` about once per second.
 * Returns { stop(), lock(bool) }. Lock adaptation while recording: a resolution
 * change mid-test would put a step into the pixel coordinates.
 */
export function startTracking(video, landmarker, { onFrame, onStats, adaptive = true } = {}) {
  let stopped = false;
  let handle = null;
  let clock = null;          // "capture" | "media" | "perf": chosen on the first frame, then fixed
  let lastTs = -1;
  let windowStart = performance.now();
  let processed = 0;
  let presented = 0;
  let inferSum = 0;
  let slowWindows = 0;
  let fastWindows = 0;
  let changing = false;
  let locked = false;
  const useRVFC = typeof video.requestVideoFrameCallback === "function";

  const frameTime = (now, meta) => {
    if (!clock) clock = meta?.captureTime ? "capture" : meta?.mediaTime != null ? "media" : "perf";
    if (clock === "capture") return meta.captureTime;
    if (clock === "media") return meta.mediaTime * 1000;
    return now;
  };

  const step = (now, meta) => {
    if (stopped) return;
    presented += 1;
    if (video.readyState >= 2 && video.videoWidth > 0) {
      let ts = frameTime(now, meta);
      if (ts <= lastTs) ts = lastTs + 0.001; // detectForVideo needs strictly increasing timestamps
      lastTs = ts;
      const t0 = performance.now();
      let result = null;
      try {
        result = landmarker.detectForVideo(video, ts);
      } catch (err) {
        console.warn("[handTracker] detectForVideo failed:", err);
      }
      inferSum += performance.now() - t0;
      processed += 1;
      if (result) onFrame?.(result, ts, video);
    }

    const elapsed = performance.now() - windowStart;
    if (elapsed >= 1000) {
      const fps = (processed * 1000) / elapsed;
      const cameraFps = (presented * 1000) / elapsed;
      const inferenceMs = processed ? inferSum / processed : 0;
      onStats?.({ fps, cameraFps, inferenceMs, level: camera?.level ?? null, ...cameraInfo() });
      if (adaptive && camera && !changing && !locked) adapt(fps, cameraFps, inferenceMs);
      windowStart = performance.now();
      processed = presented = 0;
      inferSum = 0;
    }
    schedule();
  };

  // Step down when we drop >20 % of camera frames or inference eats most of the
  // frame budget for 2 s; step up after 5 s of comfortable headroom.
  const adapt = (fps, cameraFps, inferenceMs) => {
    const budget = 1000 / Math.max(cameraFps, 1);
    const struggling = fps < cameraFps * 0.8 || inferenceMs > budget * 0.85;
    const comfortable = fps >= cameraFps * 0.95 && inferenceMs < budget * 0.4;
    slowWindows = struggling ? slowWindows + 1 : 0;
    fastWindows = comfortable ? fastWindows + 1 : 0;
    let next = null;
    if (slowWindows >= 2 && camera.level < RESOLUTION_LADDER.length - 1) next = camera.level + 1;
    else if (fastWindows >= 5 && camera.level > 0) next = camera.level - 1;
    if (next !== null) {
      changing = true;
      slowWindows = fastWindows = 0;
      setResolution(next).finally(() => { changing = false; });
    }
  };

  const schedule = () => {
    if (stopped) return;
    handle = useRVFC ? video.requestVideoFrameCallback(step) : requestAnimationFrame((now) => step(now, null));
  };

  schedule();
  return {
    stop() {
      stopped = true;
      if (handle !== null) {
        if (useRVFC) video.cancelVideoFrameCallback?.(handle);
        else cancelAnimationFrame(handle);
      }
    },
    lock(value) {
      locked = Boolean(value);
      slowWindows = fastWindows = 0;
    },
  };
}

export const LANDMARK = { WRIST: 0, THUMB_TIP: 4, INDEX_MCP: 5, INDEX_TIP: 8, MIDDLE_MCP: 9, MIDDLE_TIP: 12, PINKY_MCP: 17 };
