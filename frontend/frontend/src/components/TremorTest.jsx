/**
 * NeuroTrack AI — Postural Tremor Test
 * Webcam + MediaPipe Hands, both hands simultaneously, 15s hold.
 *
 * Fixes vs previous version:
 *  1. Left/Right label swap — canvas is mirrored (scaleX:-1) so MediaPipe's
 *     "Left" label appears on the right side of screen. We flip the assignment:
 *     MediaPipe "Left" → stored as right_hand, "Right" → stored as left_hand.
 *  2. Composite landmark tracking — instead of wrist only (landmark[0]),
 *     we use the mean of wrist(0) + middle MCP(9) + middle tip(12).
 *     This captures both proximal wrist tremor AND distal finger tremor.
 *  3. Amplitude noise floor lowered (backend) — 5px → 2px so small
 *     distal tremors at arm's length are not filtered out.
 *  4. NEW — Per-hand fps tracking. The live fps badge used to show
 *     leftFrames.length + rightFrames.length combined, which can look
 *     like "18fps" while each hand individually is only getting ~9fps —
 *     right around the backend's reliability floor for frequency
 *     detection. Now each hand's fps is tracked and shown separately.
 *  5. NEW — minTrackingConfidence lowered 0.5 → 0.4. Fast/deliberate
 *     shaking causes motion blur, which drops MediaPipe's per-frame
 *     tracking confidence. At 0.5 this was triggering full re-detection
 *     (slower than steady tracking) more often during exactly the motion
 *     we're trying to measure, dragging per-hand fps down further. 0.4
 *     tolerates brief confidence dips without forcing a re-detect, while
 *     minDetectionConfidence (used for the more expensive initial/re-
 *     detection pass) is left untouched.
 *  6. NEW — pre-submit per-hand fps diagnostic log, so a low-fps cause
 *     of "Not detected" results is visible in the console instead of
 *     only showing up as an opaque result.
 *  7. NEW — result persists across page refresh via localStorage, keyed
 *     per patientId, so reloading the page doesn't wipe the last score.
 *  8. NEW — during "recording", skip the expensive per-connection
 *     skeleton drawing every other frame (data capture is unaffected —
 *     every frame is still recorded). Canvas drawing runs synchronously
 *     inside the MediaPipe onResults callback, so its cost was eating
 *     into the time available before the next frame can be processed,
 *     directly capping per-hand fps. This is the main lever for getting
 *     more hands above the ~9fps reliable-frequency floor.
 *
 * MediaPipe loaded from CDN (not npm) — see vite.config.js for exclusion.
 */

import { useState, useRef, useEffect, useCallback } from "react";

const C = {
  bg: "#0d1117", surface: "#161b22", surfaceHigh: "#1c2128",
  border: "#30363d", borderSoft: "#21262d",
  text: "#e6edf3", textSub: "#8b949e", textMuted: "#484f58",
  accent: "#2f81f7", green: "#3fb950", yellow: "#d29922",
  orange: "#f0883e", red: "#f85149",
  greenDim: "#122c1a", yellowDim: "#2d1f07", orangeDim: "#2d1505", redDim: "#2c0d0d",
};
const SEVERITY = {
  Normal:   { color: C.green,  dim: C.greenDim  },
  Mild:     { color: C.yellow, dim: C.yellowDim },
  Moderate: { color: C.orange, dim: C.orangeDim },
  Severe:   { color: C.red,    dim: C.redDim    },
};
const TEST_DURATION_S = 15;
const MIN_RELIABLE_FPS = 9; // must match backend MIN_FREQ_HEADROOM assumptions

// FIX 7: persistence helpers
const storageKey = (patientId) => `neurotrack_last_result_${patientId}`;
function loadStoredResult(patientId) {
  try {
    const raw = localStorage.getItem(storageKey(patientId));
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}
function saveStoredResult(patientId, data) {
  try {
    localStorage.setItem(storageKey(patientId), JSON.stringify(data));
  } catch {
    // storage unavailable (private browsing, quota, etc) — fail silently,
    // the in-memory result for this session still works fine.
  }
}
function clearStoredResult(patientId) {
  try {
    localStorage.removeItem(storageKey(patientId));
  } catch {}
}

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
];

// Landmark indices
const WRIST      = 0;
const MIDDLE_MCP = 9;
const MIDDLE_TIP = 12;
// Hand size reference: wrist → middle MCP distance (px)
// Used by backend to normalize amplitude independent of camera distance

function loadScript(src) {
  return new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) { res(); return; }
    const s = Object.assign(document.createElement("script"), { src, crossOrigin: "anonymous" });
    s.onload = res; s.onerror = () => rej(new Error(`Failed: ${src}`));
    document.head.appendChild(s);
  });
}
let mpPromise = null;
function ensureMP() {
  return mpPromise || (mpPromise = Promise.all([
    loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"),
    loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"),
  ]));
}

const fmt = (n, d = 1) => typeof n === "number" ? n.toFixed(d) : "—";

function Badge({ label, severity }) {
  const cfg = SEVERITY[severity] || { color: C.textSub, dim: C.surface };
  return (
    <span style={{
      fontSize: 12, fontWeight: 500, padding: "3px 10px", borderRadius: 6, display: "inline-block",
      background: cfg.dim, color: cfg.color, border: `1px solid ${cfg.color}33`,
    }}>{label}</span>
  );
}

function StabilityCard({ side, scores }) {
  const sev = scores?.classification || "Normal";
  const detected = scores?.detected;
  return (
    <div style={{ background: C.surfaceHigh, border: `1px solid ${C.borderSoft}`, borderRadius: 10, padding: "16px 18px", flex: 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <span style={{ fontSize: 12, color: C.textSub, textTransform: "uppercase", letterSpacing: "0.04em" }}>{side} hand</span>
        {detected && <Badge label={sev} severity={sev} />}
      </div>
      {!detected ? (
        <p style={{ fontSize: 13, color: C.textMuted, margin: 0 }}>Not detected</p>
      ) : (
        <>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginBottom: 12 }}>
            <span style={{ fontSize: 32, fontWeight: 600, color: C.text, fontVariantNumeric: "tabular-nums" }}>{scores.stability_score}</span>
            <span style={{ fontSize: 13, color: C.textMuted }}>/ 100 stable</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 12 }}>
            {[
              ["Frequency",       scores.tremor_frequency_hz > 0 ? `${fmt(scores.tremor_frequency_hz)} Hz` : "n/a"],
              ["PD band power",   `${Math.round(scores.tremor_power_ratio * 100)}%`],
              ["Amplitude",       `${Math.round((scores.amplitude_norm ?? 0) * 100)}% hand`],
              ["RMS velocity",    `${fmt(scores.rms_velocity, 0)} px/s`],
              ["Velocity irreg.", `${Math.round((scores.velocity_cv ?? 0) * 100)}%`],
              ["Hand size ref",   `${fmt(scores.hand_size_px, 0)} px`],
            ].map(([label, val]) => (
              <div key={label}>
                <p style={{ color: C.textMuted, margin: "0 0 2px" }}>{label}</p>
                <p style={{ color: C.text, margin: 0, fontWeight: 500 }}>{val}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default function TremorTest({ patientId = "anonymous", apiBase = "", onResult = null }) {
  const videoRef     = useRef(null);
  const canvasRef    = useRef(null);
  const handsRef     = useRef(null);
  const cameraRef    = useRef(null);
  const timerRef     = useRef(null);
  const onResultsRef = useRef(null);

  // ── FIX 1: swap key names so "left on screen" = left_hand in data ──────
  // Canvas is mirrored (scaleX:-1). MediaPipe sees unmirrored feed and labels
  // the hand on its left as "Left". After mirroring, that hand appears on
  // screen's RIGHT. So we must swap: MP "Left" → our right, MP "Right" → our left.
  const leftFrames   = useRef([]);   // patient's physical left hand
  const rightFrames  = useRef([]);   // patient's physical right hand

  const recordingRef  = useRef(false);
  const startTimeRef  = useRef(null);
  const isBusyRef     = useRef(false);  // throttle: skip frame if MediaPipe still processing

  const [phase, setPhase]               = useState("idle");
  const [scriptsReady, setScriptsReady] = useState(false);
  const [handsVisible, setHandsVisible] = useState({ left: false, right: false });
  const [countdown, setCountdown]       = useState(3);
  const [timeLeft, setTimeLeft]         = useState(TEST_DURATION_S);
  // FIX 4: per-hand fps instead of one combined number
  const [liveFps, setLiveFps]           = useState({ left: 0, right: 0 });
  // FIX 7: seed from localStorage so a refresh doesn't wipe the last score.
  const [result, setResult]             = useState(() => loadStoredResult(patientId));
  const [error, setError]               = useState(null);
  // FIX 4: separate counters per hand
  const fpsCounterRef = useRef({ left: { count: 0, last: 0 }, right: { count: 0, last: 0 } });
  // FIX 8: phase mirrored into a ref so the onResults callback (defined
  // once, not re-created per render) can cheaply check "are we recording"
  // without depending on stale closure state.
  const phaseRef  = useRef(phase);
  const drawTickRef = useRef(0);

  useEffect(() => { phaseRef.current = phase; }, [phase]);

  useEffect(() => {
    ensureMP().then(() => setScriptsReady(true)).catch(console.warn);
    return () => { clearInterval(timerRef.current); cameraRef.current?.stop(); handsRef.current?.close(); };
  }, []);

  // FIX 7: if a result already exists on mount (restored from storage),
  // jump straight to the "done" view instead of idle.
  useEffect(() => {
    if (result) setPhase("done");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── FIX 2 + 3: composite landmark tracking ───────────────────────────
  onResultsRef.current = useCallback((results) => {
    const canvas = canvasRef.current, video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    ctx.save(); ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Mirror only the video feed — NOT the whole canvas.
      // This way labels, dots, and text render correctly (not flipped).
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(results.image, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

    let sawLeft = false, sawRight = false;

    // FIX 8: during recording, only do the expensive per-connection/dot
    // drawing on every other frame. This is purely a rendering decision —
    // frame *data capture* below is untouched and still runs every frame.
    // Canvas drawing happens synchronously inside this callback, so its
    // cost was directly delaying when `finally()` fires and the next
    // frame can be sent to MediaPipe, capping achievable fps right when
    // fast motion needs it most.
    drawTickRef.current++;
    const isRecording = phaseRef.current === "recording";
    const doFullDraw = !isRecording || (drawTickRef.current % 2 === 0);

    results.multiHandLandmarks?.forEach((lm, i) => {
      const mpLabel = results.multiHandedness[i].label; // MediaPipe's label (unmirrored)

      // FIX 1: because canvas is mirrored, swap MP Left↔Right to get screen-correct label
      // MP "Left"  → physically on screen's right → store as rightFrames
      // MP "Right" → physically on screen's left  → store as leftFrames
      const physicalSide = mpLabel === "Left" ? "right" : "left";
      const color = physicalSide === "left" ? C.green : C.accent;

      // FIX 2: composite hand position = mean of wrist + middle MCP + middle tip
      // This captures finger tremor that wrist-only tracking misses
      const trackPoints = [WRIST, MIDDLE_MCP, MIDDLE_TIP];
      // Mirror x coordinate to match the flipped video feed
      const cx_raw = trackPoints.reduce((s, idx) => s + lm[idx].x, 0) / trackPoints.length;
      const cy_raw = trackPoints.reduce((s, idx) => s + lm[idx].y, 0) / trackPoints.length;
      const px = (1 - cx_raw) * canvas.width;   // flip x
      const py = cy_raw * canvas.height;

      if (doFullDraw) {
        // Draw skeleton
        ctx.strokeStyle = color + "88"; ctx.lineWidth = 1.5;
        HAND_CONNECTIONS.forEach(([a, b]) => {
          ctx.beginPath();
          ctx.moveTo((1 - lm[a].x) * canvas.width, lm[a].y * canvas.height);
          ctx.lineTo((1 - lm[b].x) * canvas.width, lm[b].y * canvas.height);
          ctx.stroke();
        });

        // Draw tracked landmarks (mirrored x)
        trackPoints.forEach(idx => {
          ctx.beginPath();
          ctx.arc((1 - lm[idx].x) * canvas.width, lm[idx].y * canvas.height, 5, 0, Math.PI * 2);
          ctx.fillStyle = color + "bb"; ctx.fill();
        });

        // Draw centroid
        ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();

        // Label — show physical side so it matches what user sees on mirrored screen
        ctx.fillStyle = "#fff"; ctx.font = "bold 12px system-ui";
        ctx.fillText(physicalSide.charAt(0).toUpperCase() + physicalSide.slice(1), px + 10, py - 8);
      } else {
        // Lightweight marker only — keeps some visual feedback without
        // the full skeleton draw cost.
        ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
      }

      if (physicalSide === "left")  sawLeft  = true;
      if (physicalSide === "right") sawRight = true;

      // Record composite position.
      // z = wrist→middleMCP distance (px) used by backend for distance normalization.
      // This makes amplitude independent of how far the hand is from the camera.
      if (recordingRef.current && startTimeRef.current !== null) {
        const wristX = lm[WRIST].x * canvas.width,  wristY = lm[WRIST].y * canvas.height;
        const mcpX   = lm[MIDDLE_MCP].x * canvas.width, mcpY = lm[MIDDLE_MCP].y * canvas.height;
        const handSizePx = Math.hypot(wristX - mcpX, wristY - mcpY);
        const frame = {
          x: px, y: py,
          z: handSizePx,   // hand size in px — backend divides amplitude by this
          t: performance.now() - startTimeRef.current,
        };
        if (physicalSide === "left")  leftFrames.current.push(frame);
        if (physicalSide === "right") rightFrames.current.push(frame);

        // FIX 4: update live fps per-hand, not combined. A combined number
        // can read "18fps" while one hand individually is only at ~9fps —
        // right at the backend's reliability floor — and hide the problem.
        const now = performance.now();
        const counter = fpsCounterRef.current[physicalSide];
        counter.count++;
        if (now - counter.last >= 1000) {
          setLiveFps(prev => ({ ...prev, [physicalSide]: counter.count }));
          fpsCounterRef.current[physicalSide] = { count: 0, last: now };
        }
      }
    });

    ctx.restore();
    setHandsVisible({ left: sawLeft, right: sawRight });
  }, []);

  const startCamera = useCallback(async () => {
    setPhase("camera_loading"); setError(null);
    try {
      await ensureMP();
      const hands = new window.Hands({
        locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
      });
      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 0,             // 0 = lite model, much faster (~2x)
        minDetectionConfidence: 0.65,
        // FIX 5: lowered 0.5 → 0.4. Fast/deliberate shaking causes motion
        // blur, which drops per-frame tracking confidence. At 0.5 that was
        // triggering full re-detection passes (slower than steady tracking)
        // more often during exactly the motion we're measuring, which drags
        // per-hand fps down below the backend's ~9fps reliability floor.
        minTrackingConfidence: 0.4,
      });
      hands.onResults(r => onResultsRef.current(r));
      handsRef.current = hands;

      const cam = new window.Camera(videoRef.current, {
        onFrame: () => {
          // Skip frame if MediaPipe is still processing the previous one.
          // This prevents queue buildup while still running as fast as possible.
          if (isBusyRef.current || !handsRef.current) return;
          isBusyRef.current = true;
          handsRef.current.send({ image: videoRef.current })
            .catch(() => {})
            .finally(() => { isBusyRef.current = false; });
        },
        width: 320,
        height: 240,
        facingMode: "user",
      });
      await cam.start(); cameraRef.current = cam;
      setPhase("ready");
    } catch (err) {
      setError("Could not start camera. Check browser permissions."); setPhase("error");
    }
  }, []);

  function beginTest() {
    leftFrames.current = []; rightFrames.current = [];
    fpsCounterRef.current = { left: { count: 0, last: 0 }, right: { count: 0, last: 0 } };
    setLiveFps({ left: 0, right: 0 });
    setResult(null); setError(null);
    setCountdown(3); setPhase("countdown");
    let c = 3;
    const cd = setInterval(() => {
      c--; setCountdown(c);
      if (c <= 0) { clearInterval(cd); startRecording(); }
    }, 1000);
  }

  function startRecording() {
    setPhase("recording"); setTimeLeft(TEST_DURATION_S);
    startTimeRef.current = performance.now();
    recordingRef.current = true;
    let t = TEST_DURATION_S;
    timerRef.current = setInterval(() => {
      t--; setTimeLeft(t);
      if (t <= 0) { clearInterval(timerRef.current); recordingRef.current = false; submit(); }
    }, 1000);
  }

  async function submit() {
    setPhase("analyzing");
    try {
      const lf = leftFrames.current, rf = rightFrames.current;

      // FIX 6: per-hand fps diagnostic before sending, so a low-fps cause
      // of "Not detected" is visible in the console instead of only
      // showing up as an opaque backend result.
      const fpsOf = (frames) => {
        if (frames.length < 2) return 0;
        const durS = (frames[frames.length - 1].t - frames[0].t) / 1000;
        return durS > 0 ? frames.length / durS : 0;
      };
      const lFps = fpsOf(lf), rFps = fpsOf(rf);
      console.log(
        `[TremorTest] Left: ${lf.length} frames @ ${lFps.toFixed(1)}fps` +
        `${lFps > 0 && lFps < MIN_RELIABLE_FPS ? " ⚠️ below reliable-frequency floor" : ""}` +
        ` | Right: ${rf.length} frames @ ${rFps.toFixed(1)}fps` +
        `${rFps > 0 && rFps < MIN_RELIABLE_FPS ? " ⚠️ below reliable-frequency floor" : ""}`
      );
      if (lf.length > 10) {
        const lxs = lf.map(f => f.x), lys = lf.map(f => f.y);
        console.log("[TremorTest] Left X range:", Math.min(...lxs).toFixed(1), "–", Math.max(...lxs).toFixed(1),
          "| Y range:", Math.min(...lys).toFixed(1), "–", Math.max(...lys).toFixed(1));
      }
      if (rf.length > 10) {
        const rxs = rf.map(f => f.x), rys = rf.map(f => f.y);
        console.log("[TremorTest] Right X range:", Math.min(...rxs).toFixed(1), "–", Math.max(...rxs).toFixed(1),
          "| Y range:", Math.min(...rys).toFixed(1), "–", Math.max(...rys).toFixed(1));
      }

      const payload = {
        patient_id: patientId, duration_s: TEST_DURATION_S,
        left_hand:  lf.length >= 30 ? lf  : null,
        right_hand: rf.length >= 30 ? rf  : null,
      };
      if (!payload.left_hand && !payload.right_hand)
        throw new Error("No hands detected. Make sure both arms are fully visible and try again.");

      const res = await fetch(`${apiBase}/api/tremor-test`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `Server error ${res.status}`); }
      const data = await res.json();
      console.log("[TremorTest] Result — L:", data.left?.tremor_frequency_hz, "Hz score:", data.left?.stability_score,
        "| R:", data.right?.tremor_frequency_hz, "Hz score:", data.right?.stability_score,
        "| L amp:", data.left?.amplitude_px, "R amp:", data.right?.amplitude_px);
      setResult(data);
      saveStoredResult(patientId, data);   // FIX 7: survive a page refresh
      if (onResult) onResult(data);
      setPhase("done");
    } catch (err) {
      setError(err.message || "Analysis failed."); setPhase("error");
    }
  }

  function reset() {
    setResult(null); setError(null);
    clearStoredResult(patientId);
    leftFrames.current = []; rightFrames.current = [];
    // FIX 7: a restored result can exist before the camera has ever been
    // started on this page load — route to "idle" (shows Enable camera)
    // rather than "ready" (assumes camera already running) in that case.
    setPhase(cameraRef.current ? "ready" : "idle");
  }

  const canStart = handsVisible.left || handsVisible.right;
  const statusColor = { idle: scriptsReady ? C.green : C.yellow, camera_loading: C.yellow, ready: C.green, countdown: C.yellow, recording: C.red, analyzing: C.accent, done: C.green, error: C.red }[phase] || C.textSub;
  const statusLabel = { idle: scriptsReady ? "Ready" : "Loading…", camera_loading: "Starting…", ready: "Camera ready", countdown: "Get ready…", recording: `Recording · ${timeLeft}s`, analyzing: "Analysing…", done: "Complete", error: "Error" }[phase] || "—";

  return (
    <div style={{ fontFamily: "system-ui,-apple-system,sans-serif", background: C.bg, minHeight: "100vh", padding: "24px 20px", color: C.text }}>
      <div style={{ maxWidth: 700, margin: "0 auto" }}>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <div>
            <p style={{ fontSize: 11, color: C.textMuted, margin: "0 0 4px", textTransform: "uppercase", letterSpacing: "0.08em" }}>NeuroTrack AI · Motor assessment</p>
            <h1 style={{ fontSize: 22, fontWeight: 600, margin: 0 }}>Postural tremor test</h1>
            <p style={{ fontSize: 11, color: C.textMuted, margin: "4px 0 0" }}>
              Tracking: wrist · middle MCP · middle fingertip (composite)
            </p>
          </div>
          <span style={{ fontSize: 11, fontWeight: 500, padding: "4px 12px", borderRadius: 20, background: C.surfaceHigh, color: statusColor, border: `1px solid ${statusColor}44` }}>{statusLabel}</span>
        </div>

        {phase === "idle" && (
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "20px 22px", marginBottom: 16 }}>
            <p style={{ fontSize: 13, color: C.textSub, margin: "0 0 14px", lineHeight: 1.7 }}>
              Extend both arms forward toward the camera and hold still for 15 seconds.
              The camera tracks a composite of your <strong style={{ color: C.text }}>wrist, middle knuckle, and fingertip</strong> to detect both proximal and distal tremor.
            </p>
            <ol style={{ fontSize: 13, color: C.textSub, margin: "0 0 20px", paddingLeft: 20, lineHeight: 1.9 }}>
              <li>Sit so your upper body and both arms are clearly visible.</li>
              <li>Click "Enable camera".</li>
              <li>Extend both arms forward, palms facing down.</li>
              <li>Once both hands are detected, tap "Begin test".</li>
              <li>Hold as still as possible for the full 15 seconds.</li>
            </ol>
            <button onClick={startCamera} disabled={!scriptsReady} style={{ width: "100%", padding: "11px 0", borderRadius: 8, border: "none", background: scriptsReady ? C.accent : C.surfaceHigh, color: scriptsReady ? "#fff" : C.textMuted, fontSize: 14, fontWeight: 600, cursor: scriptsReady ? "pointer" : "not-allowed" }}>
              {scriptsReady ? "Enable camera" : "Loading MediaPipe…"}
            </button>
          </div>
        )}

        {phase !== "idle" && (
          <div style={{ marginBottom: 14 }}>
            <div style={{ position: "relative", borderRadius: 10, overflow: "hidden", border: `1px solid ${phase === "recording" ? C.red + "66" : C.border}` }}>
              <video ref={videoRef} style={{ display: "none" }} playsInline muted />
              <canvas
                ref={canvasRef} width={320} height={240}
                style={{ width: "100%", display: "block", background: C.surfaceHigh }}
                aria-label="Webcam feed with hand tracking"
              />

              {["ready","countdown","recording"].includes(phase) && (
                <div style={{ position: "absolute", top: 10, left: 10, display: "flex", gap: 8 }}>
                  {[["left", C.green, "Left"], ["right", C.accent, "Right"]].map(([side, color, label]) => (
                    <span key={side} style={{
                      fontSize: 11, padding: "3px 9px", borderRadius: 12,
                      background: handsVisible[side] ? color + "22" : "#00000066",
                      color: handsVisible[side] ? color : C.textMuted,
                      border: `1px solid ${handsVisible[side] ? color : C.border}`,
                    }}>
                      {handsVisible[side] ? "●" : "○"} {label}
                      {phase === "recording" && liveFps[side] > 0 && (
                        <span style={{
                          marginLeft: 5,
                          color: liveFps[side] >= MIN_RELIABLE_FPS + 3 ? "#3fb950" : liveFps[side] >= MIN_RELIABLE_FPS ? "#d29922" : "#f85149",
                        }}>
                          {liveFps[side]}fps
                        </span>
                      )}
                    </span>
                  ))}
                </div>
              )}

              {phase === "countdown" && (
                <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "rgba(13,17,23,0.65)" }}>
                  <span style={{ fontSize: 88, fontWeight: 700, color: "#fff", lineHeight: 1 }}>{countdown}</span>
                  <span style={{ fontSize: 14, color: C.textSub, marginTop: 10 }}>Extend arms and hold still…</span>
                </div>
              )}

              {phase === "recording" && (
                <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "rgba(13,17,23,0.75)", padding: "10px 14px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <span style={{ width: 7, height: 7, borderRadius: "50%", background: C.red, animation: "pulse 1s infinite" }} />
                    <span style={{ fontSize: 12, color: "#fff", fontWeight: 500 }}>Hold still — {timeLeft}s remaining</span>
                  </div>
                  <div style={{ background: "#ffffff22", borderRadius: 4, height: 5, overflow: "hidden" }}>
                    <div style={{ height: "100%", borderRadius: 4, background: C.red, width: `${((TEST_DURATION_S - timeLeft) / TEST_DURATION_S) * 100}%`, transition: "width 1s linear" }} />
                  </div>
                  {(liveFps.left > 0 && liveFps.left < MIN_RELIABLE_FPS) || (liveFps.right > 0 && liveFps.right < MIN_RELIABLE_FPS) ? (
                    <p style={{ fontSize: 11, color: "#f85149", margin: "6px 0 0" }}>
                      ⚠️ Low per-hand frame rate — tremor frequency detection needs {MIN_RELIABLE_FPS}+ fps per hand.
                      Move closer to the camera, improve lighting, or shake with slightly less blur.
                    </p>
                  ) : null}
                </div>
              )}

              {phase === "analyzing" && (
                <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(13,17,23,0.7)" }}>
                  <p style={{ fontSize: 14, color: C.textSub }}>Running FFT analysis…</p>
                </div>
              )}
            </div>

            {phase === "ready" && (
              <button onClick={beginTest} disabled={!canStart} style={{
                marginTop: 12, width: "100%", padding: "11px 0", borderRadius: 8, border: "none",
                background: canStart ? C.accent : C.surfaceHigh, color: canStart ? "#fff" : C.textMuted,
                fontSize: 14, fontWeight: 600, cursor: canStart ? "pointer" : "not-allowed", transition: "all 0.2s",
              }}>
                {canStart ? "Begin test" : "Show both hands to camera first"}
              </button>
            )}
          </div>
        )}

        {error && (
          <div style={{ background: C.redDim, border: `1px solid ${C.red}44`, borderRadius: 8, padding: "12px 16px", marginBottom: 16 }}>
            <p style={{ fontSize: 13, color: C.red, margin: "0 0 8px" }}>{error}</p>
            <button onClick={reset} style={{ fontSize: 12, color: C.accent, background: "none", border: "none", cursor: "pointer", padding: 0 }}>Try again →</button>
          </div>
        )}

        {result && phase === "done" && (
          <>
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "18px 20px", marginBottom: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontSize: 17, fontWeight: 600 }}>Overall: {result.overall_score}/100</span>
                <Badge label={result.overall_severity} severity={result.overall_severity} />
              </div>
              <p style={{ fontSize: 13, color: C.textMuted, margin: 0, lineHeight: 1.7 }}>
                Asymmetry: <strong style={{ color: C.text }}>{Math.round(result.asymmetry_index * 100)}%</strong>
                &nbsp;·&nbsp;
                Dominant side: <strong style={{ color: C.text, textTransform: "capitalize" }}>{result.dominant_side}</strong>
                {result.asymmetry_index > 0.3 && <span style={{ color: C.orange }}> — notable asymmetry, often seen in early-stage PD</span>}
              </p>
            </div>

            <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
              <StabilityCard side="Left"  scores={result.left}  />
              <StabilityCard side="Right" scores={result.right} />
            </div>

            <button onClick={reset} style={{ width: "100%", padding: "10px 0", borderRadius: 8, cursor: "pointer", border: `1px solid ${C.border}`, background: "none", color: C.textSub, fontSize: 14, marginBottom: 14 }}>
              Test again
            </button>

            <div style={{ background: "#2d1f07", border: `1px solid ${C.yellow}33`, borderRadius: 8, padding: "10px 14px" }}>
              <p style={{ fontSize: 12, color: C.yellow, margin: 0 }}>⚠️ For research and screening only. Does not replace clinical diagnosis.</p>
            </div>
          </>
        )}
      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
    </div>
  );
}