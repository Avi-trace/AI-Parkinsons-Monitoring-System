/**
 * NeuroTrack AI — Postural Tremor Test
 * Webcam + shared GPU hand tracker (MediaPipe Tasks), both hands, 15 s hold.
 *
 *  - Handedness: the tracker sees the un-mirrored camera image, and MediaPipe's
 *    labels assume a mirrored (selfie) image, so its "Left" is the patient's
 *    physical right hand. We swap accordingly.
 *  - Tracked point per hand = mean of wrist(0), middle MCP(9) and middle tip(12);
 *    hand size = wrist -> middle MCP (backend normalises amplitude by it).
 *  - Timestamps are the camera frame capture times from the tracker.
 *  - The measured fps is sent; the backend rejects < 15 fps and only searches the
 *    full 3-12 Hz band at 30+ fps.
 *  - The last result persists across refresh (localStorage, keyed per patient)
 *    and is re-published to the dashboard on load.
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { useHandTracking, px } from "../tracking/useHandTracking";

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
const MIN_RELIABLE_FPS = 15; // must match backend MIN_TREMOR_FPS (constants.py)

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

export default function TremorTest({ patientId = "anonymous", apiBase = "", onResult = null, active = true }) {
  const canvasRef     = useRef(null);
  const timerRef      = useRef(null);
  const countdownRef  = useRef(null);
  const leftFrames    = useRef([]);   // patient's physical left hand
  const rightFrames   = useRef([]);   // patient's physical right hand
  const recordingRef  = useRef(false);
  const startTsRef    = useRef(null);
  const uiTickRef     = useRef(0);

  const [phase, setPhase]               = useState("idle");
  const [handsVisible, setHandsVisible] = useState({ left: false, right: false });
  const [countdown, setCountdown]       = useState(3);
  const [timeLeft, setTimeLeft]         = useState(TEST_DURATION_S);
  const [liveFps, setLiveFps]           = useState({ left: 0, right: 0 });   // per hand
  const [result, setResult]             = useState(() => loadStoredResult(patientId));
  const [error, setError]               = useState(null);
  const fpsCounterRef = useRef({ left: { count: 0, last: 0 }, right: { count: 0, last: 0 } });

  // A result restored from storage goes straight to the results view and is
  // re-published so the dashboard sees it too.
  useEffect(() => {
    if (result) { setPhase("done"); onResult?.(result); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onFrame = useCallback((res, ts, video) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.save(); ctx.scale(-1, 1); ctx.drawImage(video, -W, 0, W, H); ctx.restore();   // mirror video only

    let sawLeft = false, sawRight = false;
    const recording = recordingRef.current;
    if (recording && startTsRef.current === null) startTsRef.current = ts;

    (res.landmarks || []).forEach((lm, i) => {
      const mpLabel = res.handedness?.[i]?.[0]?.categoryName;   // "Left" | "Right"
      if (!mpLabel) return;
      const physicalSide = mpLabel === "Left" ? "right" : "left";
      // Two detections with the same label in one frame: keep only the first.
      if ((physicalSide === "left" && sawLeft) || (physicalSide === "right" && sawRight)) return;
      if (physicalSide === "left") sawLeft = true; else sawRight = true;
      const color = physicalSide === "left" ? C.green : C.accent;

      const trackPoints = [WRIST, MIDDLE_MCP, MIDDLE_TIP];
      const cx = trackPoints.reduce((acc, idx) => acc + lm[idx].x, 0) / trackPoints.length;
      const cy = trackPoints.reduce((acc, idx) => acc + lm[idx].y, 0) / trackPoints.length;
      const dx = (1 - cx) * W, dy = cy * H;   // mirrored, for drawing

      ctx.strokeStyle = color + "88"; ctx.lineWidth = 1.5;
      HAND_CONNECTIONS.forEach(([a, b]) => {
        ctx.beginPath();
        ctx.moveTo((1 - lm[a].x) * W, lm[a].y * H);
        ctx.lineTo((1 - lm[b].x) * W, lm[b].y * H);
        ctx.stroke();
      });
      ctx.beginPath(); ctx.arc(dx, dy, 8, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = "bold 12px system-ui";
      ctx.fillText(physicalSide.charAt(0).toUpperCase() + physicalSide.slice(1), dx + 10, dy - 8);

      if (recording) {
        const wr = px(lm, WRIST, video), mc = px(lm, MIDDLE_MCP, video);
        const frame = { x: cx * W, y: cy * H, s: Math.hypot(wr.x - mc.x, wr.y - mc.y), t: ts - startTsRef.current };
        (physicalSide === "left" ? leftFrames : rightFrames).current.push(frame);

        const now = performance.now();
        const counter = fpsCounterRef.current[physicalSide];
        counter.count++;
        if (now - counter.last >= 1000) {
          setLiveFps(prev => ({ ...prev, [physicalSide]: counter.count }));
          fpsCounterRef.current[physicalSide] = { count: 0, last: now };
        }
      }
    });

    const now = performance.now();
    if (now - uiTickRef.current >= 100) {   // live UI at 10 Hz, not every frame
      uiTickRef.current = now;
      setHandsVisible(prev => (prev.left === sawLeft && prev.right === sawRight) ? prev : { left: sawLeft, right: sawRight });
    }
  }, []);

  const tracking = useHandTracking({ numHands: 2, onFrame });

  const clearTimers = () => {
    clearInterval(timerRef.current); timerRef.current = null;
    clearInterval(countdownRef.current); countdownRef.current = null;
  };
  useEffect(() => clearTimers, []);

  useEffect(() => {
    if (active) return;
    clearTimers();
    recordingRef.current = false;
    tracking.stop();
    setPhase((p) => (p === "done" || p === "error" ? p : "idle"));
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (tracking.status === "error" && tracking.error) setError(tracking.error);
  }, [tracking.status, tracking.error]);

  const startCamera = useCallback(async () => {
    setPhase("camera_loading"); setError(null);
    const ok = await tracking.start();
    setPhase(ok ? "ready" : "error");
  }, [tracking]);

  function beginTest() {
    leftFrames.current = []; rightFrames.current = [];
    fpsCounterRef.current = { left: { count: 0, last: 0 }, right: { count: 0, last: 0 } };
    setLiveFps({ left: 0, right: 0 });
    setResult(null); setError(null);
    setCountdown(3); setPhase("countdown");
    tracking.lock(true);
    let c = 3;
    countdownRef.current = setInterval(() => {
      c--; setCountdown(c);
      if (c <= 0) { clearInterval(countdownRef.current); countdownRef.current = null; startRecording(); }
    }, 1000);
  }

  function startRecording() {
    setPhase("recording"); setTimeLeft(TEST_DURATION_S);
    startTsRef.current = null;
    recordingRef.current = true;
    let t = TEST_DURATION_S;
    timerRef.current = setInterval(() => {
      t--; setTimeLeft(t);
      if (t <= 0) { clearInterval(timerRef.current); timerRef.current = null; recordingRef.current = false; submit(); }
    }, 1000);
  }

  async function submit() {
    setPhase("analyzing");
    try {
      const lf = leftFrames.current, rf = rightFrames.current;
      const payload = {
        patient_id: patientId, mode: "postural", duration_s: TEST_DURATION_S,
        left_hand:  lf.length >= 30 ? lf : null,
        right_hand: rf.length >= 30 ? rf : null,
        reported_fps: tracking.statsRef.current?.fps || undefined,
      };
      if (!payload.left_hand && !payload.right_hand)
        throw new Error("No hands detected. Make sure both arms are fully visible and try again.");

      const res = await fetch(`${apiBase}/api/tremor-test`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `Server error ${res.status}`); }
      const data = await res.json();
      setResult(data);
      saveStoredResult(patientId, data);
      if (onResult) onResult(data);
      setPhase("done");
    } catch (err) {
      setError(err.message || "Analysis failed."); setPhase("error");
    } finally {
      tracking.lock(false);
    }
  }

  async function reset() {
    setResult(null); setError(null);
    clearStoredResult(patientId);
    leftFrames.current = []; rightFrames.current = [];
    if (tracking.status === "ready") { setPhase("ready"); return; }
    setPhase("idle");   // camera not running (e.g. result restored after a refresh)
  }

  const canStart = handsVisible.left || handsVisible.right;
  const statusColor = { idle: C.green, camera_loading: C.yellow, ready: C.green, countdown: C.yellow, recording: C.red, analyzing: C.accent, done: C.green, error: C.red }[phase] || C.textSub;
  const statusLabel = { idle: "Ready", camera_loading: "Starting…", ready: "Camera ready", countdown: "Get ready…", recording: `Recording · ${timeLeft}s`, analyzing: "Analysing…", done: "Complete", error: "Error" }[phase] || "—";

  return (
    <div style={{ fontFamily: "system-ui,-apple-system,sans-serif", background: C.bg, minHeight: "100vh", padding: "24px 20px", color: C.text }}>
      <div style={{ maxWidth: 700, margin: "0 auto" }}>
        {/* Camera feed for the tracker: always mounted, rendered (not display:none) so
            requestVideoFrameCallback keeps firing. */}
        <video ref={tracking.videoRef} style={{ position: "fixed", left: 0, top: 0, width: 1, height: 1, opacity: 0, pointerEvents: "none" }} playsInline muted />

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
            <button onClick={startCamera} style={{ width: "100%", padding: "11px 0", borderRadius: 8, border: "none", background: C.accent, color: "#fff", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
              Enable camera
            </button>
          </div>
        )}

        {phase !== "idle" && (
          <div style={{ marginBottom: 14 }}>
            <div style={{ position: "relative", borderRadius: 10, overflow: "hidden", border: `1px solid ${phase === "recording" ? C.red + "66" : C.border}` }}>
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