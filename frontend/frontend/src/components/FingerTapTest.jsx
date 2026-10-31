/**
 * NeuroTrack AI — Finger Tap Test (MDS-UPDRS 3.4)
 *
 * Records the raw thumb-tip <-> index-tip aperture plus hand size (wrist -> middle
 * MCP) on every camera frame, using the shared GPU hand tracker. The backend
 * normalises aperture by hand size, detects taps and computes the biomarkers.
 *
 * Sends: [{v: aperture_px, t: ms, s: hand_size_px}] + measured fps.
 * Receives: tap_events, raw_frames, biomarkers, score, severity, quality, result_id.
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { useHandTracking, px } from "../tracking/useHandTracking";
import { LANDMARK } from "../tracking/handTracker";

// ─── Design tokens ────────────────────────────────────────────────────────────
const C = {
  bg: "#0d1117", surface: "#161b22", surfaceHigh: "#1c2128",
  border: "#30363d", borderSoft: "#21262d",
  text: "#e6edf3", textSub: "#8b949e", textMuted: "#484f58",
  accent: "#2f81f7", accentDim: "#1a4b8c",
  green: "#3fb950", yellow: "#d29922", orange: "#f0883e", red: "#f85149",
  greenDim: "#122c1a", yellowDim: "#2d1f07", orangeDim: "#2d1505", redDim: "#2c0d0d",
};
const SEVERITY = {
  Normal:   { color: C.green,  dim: C.greenDim  },
  Mild:     { color: C.yellow, dim: C.yellowDim },
  Moderate: { color: C.orange, dim: C.orangeDim },
  Severe:   { color: C.red,    dim: C.redDim    },
};
const TEST_DURATION_S  = 10;
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
];
const UI_UPDATE_MS = 100; // live readouts refresh at 10 Hz instead of every frame

// ─── Helpers ──────────────────────────────────────────────────────────────────
const fmt = (n, d = 1) => typeof n === "number" ? n.toFixed(d) : "—";
const pct = n => typeof n === "number" ? Math.round(n * 100) + "%" : "—";

function Badge({ label, severity }) {
  const cfg = SEVERITY[severity] || { color: C.textSub, dim: C.surface };
  return (
    <span style={{
      fontSize: 12, fontWeight: 500, padding: "3px 10px", borderRadius: 6, display: "inline-block",
      background: cfg.dim, color: cfg.color, border: `1px solid ${cfg.color}33`,
    }}>{label}</span>
  );
}

function MetricCard({ label, value, sub, color }) {
  return (
    <div style={{ background: C.surfaceHigh, border: `1px solid ${C.borderSoft}`, borderRadius: 8, padding: "12px 14px", textAlign: "center" }}>
      <p style={{ fontSize: 10, color: C.textMuted, margin: "0 0 5px", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</p>
      <p style={{ fontSize: 20, fontWeight: 600, color: color || C.text, margin: 0, fontVariantNumeric: "tabular-nums" }}>{value}</p>
      {sub && <p style={{ fontSize: 10, color: C.textMuted, margin: "3px 0 0" }}>{sub}</p>}
    </div>
  );
}

function BioBar({ label, value, max = 100, unit = "%", color, invert = false }) {
  const fill = Math.min(100, Math.max(0, Math.round((value / max) * 100)));
  const displayFill = invert ? 100 - fill : fill;
  return (
    <div style={{ marginBottom: 9 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 12, color: C.textSub }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 500, color: C.text }}>{fmt(value)}{unit ? " " + unit : ""}</span>
      </div>
      <div style={{ background: C.bg, borderRadius: 4, height: 6, overflow: "hidden" }}>
        <div style={{ width: displayFill + "%", height: "100%", background: color, borderRadius: 4, transition: "width 0.7s ease" }} />
      </div>
    </div>
  );
}

// ─── Aperture signal chart ────────────────────────────────────────────────────
function ApertureChart({ frames, tapEvents, width = 620, height = 140 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!frames?.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const padL = 36, padR = 12, padT = 16, padB = 28;
    const w = canvas.width  - padL - padR;
    const h = canvas.height - padT - padB;

    const ts   = frames.map(f => f.t);
    const vs   = frames.map(f => f.v);
    const tMin = ts[0], tMax = ts[ts.length - 1];
    const vMax = Math.max(...vs) * 1.05;

    const tx = t => padL + ((t - tMin) / (tMax - tMin || 1)) * w;
    const ty = v => padT + (1 - v / vMax) * h;

    // Grid lines
    ctx.strokeStyle = C.border;
    ctx.lineWidth   = 0.5;
    [0, 25, 50, 75, 100].forEach(pctV => {
      const v = vMax * pctV / 100;
      const y = ty(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + w, y); ctx.stroke();
      ctx.fillStyle = C.textMuted; ctx.font = "9px system-ui"; ctx.textAlign = "right";
      ctx.fillText(vMax < 10 ? v.toFixed(1) : Math.round(v), padL - 4, y + 3);   // v2 sends hand-size units
    });

    // Axes
    ctx.strokeStyle = C.border; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + h); ctx.lineTo(padL + w, padT + h); ctx.stroke();

    // Aperture signal — filled area
    ctx.beginPath();
    ctx.moveTo(tx(ts[0]), padT + h);
    frames.forEach(f => ctx.lineTo(tx(f.t), ty(f.v)));
    ctx.lineTo(tx(ts[ts.length - 1]), padT + h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, padT, 0, padT + h);
    grad.addColorStop(0,   C.accent + "66");
    grad.addColorStop(1,   C.accent + "11");
    ctx.fillStyle = grad;
    ctx.fill();

    // Signal line
    ctx.beginPath();
    frames.forEach((f, i) => i === 0 ? ctx.moveTo(tx(f.t), ty(f.v)) : ctx.lineTo(tx(f.t), ty(f.v)));
    ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5; ctx.stroke();

    // Tap peak markers
    tapEvents?.forEach(tap => {
      const x = tx(tap.t), y = ty(tap.amp);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = C.green; ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
    });

    // Amplitude trend line (from tap events)
    if (tapEvents?.length >= 2) {
      const txs = tapEvents.map(e => tx(e.t));
      const tys = tapEvents.map(e => ty(e.amp));
      const n   = txs.length;
      const mx  = txs.reduce((a,b)=>a+b)/n, my = tys.reduce((a,b)=>a+b)/n;
      const sl  = txs.reduce((s,x,i)=>s+(x-mx)*(tys[i]-my),0) / txs.reduce((s,x)=>s+(x-mx)**2,0);
      const ic  = my - sl * mx;
      ctx.beginPath();
      ctx.moveTo(txs[0],   sl * txs[0]   + ic);
      ctx.lineTo(txs[n-1], sl * txs[n-1] + ic);
      ctx.strokeStyle = sl > 0 ? C.red + "cc" : C.green + "cc";
      ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
    }

    // X-axis labels
    ctx.fillStyle = C.textMuted; ctx.font = "9px system-ui"; ctx.textAlign = "center";
    [0, 2.5, 5, 7.5, 10].forEach(s => {
      const t = tMin + s * 1000;
      ctx.fillText(s + "s", tx(t), padT + h + 16);
    });

    // Legend
    ctx.fillStyle = C.green; ctx.font = "9px system-ui"; ctx.textAlign = "left";
    ctx.fillText("● tap peak", padL + 4, padT + 12);
    ctx.fillStyle = C.textMuted;
    ctx.fillText("- - amplitude trend", padL + 70, padT + 12);

  }, [frames, tapEvents]);

  return (
    <canvas
      ref={canvasRef}
      width={width} height={height}
      style={{ width: "100%", display: "block", borderRadius: 8, background: C.surface, border: `1px solid ${C.border}` }}
      aria-label="Aperture signal: distance between thumb and index finger over time"
    />
  );
}

// ─── Speed chart ──────────────────────────────────────────────────────────────
function SpeedChart({ tapEvents, width = 620, height = 100 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!tapEvents?.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const padL = 36, padR = 12, padT = 12, padB = 24;
    const w = canvas.width  - padL - padR;
    const h = canvas.height - padT - padB;

    const speeds = tapEvents.map(e => e.speed).filter(s => s > 0);
    if (speeds.length < 2) return;

    const vMax = Math.max(...speeds) * 1.1;
    const n    = speeds.length;

    const tx = i => padL + (i / (n - 1)) * w;
    const ty = v => padT + (1 - v / vMax) * h;

    // Bars
    speeds.forEach((s, i) => {
      ctx.fillStyle = C.yellow + "88";
      ctx.fillRect(tx(i) - (w / n / 2) * 0.6, ty(s), (w / n) * 0.6, padT + h - ty(s));
    });

    // Dots
    speeds.forEach((s, i) => {
      ctx.beginPath(); ctx.arc(tx(i), ty(s), 3, 0, Math.PI * 2);
      ctx.fillStyle = C.yellow; ctx.fill();
    });

    // Trend
    const txs = speeds.map((_, i) => tx(i));
    const tys = speeds.map(s => ty(s));
    const mx  = txs.reduce((a,b)=>a+b)/n, my = tys.reduce((a,b)=>a+b)/n;
    const denom = txs.reduce((s,x)=>s+(x-mx)**2,0);
    if (denom > 0) {
      const sl = txs.reduce((s,x,i)=>s+(x-mx)*(tys[i]-my),0) / denom;
      const ic = my - sl * mx;
      ctx.beginPath();
      ctx.moveTo(txs[0], sl*txs[0]+ic); ctx.lineTo(txs[n-1], sl*txs[n-1]+ic);
      ctx.strokeStyle = sl > 0 ? C.red + "cc" : C.green + "cc";
      ctx.lineWidth = 1.5; ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
    }

    // Y labels
    ctx.fillStyle = C.textMuted; ctx.font = "9px system-ui"; ctx.textAlign = "right";
    [0, 50, 100].forEach(p => {
      const v = vMax * p / 100;
      ctx.fillText(Math.round(v), padL - 4, ty(v) + 3);
    });
    ctx.textAlign = "center";
    ctx.fillText("Tap #", padL + w / 2, padT + h + 18);

  }, [tapEvents]);

  return (
    <canvas
      ref={canvasRef}
      width={width} height={height}
      style={{ width: "100%", display: "block", borderRadius: 8, background: C.surface, border: `1px solid ${C.border}` }}
      aria-label="Closing speed per tap"
    />
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function FingerTapTest({ patientId = "anonymous", apiBase = "", onResult = null, active = true }) {
  const canvasRef     = useRef(null);
  const timerRef      = useRef(null);
  const countdownRef  = useRef(null);
  const recordingRef  = useRef(false);
  const startTsRef    = useRef(null);    // timestamp of the first recorded frame
  const framesRef     = useRef([]);      // [{v, t, s}] raw aperture stream
  const uiTickRef     = useRef(0);

  const [phase, setPhase]             = useState("idle");
  const [handVisible, setHandVisible] = useState(false);
  const [countdown, setCountdown]     = useState(3);
  const [timeLeft, setTimeLeft]       = useState(TEST_DURATION_S);
  const [liveAperture, setLiveAperture] = useState(0);
  const [liveFrameCount, setLiveFrameCount] = useState(0);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);

  // ── Per-frame handler (runs inside the tracker loop) ────────────────────
  const onFrame = useCallback((res, ts, video) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;
    const ctx = canvas.getContext("2d");
    // Mirror only the video, not the canvas, so overlay text reads correctly.
    ctx.save(); ctx.scale(-1, 1); ctx.drawImage(video, -W, 0, W, H); ctx.restore();

    const lm = res.landmarks?.[0];
    let aperture = null;
    if (lm) {
      const mx = (i) => (1 - lm[i].x) * W, my = (i) => lm[i].y * H;   // mirrored, for drawing
      ctx.strokeStyle = C.accent + "77"; ctx.lineWidth = 1.5;
      HAND_CONNECTIONS.forEach(([a, b]) => {
        ctx.beginPath(); ctx.moveTo(mx(a), my(a)); ctx.lineTo(mx(b), my(b)); ctx.stroke();
      });
      const th = px(lm, LANDMARK.THUMB_TIP, video), ix = px(lm, LANDMARK.INDEX_TIP, video);
      const wr = px(lm, LANDMARK.WRIST, video), mc = px(lm, LANDMARK.MIDDLE_MCP, video);
      aperture = Math.hypot(th.x - ix.x, th.y - ix.y);
      const handSize = Math.hypot(wr.x - mc.x, wr.y - mc.y);
      const open = handSize > 0 && aperture / handSize > 0.5;

      ctx.beginPath(); ctx.moveTo(mx(LANDMARK.THUMB_TIP), my(LANDMARK.THUMB_TIP));
      ctx.lineTo(mx(LANDMARK.INDEX_TIP), my(LANDMARK.INDEX_TIP));
      ctx.strokeStyle = open ? C.green : C.orange; ctx.lineWidth = 2.5; ctx.stroke();
      [[LANDMARK.THUMB_TIP, C.yellow], [LANDMARK.INDEX_TIP, C.orange]].forEach(([i, c]) => {
        ctx.beginPath(); ctx.arc(mx(i), my(i), 7, 0, Math.PI * 2);
        ctx.fillStyle = c; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
      });
      ctx.fillStyle = "#fff"; ctx.font = "bold 13px system-ui";
      ctx.fillText(`${Math.round(aperture)}px`,
        (mx(LANDMARK.THUMB_TIP) + mx(LANDMARK.INDEX_TIP)) / 2 + 8, (my(LANDMARK.THUMB_TIP) + my(LANDMARK.INDEX_TIP)) / 2 - 8);

      if (recordingRef.current) {
        if (startTsRef.current === null) startTsRef.current = ts;
        framesRef.current.push({ v: aperture, t: ts - startTsRef.current, s: handSize });
      }
    }

    const now = performance.now();
    if (now - uiTickRef.current >= UI_UPDATE_MS) {
      uiTickRef.current = now;
      setHandVisible(Boolean(lm));
      if (aperture !== null) setLiveAperture(Math.round(aperture));
      if (recordingRef.current) setLiveFrameCount(framesRef.current.length);
    }
  }, []);

  const tracking = useHandTracking({ numHands: 1, onFrame });

  const clearTimers = () => {
    clearInterval(timerRef.current); timerRef.current = null;
    clearInterval(countdownRef.current); countdownRef.current = null;
  };
  useEffect(() => clearTimers, []);

  // Leaving the tab stops the camera and abandons an in-progress recording.
  useEffect(() => {
    if (active) return;
    clearTimers();
    recordingRef.current = false;
    tracking.stop();
    setPhase((p) => (p === "done" || p === "error" ? p : "idle"));
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  const startCamera = useCallback(async () => {
    setPhase("camera_loading"); setError(null);
    const ok = await tracking.start();
    setPhase(ok ? "ready" : "error");
  }, [tracking]);

  useEffect(() => {
    if (tracking.status === "error" && tracking.error) setError(tracking.error);
  }, [tracking.status, tracking.error]);

  function beginTest() {
    framesRef.current = [];
    setLiveFrameCount(0); setLiveAperture(0);
    setResult(null); setError(null);
    setCountdown(3); setPhase("countdown");
    tracking.lock(true);   // fixed resolution from now until the result arrives

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
      if (framesRef.current.length < 20) throw new Error("Too few frames captured. Make sure your hand is visible throughout the test.");
      const res = await fetch(`${apiBase}/api/finger-tap`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId, hand: "dominant", duration_s: TEST_DURATION_S,
          frames: framesRef.current, reported_fps: tracking.statsRef.current?.fps || undefined,
        }),
      });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `Server error ${res.status}`); }
      const data = await res.json();
      setResult(data); if (onResult) onResult(data); setPhase("done");
    } catch (err) { setError(err.message || "Analysis failed."); setPhase("error"); }
    finally { tracking.lock(false); }
  }

  async function reset() {
    setResult(null); setError(null); setLiveFrameCount(0);
    if (tracking.status === "ready") { setPhase("ready"); return; }
    await startCamera();
  }

  const bio = result?.biomarkers;
  const sev = result?.severity || "Normal";
  const sevCfg = SEVERITY[sev] || SEVERITY.Normal;

  const statusColor = { idle: C.green, camera_loading: C.yellow, ready: C.green, countdown: C.yellow, recording: C.red, analyzing: C.accent, done: C.green, error: C.red }[phase] || C.textSub;
  const statusLabel = { idle: "Ready", camera_loading: "Starting…", ready: "Camera ready", countdown: "Get ready…", recording: `Recording · ${timeLeft}s`, analyzing: "Analysing…", done: "Complete", error: "Error" }[phase] || "—";

  return (
    <div style={{ fontFamily: "system-ui,-apple-system,sans-serif", background: C.bg, minHeight: "100vh", padding: "24px 20px", color: C.text }}>
      <div style={{ maxWidth: 700, margin: "0 auto" }}>
        {/* Camera feed for the tracker. Always mounted (so it exists before start() runs) and
            kept rendered rather than display:none, so requestVideoFrameCallback keeps firing. */}
        <video ref={tracking.videoRef} style={{ position: "fixed", left: 0, top: 0, width: 1, height: 1, opacity: 0, pointerEvents: "none" }} playsInline muted />

        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <div>
            <p style={{ fontSize: 11, color: C.textMuted, margin: "0 0 4px", textTransform: "uppercase", letterSpacing: "0.08em" }}>NeuroTrack AI · Motor assessment</p>
            <h1 style={{ fontSize: 22, fontWeight: 600, margin: 0 }}>Finger tap test</h1>
            <p style={{ fontSize: 12, color: C.textMuted, margin: "4px 0 0" }}>UPDRS 3.4 — Bradykinesia & amplitude decay</p>
          </div>
          <span style={{ fontSize: 11, fontWeight: 500, padding: "4px 12px", borderRadius: 20, background: C.surfaceHigh, color: statusColor, border: `1px solid ${statusColor}44` }}>{statusLabel}</span>
        </div>

        {/* Instructions */}
        {phase === "idle" && (
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "20px 22px", marginBottom: 16 }}>
            <p style={{ fontSize: 13, color: C.textSub, margin: "0 0 14px", lineHeight: 1.7 }}>
              Tap your <strong style={{ color: C.text }}>index finger to thumb</strong> as fast and as wide as possible for 10 seconds.
              The <span style={{ color: C.accent }}>blue line</span> between your fingertips shows the aperture being tracked.
            </p>
            <ol style={{ fontSize: 13, color: C.textSub, margin: "0 0 20px", paddingLeft: 20, lineHeight: 1.9 }}>
              <li>Hold your dominant hand toward the camera, palm facing you.</li>
              <li>Keep other fingers extended and out of the way.</li>
              <li>Click "Enable camera" — point hand at camera until detected.</li>
              <li>Tap as fast and as wide open as possible for the full 10 seconds.</li>
            </ol>
            <button onClick={startCamera} style={{ width: "100%", padding: "11px 0", borderRadius: 8, border: "none", background: C.accent, color: "#fff", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
              Enable camera
            </button>
          </div>
        )}

        {/* Webcam */}
        {phase !== "idle" && (
          <div style={{ marginBottom: 14 }}>
            <div style={{ position: "relative", borderRadius: 10, overflow: "hidden", border: `1px solid ${phase === "recording" ? C.red + "66" : C.border}` }}>
              <canvas ref={canvasRef} width={640} height={480} style={{ width: "100%", display: "block", background: C.surfaceHigh }} />

              {/* Badges */}
              {["ready","countdown","recording"].includes(phase) && (
                <div style={{ position: "absolute", top: 10, left: 10, display: "flex", gap: 8 }}>
                  <span style={{ fontSize: 11, padding: "3px 9px", borderRadius: 12, background: handVisible ? C.accent+"22":"#00000066", color: handVisible ? C.accent : C.textMuted, border: `1px solid ${handVisible ? C.accent : C.border}` }}>
                    {handVisible ? "● Hand detected" : "○ No hand"}
                  </span>
                  {phase === "recording" && (
                    <span style={{ fontSize: 11, padding: "3px 9px", borderRadius: 12, background: "#00000066", color: C.text, border: `1px solid ${C.border}`, fontVariantNumeric: "tabular-nums" }}>
                      {liveAperture}px aperture
                    </span>
                  )}
                </div>
              )}
              {phase === "recording" && (
                <div style={{ position: "absolute", top: 10, right: 10 }}>
                  <span style={{ fontSize: 11, padding: "3px 9px", borderRadius: 12, background: "#00000066", color: C.green, border: `1px solid ${C.green}44`, fontVariantNumeric: "tabular-nums" }}>
                    {liveFrameCount} frames · {Math.round(tracking.stats?.fps || 0)} fps · {tracking.stats?.delegate || "—"}
                  </span>
                </div>
              )}

              {/* Countdown */}
              {phase === "countdown" && (
                <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "rgba(13,17,23,0.65)" }}>
                  <span style={{ fontSize: 88, fontWeight: 700, color: "#fff", lineHeight: 1 }}>{countdown}</span>
                  <span style={{ fontSize: 14, color: C.textSub, marginTop: 10 }}>Get ready — start tapping!</span>
                </div>
              )}

              {/* Recording progress */}
              {phase === "recording" && (
                <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "rgba(13,17,23,0.75)", padding: "10px 14px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <span style={{ width: 7, height: 7, borderRadius: "50%", background: C.red, animation: "pulse 1s infinite" }} />
                    <span style={{ fontSize: 12, color: "#fff", fontWeight: 500 }}>Tap as fast and wide as you can — {timeLeft}s left</span>
                  </div>
                  <div style={{ background: "#ffffff22", borderRadius: 4, height: 5, overflow: "hidden" }}>
                    <div style={{ height: "100%", borderRadius: 4, background: C.red, width: `${((TEST_DURATION_S - timeLeft) / TEST_DURATION_S) * 100}%`, transition: "width 1s linear" }} />
                  </div>
                </div>
              )}

              {phase === "analyzing" && (
                <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(13,17,23,0.7)" }}>
                  <p style={{ fontSize: 14, color: C.textSub }}>Detecting taps & analysing signal…</p>
                </div>
              )}
            </div>

            {phase === "ready" && (
              <button onClick={beginTest} disabled={!handVisible} style={{ marginTop: 12, width: "100%", padding: "11px 0", borderRadius: 8, border: "none", background: handVisible ? C.accent : C.surfaceHigh, color: handVisible ? "#fff" : C.textMuted, fontSize: 14, fontWeight: 600, cursor: handVisible ? "pointer" : "not-allowed", transition: "all 0.2s" }}>
                {handVisible ? "Begin test" : "Show your hand to the camera first"}
              </button>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{ background: C.redDim, border: `1px solid ${C.red}44`, borderRadius: 8, padding: "12px 16px", marginBottom: 16 }}>
            <p style={{ fontSize: 13, color: C.red, margin: "0 0 8px" }}>{error}</p>
            <button onClick={reset} style={{ fontSize: 12, color: C.accent, background: "none", border: "none", cursor: "pointer", padding: 0 }}>Try again →</button>
          </div>
        )}

        {/* Results */}
        {result && phase === "done" && (
          <>
            {/* Score card */}
            <div style={{ background: C.surface, border: `1px solid ${sevCfg.color}44`, borderRadius: 10, padding: "20px 24px", marginBottom: 14, display: "flex", alignItems: "center", gap: 24 }}>
              <div style={{ textAlign: "center", minWidth: 80 }}>
                <p style={{ fontSize: 44, fontWeight: 700, color: sevCfg.color, margin: 0, fontVariantNumeric: "tabular-nums" }}>{result.overall_score}</p>
                <p style={{ fontSize: 11, color: C.textMuted, margin: 0 }}>/100</p>
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <span style={{ fontSize: 17, fontWeight: 600 }}>{sev} impairment</span>
                  <Badge label={sev} severity={sev} />
                </div>
                <p style={{ fontSize: 13, color: C.textSub, margin: "0 0 10px", lineHeight: 1.6 }}>
                  {{ Normal: "Tapping speed, amplitude, and rhythm are within normal range.", Mild: "Slight reduction in tap speed or amplitude consistency.", Moderate: "Notable bradykinesia or amplitude decay detected.", Severe: "Significant motor impairment — slow taps, major decay, or freezing." }[sev]}
                </p>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {[
                    [`${result.tap_count} taps detected`, C.accent],
                    [`${fmt(bio.tap_rate)} taps/s`, C.text],
                    [`Dominant: ${result.dominant_feature}`, sevCfg.color],
                  ].map(([label, color]) => (
                    <span key={label} style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: C.surfaceHigh, color, border: `1px solid ${C.border}` }}>{label}</span>
                  ))}
                </div>
              </div>
            </div>

            {/* Metric grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
              <MetricCard label="Tap rate" value={fmt(bio.tap_rate)} sub="taps/sec" color={bio.tap_rate >= 4 ? C.green : bio.tap_rate >= 2.5 ? C.yellow : C.red} />
              <MetricCard label="Mean amplitude" value={result.amplitude_unit === "hand" ? pct(bio.mean_amplitude) : fmt(bio.mean_amplitude, 0)} sub={result.amplitude_unit === "hand" ? "of hand size" : "px aperture"} />
              <MetricCard label="Mean speed" value={fmt(bio.mean_speed, 1)} sub={result.amplitude_unit === "hand" ? "hand-lengths/s" : "px/s"} color={result.amplitude_unit !== "hand" ? C.text : bio.mean_speed > 8 ? C.green : bio.mean_speed > 4 ? C.yellow : C.red} />
              <MetricCard label="Hesitations" value={bio.hesitation_count} sub={bio.hesitation_count === 0 ? "None" : "Freezing events"} color={bio.hesitation_count === 0 ? C.green : bio.hesitation_count <= 2 ? C.yellow : C.red} />
              <MetricCard label="Rhythm CV" value={pct(bio.rhythm_cv)} sub={bio.rhythm_cv < 0.15 ? "Regular" : bio.rhythm_cv < 0.3 ? "Some irregularity" : "Arrhythmic"} color={bio.rhythm_cv < 0.15 ? C.green : bio.rhythm_cv < 0.3 ? C.yellow : C.red} />
              <MetricCard label="Decrement" value={pct(bio.decrement_index)} sub="0% = no decay" color={bio.decrement_index < 0.2 ? C.green : bio.decrement_index < 0.5 ? C.yellow : C.red} />
            </div>

            {/* Biomarker bars */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "18px 20px", marginBottom: 14 }}>
              <p style={{ fontSize: 12, color: C.textSub, margin: "0 0 14px", textTransform: "uppercase", letterSpacing: "0.04em" }}>Biomarker breakdown</p>
              <BioBar label="Tap rate (vs 4.5/s norm)"   value={Math.min((bio.tap_rate / 4.5) * 100, 100)} unit="%" color={C.accent} />
              <BioBar label="Amplitude stability"         value={Math.max(0, (1 + bio.amplitude_decay) * 100)} unit="%" color={C.green} />
              <BioBar label="Speed stability"             value={Math.max(0, (1 + bio.speed_decay) * 100)} unit="%" color={C.yellow} />
              <BioBar label="Rhythm regularity (1 − CV)" value={Math.max(0, (1 - bio.rhythm_cv) * 100)} unit="%" color={C.orange} />
              <BioBar label="Hesitation-free time"       value={Math.max(0, (1 - bio.arrest_ratio) * 100)} unit="%" color={C.red} />
            </div>

            {/* Aperture signal chart */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "18px 20px", marginBottom: 14 }}>
              <p style={{ fontSize: 12, color: C.textSub, margin: "0 0 4px", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                Thumb ↔ Index distance over time
              </p>
              <p style={{ fontSize: 11, color: C.textMuted, margin: "0 0 10px" }}>
                Green dots = detected taps &nbsp;·&nbsp; Dashed line = amplitude trend (green rising / red falling)
              </p>
              <ApertureChart frames={result.raw_frames} tapEvents={result.tap_events} />
            </div>

            {/* Closing speed chart */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "18px 20px", marginBottom: 14 }}>
              <p style={{ fontSize: 12, color: C.textSub, margin: "0 0 4px", textTransform: "uppercase", letterSpacing: "0.04em" }}>Closing speed per tap</p>
              <p style={{ fontSize: 11, color: C.textMuted, margin: "0 0 10px" }}>Speed should stay high throughout — a falling trend indicates fatigue or bradykinesia.</p>
              <SpeedChart tapEvents={result.tap_events} />
            </div>

            <div style={{ display: "flex", gap: 10, marginBottom: 14 }}>
              <button onClick={reset} style={{ flex: 1, padding: "10px 0", borderRadius: 8, cursor: "pointer", border: `1px solid ${C.border}`, background: "none", color: C.textSub, fontSize: 14 }}>Test again</button>
            </div>

            <div style={{ background: C.yellowDim, border: `1px solid ${C.yellow}33`, borderRadius: 8, padding: "10px 14px" }}>
              <p style={{ fontSize: 12, color: C.yellow, margin: 0, lineHeight: 1.6 }}>⚠️ For research and screening only. Does not replace clinical diagnosis by a neurologist.</p>
            </div>
          </>
        )}
      </div>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
    </div>
  );
}