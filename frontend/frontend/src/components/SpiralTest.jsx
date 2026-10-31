/**
 * NeuroTrack AI — Spiral Drawing Test
 * React frontend for Parkinson's Disease motor assessment
 *
 * Connects to: POST http://localhost:5000/api/spiral-test
 *
 * Drop this into your existing NeuroTrack AI React/Vite project as a tab component.
 * Usage: <SpiralTest patientId="PT001" apiBase="http://localhost:5000" />
 */

import { useState, useRef, useEffect, useCallback } from "react";


// ─── Design tokens (matches NeuroTrack AI medical palette) ──────────────────
const C = {
  bg:          "#0d1117",
  surface:     "#161b22",
  surfaceHigh: "#1c2128",
  border:      "#30363d",
  borderSoft:  "#21262d",
  text:        "#e6edf3",
  textSub:     "#8b949e",
  textMuted:   "#484f58",
  accent:      "#2f81f7",
  accentDim:   "#1a4b8c",
  green:       "#3fb950",
  yellow:      "#d29922",
  orange:      "#f0883e",
  red:         "#f85149",
  greenDim:    "#122c1a",
  yellowDim:   "#2d1f07",
  orangeDim:   "#2d1505",
  redDim:      "#2c0d0d",
};

// ─── Canvas constants (must match Flask backend) ─────────────────────────────
const CANVAS_W   = 640;
const CANVAS_H   = 320;
const CX         = CANVAS_W / 2;
const CY         = CANVAS_H / 2;
const MAX_RADIUS = Math.min(CX, CY) - 20;
const TURNS      = 3.5;

// ─── Severity config ─────────────────────────────────────────────────────────
const SEVERITY = {
  Normal:   { color: C.green,  dim: C.greenDim,  ring: "#3fb950" },
  Mild:     { color: C.yellow, dim: C.yellowDim, ring: "#d29922" },
  Moderate: { color: C.orange, dim: C.orangeDim, ring: "#f0883e" },
  Severe:   { color: C.red,    dim: C.redDim,    ring: "#f85149" },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────
function fmt(n, dec = 1) { return typeof n === "number" ? n.toFixed(dec) : "—"; }
function pct(n) { return typeof n === "number" ? Math.round(n * 100) + "%" : "—"; }

function ScoreRing({ score, severity }) {
  const cfg     = SEVERITY[severity] || SEVERITY.Normal;
  const r       = 44;
  const circ    = 2 * Math.PI * r;
  const dash    = circ - (score / 100) * circ;

  return (
    <svg width="112" height="112" viewBox="0 0 112 112" aria-label={`Overall score: ${score} out of 100`}>
      <circle cx="56" cy="56" r={r} fill="none" stroke={C.border} strokeWidth="9" />
      <circle
        cx="56" cy="56" r={r}
        fill="none"
        stroke={cfg.ring}
        strokeWidth="9"
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={dash}
        transform="rotate(-90 56 56)"
        style={{ transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1), stroke 0.4s" }}
      />
      <text x="56" y="52" textAnchor="middle" fill={C.text} fontSize="22" fontWeight="600" fontFamily="system-ui">
        {score ?? "—"}
      </text>
      <text x="56" y="67" textAnchor="middle" fill={C.textSub} fontSize="10" fontFamily="system-ui">
        / 100
      </text>
    </svg>
  );
}

function Badge({ label, severity }) {
  const cfg = SEVERITY[severity] || { color: C.textSub, dim: C.surface };
  return (
    <span style={{
      display: "inline-block", fontSize: 12, fontWeight: 500,
      padding: "3px 10px", borderRadius: 6,
      background: cfg.dim, color: cfg.color,
      border: `1px solid ${cfg.color}33`,
    }}>
      {label}
    </span>
  );
}

function MetricCard({ label, value, sub, highlight }) {
  return (
    <div style={{
      background: C.surfaceHigh, borderRadius: 8,
      border: `1px solid ${highlight ? C.accentDim : C.borderSoft}`,
      padding: "14px 16px", textAlign: "center",
    }}>
      <p style={{ fontSize: 11, color: C.textSub, margin: "0 0 6px", textTransform: "uppercase", letterSpacing: "0.04em" }}>{label}</p>
      <p style={{ fontSize: 24, fontWeight: 600, color: C.text, margin: 0, fontVariantNumeric: "tabular-nums" }}>{value}</p>
      {sub && <p style={{ fontSize: 11, color: C.textMuted, margin: "4px 0 0" }}>{sub}</p>}
    </div>
  );
}

function BiomarkerBar({ label, value, max, unit = "", color }) {
  const pctFill = Math.min(100, Math.round((value / max) * 100));
  const displayVal = typeof value === "number" ? (unit === "%" ? Math.round(value) + "%" : fmt(value) + (unit ? " " + unit : "")) : "—";
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span style={{ fontSize: 12, color: C.textSub }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 500, color: C.text }}>{displayVal}</span>
      </div>
      <div style={{ background: C.bg, borderRadius: 4, height: 6, overflow: "hidden" }}>
        <div style={{
          width: pctFill + "%", height: "100%", borderRadius: 4,
          background: color,
          transition: "width 0.7s cubic-bezier(.4,0,.2,1)",
        }} />
      </div>
    </div>
  );
}

function FFTChart({ spectrum, freqRes }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!spectrum?.length || !canvasRef.current) return;
    const c   = canvasRef.current;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);

    const padL = 32, padR = 12, padT = 20, padB = 28;
    const w = c.width - padL - padR;
    const h = c.height - padT - padB;
    const n = Math.min(spectrum.length, 80);
    const maxMag = Math.max(...spectrum.slice(1, n));

    // axis
    ctx.strokeStyle = C.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + h); ctx.lineTo(padL + w, padT + h);
    ctx.stroke();

    // PD band highlight (3–6 Hz)
    const pdX1 = padL + (3 / (n * freqRes)) * w;
    const pdX2 = padL + (6 / (n * freqRes)) * w;
    ctx.fillStyle = "rgba(248,81,73,0.07)";
    ctx.fillRect(pdX1, padT, pdX2 - pdX1, h);

    // bars
    for (let i = 1; i < n; i++) {
      const freq = i * freqRes;
      const x    = padL + ((i - 1) / (n - 1)) * w;
      const bw   = w / n;
      const bh   = maxMag > 0 ? (spectrum[i] / maxMag) * h : 0;
      ctx.fillStyle = (freq >= 3 && freq <= 6) ? "rgba(248,81,73,0.8)" : "rgba(47,129,247,0.5)";
      ctx.fillRect(x, padT + h - bh, bw - 1, bh);
    }

    // PD band boundary lines
    ctx.strokeStyle = "rgba(248,81,73,0.5)";
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1;
    [pdX1, pdX2].forEach(px => {
      ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, padT + h); ctx.stroke();
    });
    ctx.setLineDash([]);

    // labels
    ctx.fillStyle = C.textMuted;
    ctx.font = "10px system-ui";
    ctx.fillText("3 Hz", pdX1 + 2, padT + h + 18);
    ctx.fillText("6 Hz", pdX2 + 2, padT + h + 18);
    ctx.fillStyle = "rgba(248,81,73,0.6)";
    ctx.fillText("PD band", (pdX1 + pdX2) / 2 - 18, padT + 12);

    // y-axis label
    ctx.save();
    ctx.translate(12, padT + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = C.textMuted;
    ctx.font = "10px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("magnitude", 0, 0);
    ctx.restore();
  }, [spectrum, freqRes]);

  return (
    <div>
      <p style={{ fontSize: 12, color: C.textSub, margin: "0 0 8px", textTransform: "uppercase", letterSpacing: "0.04em" }}>
        Velocity spectrum (FFT)
      </p>
      <canvas
        ref={canvasRef}
        width={600}
        height={150}
        style={{ width: "100%", borderRadius: 8, background: C.surface, border: `1px solid ${C.border}`, display: "block" }}
        aria-label="FFT frequency spectrum. Red bars show the 3–6 Hz PD band."
      />
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function SpiralTest({ patientId = "anonymous", apiBase = "", onResult = null }) {
  const canvasRef     = useRef(null);
  const pointsRef     = useRef([]);
  const isDrawingRef  = useRef(false);
  const startTimeRef  = useRef(null);
  const timerRef      = useRef(null);
  const strokeRef     = useRef(-1);          // stroke index, +1 on every pen-down
  const pathLenRef    = useRef(0);           // running path length (px) for live speed
  const deviceRef     = useRef("unknown");   // pointerType: mouse | touch | pen

  const [tab,       setTab]       = useState("draw");   // "draw" | "results" | "guide"
  const [phase,     setPhase]     = useState("idle");   // "idle" | "drawing" | "done" | "loading" | "error"
  const [live,      setLive]      = useState({ points: 0, time: "0.0s", speed: "—" });
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState(null);

  // ── Draw the guide spiral on canvas ────────────────────────────────────────
  const drawGuide = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Guide spiral (dashed)
    ctx.save();
    ctx.setLineDash([4, 7]);
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    let first = true;
    for (let a = 0; a <= TURNS * 2 * Math.PI; a += 0.04) {
      const r = (a / (TURNS * 2 * Math.PI)) * MAX_RADIUS;
      const x = CX + r * Math.cos(a);
      const y = CY + r * Math.sin(a);
      first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      first = false;
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Centre start dot
    ctx.beginPath();
    ctx.arc(CX, CY, 5, 0, Math.PI * 2);
    ctx.fillStyle = C.accent;
    ctx.fill();

    // Pulse ring on centre dot
    ctx.beginPath();
    ctx.arc(CX, CY, 10, 0, Math.PI * 2);
    ctx.strokeStyle = C.accent + "55";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, []);

  // ── Redraw the whole user path (only on remount / clear, not per move) ──────
  const redrawPath = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const pts = pointsRef.current;
    if (pts.length < 2) return;
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    pts.forEach((p, i) => {
      if (i === 0 || p.stroke !== pts[i - 1].stroke) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }, []);

  // Draw only the newest segment(s): O(1) per event instead of redrawing everything.
  const drawSegment = useCallback((from, newPts) => {
    const canvas = canvasRef.current;
    if (!canvas || !newPts.length) return;
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo((from || newPts[0]).x, (from || newPts[0]).y);
    newPts.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.stroke();
  }, []);

  // The canvas unmounts when another tab is shown; repaint guide + path on return.
  useEffect(() => { if (tab === "draw") { drawGuide(); redrawPath(); } }, [tab, drawGuide, redrawPath]);

  // ── Pointer input ─────────────────────────────────────────────────────────
  // Pointer events cover mouse, touch and stylus. getCoalescedEvents() returns every
  // sample the device produced since the last event (browsers otherwise deliver at
  // most one per display frame), and e.timeStamp is a sub-millisecond timestamp
  // (Date.now() has 1 ms steps and caused zero-interval "speed spikes").
  function toPoint(e, rect, scaleX, scaleY) {
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
      t: e.timeStamp,
      stroke: strokeRef.current,
      ...(e.pointerType === "pen" && e.pressure > 0 ? { p: e.pressure } : {}),
    };
  }

  function addSamples(e) {
    const canvas = canvasRef.current;
    const rect   = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
    const events = typeof e.getCoalescedEvents === "function" ? e.getCoalescedEvents() : [];
    const samples = (events.length ? events : [e]).map(ev => toPoint(ev, rect, sx, sy));
    const pts = pointsRef.current;
    const prev = pts.length && pts[pts.length - 1].stroke === strokeRef.current ? pts[pts.length - 1] : null;
    let last = prev;
    samples.forEach(p => {
      if (last) pathLenRef.current += Math.hypot(p.x - last.x, p.y - last.y);
      last = p;
      pts.push(p);
    });
    drawSegment(prev, samples);
  }

  function onPointerDown(e) {
    if (e.button !== undefined && e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);   // keep receiving moves outside the canvas
    isDrawingRef.current = true;
    deviceRef.current = e.pointerType || "unknown";
    strokeRef.current += 1;
    if (!startTimeRef.current) startTimeRef.current = e.timeStamp;
    setPhase("drawing");
    addSamples(e);

    timerRef.current = timerRef.current || setInterval(() => {
      const pts = pointsRef.current;
      if (!pts.length) return;
      const secs = (pts[pts.length - 1].t - startTimeRef.current) / 1000;
      setLive({
        points: pts.length,
        time: secs.toFixed(1) + "s",
        speed: secs > 0 ? Math.round(pathLenRef.current / secs) : "—",
      });
    }, 100);
  }

  function onPointerMove(e) {
    if (!isDrawingRef.current) return;
    e.preventDefault();
    addSamples(e);
  }

  function onPointerUp(e) {
    if (!isDrawingRef.current) return;
    isDrawingRef.current = false;
    e?.currentTarget?.releasePointerCapture?.(e.pointerId);
    if (pointsRef.current.length > 5) setPhase("done");
  }

  // ── Clear ───────────────────────────────────────────────────────────────────
  function clearDrawing() {
    clearInterval(timerRef.current);
    timerRef.current   = null;
    startTimeRef.current = null;
    pointsRef.current  = [];
    strokeRef.current  = -1;
    pathLenRef.current = 0;
    setPhase("idle");
    setLive({ points: 0, time: "0.0s", speed: "—" });
    setResult(null);
    setError(null);
    drawGuide();
  }

  // ── Analyze ─────────────────────────────────────────────────────────────────
  async function analyze() {
    const pts = pointsRef.current;
    if (pts.length < 30) {
      setError("Draw a full spiral first — at least 30 points needed.");
      return;
    }
    clearInterval(timerRef.current);
    timerRef.current = null;
    setPhase("loading");
    setError(null);

    try {
      const res = await fetch(`${apiBase}/api/spiral-test`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId, hand: "dominant",
          // e.timeStamp counts from page load; send times relative to the first point.
          points: pts.map(p => ({ ...p, t: p.t - pts[0].t })),
          input_device: ["mouse", "touch", "pen"].includes(deviceRef.current) ? deviceRef.current : "unknown",
          center_x: CX, center_y: CY, guide_radius: MAX_RADIUS,
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server error ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
      onResult?.(data);
      setPhase("done");
      setTab("results");
    } catch (err) {
      setError(err.message || "Could not connect to the analysis server.");
      setPhase("done");
    }
  }

  // ─── Render ────────────────────────────────────────────────────────────────
  const bio  = result?.biomarkers;
  const sev  = result?.severity || "Normal";

  const statusMap = {
    idle:    { label: "Ready",          color: C.textSub  },
    drawing: { label: "Drawing…",       color: C.yellow   },
    done:    { label: "Ready to analyze", color: C.green  },
    loading: { label: "Analysing…",     color: C.accent   },
    error:   { label: "Error",          color: C.red      },
  };
  const status = statusMap[phase] || statusMap.idle;

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, sans-serif", background: C.bg, minHeight: "100vh", padding: "24px 20px", color: C.text }}>
      <div style={{ maxWidth: 700, margin: "0 auto" }}>

        {/* ── Header ── */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
          <div>
            <p style={{ fontSize: 11, color: C.textMuted, margin: "0 0 4px", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              NeuroTrack AI  ·  Motor assessment
            </p>
            <h1 style={{ fontSize: 22, fontWeight: 600, margin: 0, color: C.text }}>
              Spiral drawing test
            </h1>
          </div>
          <span style={{
            fontSize: 11, fontWeight: 500, padding: "4px 10px", borderRadius: 20,
            background: C.surfaceHigh, color: status.color,
            border: `1px solid ${status.color}44`,
          }}>
            {status.label}
          </span>
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: "flex", marginBottom: 20, background: C.surfaceHigh, borderRadius: 8, padding: 3, border: `1px solid ${C.border}` }}>
          {["draw", "results", "guide"].map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                flex: 1, padding: "8px 0", borderRadius: 6, border: "none", cursor: "pointer",
                fontSize: 13, fontWeight: 500, transition: "all 0.15s",
                background: tab === t ? C.accent : "transparent",
                color:      tab === t ? "#fff"   : C.textSub,
              }}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        {/* ════════════════════════════════ DRAW TAB ════════════════════════════════ */}
        {tab === "draw" && (
          <>
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "12px 16px", marginBottom: 14 }}>
              <p style={{ fontSize: 13, color: C.textSub, margin: 0, lineHeight: 1.6 }}>
                Start at the <strong style={{ color: C.accent }}>blue dot</strong> in the centre and draw a spiral outward.
                Take your time — don't try to trace perfectly. Lift your finger or release the mouse when you're done.
              </p>
            </div>

            {/* Canvas */}
            <div style={{ position: "relative", marginBottom: 14 }}>
              <canvas
                ref={canvasRef}
                width={CANVAS_W}
                height={CANVAS_H}
                style={{
                  width: "100%", display: "block", borderRadius: 10, cursor: "crosshair",
                  background: C.surface, border: `1px solid ${tab === "draw" && phase === "drawing" ? C.accent + "88" : C.border}`,
                  touchAction: "none",
                  transition: "border-color 0.2s",
                }}
                aria-label="Drawing canvas. Draw a spiral from the centre dot outward."
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
              />
              {phase === "idle" && (
                <div style={{
                  position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
                  pointerEvents: "none", borderRadius: 10,
                }}>
                  <p style={{ fontSize: 13, color: C.textMuted, background: C.bg + "cc", padding: "6px 14px", borderRadius: 20 }}>
                    Tap or click the centre dot to begin
                  </p>
                </div>
              )}
            </div>

            {/* Actions */}
            <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
              <button
                onClick={clearDrawing}
                style={{
                  flex: 1, padding: "10px 0", borderRadius: 8, cursor: "pointer",
                  background: "none", border: `1px solid ${C.border}`,
                  color: C.textSub, fontSize: 14, fontWeight: 500,
                }}
              >
                Clear
              </button>
              <button
                onClick={analyze}
                disabled={phase === "loading" || phase === "idle"}
                style={{
                  flex: 2, padding: "10px 0", borderRadius: 8, cursor: phase === "loading" || phase === "idle" ? "not-allowed" : "pointer",
                  background: phase === "loading" || phase === "idle" ? C.surfaceHigh : C.accent,
                  border: "none", color: phase === "loading" || phase === "idle" ? C.textMuted : "#fff",
                  fontSize: 14, fontWeight: 600, transition: "all 0.2s",
                }}
              >
                {phase === "loading" ? "Analysing…" : "Analyse drawing"}
              </button>
            </div>

            {/* Error */}
            {error && (
              <div style={{ background: C.redDim, border: `1px solid ${C.red}44`, borderRadius: 8, padding: "10px 14px", marginBottom: 14 }}>
                <p style={{ fontSize: 13, color: C.red, margin: 0 }}>{error}</p>
              </div>
            )}

            {/* Live metrics */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
              <MetricCard label="Points"        value={live.points} />
              <MetricCard label="Drawing time"  value={live.time}   />
              <MetricCard label="Avg speed"     value={live.speed}  sub="px / s" />
            </div>
          </>
        )}

        {/* ════════════════════════════════ RESULTS TAB ═════════════════════════════ */}
        {tab === "results" && (
          <>
            {!result ? (
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "48px 24px", textAlign: "center" }}>
                <p style={{ fontSize: 32, margin: "0 0 12px" }}>📊</p>
                <p style={{ fontSize: 14, color: C.textSub, margin: 0 }}>Draw and analyse a spiral to see results here.</p>
              </div>
            ) : (
              <>
                {/* Score card */}
                <div style={{
                  background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
                  padding: "20px 24px", marginBottom: 16,
                  display: "flex", alignItems: "center", gap: 24,
                }}>
                  <ScoreRing score={result.overall_score} severity={sev} />
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                      <span style={{ fontSize: 18, fontWeight: 600, color: C.text }}>
                        {sev} impairment
                      </span>
                      <Badge label={sev} severity={sev} />
                    </div>
                    <p style={{ fontSize: 13, color: C.textSub, margin: "0 0 10px", lineHeight: 1.6 }}>
                      {{
                        Normal:   "Motor function appears within normal range. No significant tremor or irregularity detected.",
                        Mild:     "Slight tremor or drawing irregularity detected. Consider clinical follow-up.",
                        Moderate: "Notable motor impairment detected. Significant deviation from ideal spiral pattern.",
                        Severe:   "Substantial motor impairment. Strong tremor signal and highly irregular spiral pattern.",
                      }[sev]}
                    </p>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      <span style={{ fontSize: 11, color: C.textMuted, background: C.surfaceHigh, border: `1px solid ${C.border}`, borderRadius: 4, padding: "2px 8px" }}>
                        Dominant impairment: <strong style={{ color: C.text }}>{result.dominant_impairment}</strong>
                      </span>
                      <span style={{ fontSize: 11, color: C.textMuted, background: C.surfaceHigh, border: `1px solid ${C.border}`, borderRadius: 4, padding: "2px 8px" }}>
                        Duration: <strong style={{ color: C.text }}>{result.drawing_duration_s}s</strong>
                      </span>
                      <span style={{ fontSize: 11, color: C.textMuted, background: C.surfaceHigh, border: `1px solid ${C.border}`, borderRadius: 4, padding: "2px 8px" }}>
                        Points: <strong style={{ color: C.text }}>{result.point_count}</strong>
                      </span>
                    </div>
                  </div>
                </div>

                {/* Biomarker grid */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10, marginBottom: 16 }}>
                  <MetricCard
                    label="Tremor power (3–6 Hz)"
                    value={pct(bio.tremor_power_ratio)}
                    sub={bio.tremor_power_ratio > 0.2 ? "High — PD indicator" : bio.tremor_power_ratio > 0.1 ? "Moderate" : "Low — normal range"}
                  />
                  <MetricCard
                    label="Spiral regularity"
                    value={pct(bio.spiral_regularity)}
                    sub={bio.spiral_regularity > 0.75 ? "Good consistency" : bio.spiral_regularity > 0.5 ? "Some irregularity" : "High irregularity"}
                  />
                  <MetricCard
                    label="RMSE from ideal"
                    value={fmt(bio.rmse_from_ideal) + " px"}
                    sub={bio.rmse_from_ideal < 15 ? "Close to ideal" : bio.rmse_from_ideal < 35 ? "Moderate deviation" : "High deviation"}
                  />
                  <MetricCard
                    label="Drawing speed"
                    value={fmt(bio.avg_speed_px_per_s, 0) + " px/s"}
                    sub={bio.avg_speed_px_per_s < 40 ? "Slow (not scored)" : bio.avg_speed_px_per_s < 120 ? "Normal range" : "Fast"}
                  />
                </div>

                {/* Biomarker bars */}
                <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "18px 20px", marginBottom: 16 }}>
                  <p style={{ fontSize: 12, color: C.textSub, margin: "0 0 14px", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                    Biomarker breakdown
                  </p>
                  <BiomarkerBar label="Tremor (PD band)"   value={Math.round(bio.tremor_power_ratio * 100)} max={50}  unit="%"    color={C.red}    />
                  <BiomarkerBar label="Spiral regularity"  value={Math.round(bio.spiral_regularity * 100)} max={100} unit="%"    color={C.green}  />
                  <BiomarkerBar label="Path accuracy"      value={Math.max(0, 100 - Math.round(bio.rmse_from_ideal / 0.6))} max={100} unit="%" color={C.accent} />
                  <BiomarkerBar label="Motor speed index"  value={Math.min(100, Math.round(bio.avg_speed_px_per_s / 2))}    max={100} unit="%" color={C.yellow} />
                  <BiomarkerBar label="Zero-crossing rate" value={Math.round(bio.zero_crossing_rate * 100)} max={60}  unit="%"    color={C.orange}  />
                </div>

                {/* FFT chart */}
                <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "18px 20px", marginBottom: 16 }}>
                  <FFTChart spectrum={result.fft_spectrum} freqRes={result.fft_freq_resolution} />
                  <p style={{ fontSize: 11, color: C.textMuted, margin: "8px 0 0" }}>
                    Red bars = 3–6 Hz Parkinson's tremor band. High energy in this region is a primary PD motor biomarker.
                  </p>
                </div>

                {/* Disclaimer */}
                <div style={{ background: C.yellowDim, border: `1px solid ${C.yellow}33`, borderRadius: 8, padding: "10px 14px" }}>
                  <p style={{ fontSize: 12, color: C.yellow, margin: 0, lineHeight: 1.6 }}>
                    ⚠️ For research and screening only. This tool does not replace clinical diagnosis by a qualified neurologist.
                  </p>
                </div>
              </>
            )}
          </>
        )}

        {/* ════════════════════════════════ GUIDE TAB ═══════════════════════════════ */}
        {tab === "guide" && (
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "20px 24px" }}>
            <h2 style={{ fontSize: 15, fontWeight: 600, margin: "0 0 16px", color: C.text }}>How to take the test</h2>
            {[
              ["Sit comfortably", "Rest your arm on a flat surface. Use your dominant hand."],
              ["Start at the centre dot", "On a touchscreen, use a single finger or stylus. On desktop, use a mouse."],
              ["Draw outward", "Begin at the blue dot and spiral outward — like unwinding a coil. Move at a comfortable pace."],
              ["Don't trace perfectly", "Natural tremor and irregularity are part of the data. Don't try to follow the guide exactly."],
              ["Lift when done", "Release your finger or mouse, then tap Analyse drawing."],
            ].map(([title, desc], i) => (
              <div key={i} style={{ display: "flex", gap: 14, marginBottom: 14 }}>
                <div style={{
                  width: 24, height: 24, borderRadius: "50%", background: C.accentDim,
                  color: C.accent, fontSize: 11, fontWeight: 600, display: "flex",
                  alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1,
                }}>
                  {i + 1}
                </div>
                <div>
                  <p style={{ fontSize: 13, fontWeight: 500, color: C.text, margin: "0 0 2px" }}>{title}</p>
                  <p style={{ fontSize: 13, color: C.textSub, margin: 0, lineHeight: 1.6 }}>{desc}</p>
                </div>
              </div>
            ))}

            <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 16, marginTop: 6 }}>
              <h3 style={{ fontSize: 13, fontWeight: 600, color: C.text, margin: "0 0 10px" }}>What we measure</h3>
              {[
                ["Tremor power", "FFT analysis isolates energy in the 3–6 Hz band — the characteristic Parkinson's tremor frequency. Higher band power indicates more tremor."],
                ["Spiral regularity", "Tracks whether the radius grows consistently. Irregular growth suggests rigidity or micrographia (progressively shrinking handwriting)."],
                ["RMSE from ideal", "Measures deviation from a mathematically perfect Archimedes spiral. Higher values mean larger overall motor impairment."],
                ["Drawing speed", "Bradykinesia (slowness of movement) is one of the cardinal PD symptoms. Speed below ~40 px/s is a flag."],
                ["Zero-crossing rate", "Counts how often the velocity signal changes direction — a secondary tremor intensity proxy."],
              ].map(([title, desc]) => (
                <div key={title} style={{ marginBottom: 10 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: C.text }}>{title}</span>
                  <span style={{ fontSize: 12, color: C.textSub }}> — {desc}</span>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}