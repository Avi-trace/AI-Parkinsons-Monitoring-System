/**
 * NeuroTrack AI — Patient Dashboard
 *
 * Aggregates results from all three tests (finger tap, spiral, postural tremor)
 * into a single composite PD risk score with session history timeline.
 *
 * Usage:
 *   <Dashboard
 *     patientId="PT001"
 *     apiBase=""                  // empty = use Vite proxy
 *     fingerResult={...}          // pass result objects from each test component
 *     spiralResult={...}          // (optional — show whatever tests were completed)
 *     tremorResult={...}
 *   />
 *
 * The component also auto-fetches history from GET /api/patient/:id/history
 * and can save a session via POST /api/session.
 */

import { useState, useEffect, useRef } from "react";

// ─── Tokens ──────────────────────────────────────────────────────────────────
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
  teal:        "#39c5bb",   // composite score accent — distinct from per-test blue
  green:       "#3fb950",
  yellow:      "#d29922",
  orange:      "#f0883e",
  red:         "#f85149",
  greenDim:    "#122c1a",
  yellowDim:   "#2d1f07",
  orangeDim:   "#2d1505",
  redDim:      "#2c0d0d",
  tealDim:     "#0d2622",
};

const RISK = {
  Low:       { color: C.green,  dim: C.greenDim,  label: "Low risk"       },
  Moderate:  { color: C.yellow, dim: C.yellowDim, label: "Moderate risk"  },
  High:      { color: C.orange, dim: C.orangeDim, label: "High risk"      },
  "Very High":{ color: C.red,   dim: C.redDim,    label: "Very high risk" },
  Unknown:   { color: C.textSub,dim: C.surface,   label: "Incomplete"     },
};

const TEST_META = {
  finger: { label: "Finger tap",      icon: "👆", desc: "Bradykinesia & rhythm" },
  spiral: { label: "Spiral drawing",  icon: "🌀", desc: "Tremor & rigidity"     },
  tremor: { label: "Postural tremor", icon: "✋", desc: "Action tremor & asymmetry" },
};

function riskLevel(score) {
  if (score === null || score === undefined) return "Unknown";
  if (score >= 80) return "Low";
  if (score >= 60) return "Moderate";
  if (score >= 40) return "High";
  return "Very High";
}

function composite(f, s, t) {
  const vals = [f, s, t].filter(v => v !== null && v !== undefined);
  if (!vals.length) return null;
  return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
}

// ─── Composite gauge (SVG arc) ────────────────────────────────────────────────

// ─── Tests completed counter ──────────────────────────────────────────────────
function TestsCompleted({ done, total = 3, comp, risk }) {
  const cfg   = RISK[risk] || RISK.Unknown;
  const steps = Array.from({ length: total }, (_, i) => i < done);
  return (
    <div style={{ textAlign: "center", padding: "8px 0" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "center", gap: 6, marginBottom: 6 }}>
        <span style={{ fontSize: 52, fontWeight: 700, color: done === total ? C.teal : C.text, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>
          {done}
        </span>
        <span style={{ fontSize: 22, color: C.textMuted, fontWeight: 400 }}>/ {total}</span>
      </div>
      <p style={{ fontSize: 11, color: C.textMuted, margin: "0 0 14px", textTransform: "uppercase", letterSpacing: "0.06em" }}>
        tests completed
      </p>
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 14 }}>
        {steps.map((filled, i) => (
          <div key={i} style={{
            width: 36, height: 6, borderRadius: 3,
            background: filled ? C.teal : C.border,
            transition: "background 0.4s",
          }} />
        ))}
      </div>
      {comp !== null && (
        <div style={{
          background: cfg.dim, border: `1px solid ${cfg.color}44`,
          borderRadius: 8, padding: "8px 14px", display: "inline-block",
        }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: cfg.color }}>
            Composite: {comp}/100 — {cfg.label}
          </span>
        </div>
      )}
    </div>
  );
}

// ─── Test score card ──────────────────────────────────────────────────────────
function TestCard({ testKey, score, result, onRun }) {
  const meta = TEST_META[testKey];
  const risk = riskLevel(score);
  const cfg  = RISK[risk];
  const done = score !== null && score !== undefined;

  return (
    <div style={{
      background: C.surface, border: `1px solid ${done ? cfg.color + "44" : C.border}`,
      borderRadius: 10, padding: "18px 20px", flex: 1, minWidth: 0,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
        <div>
          <span style={{ fontSize: 20 }}>{meta.icon}</span>
          <p style={{ fontSize: 13, fontWeight: 600, color: C.text, margin: "4px 0 2px" }}>{meta.label}</p>
          <p style={{ fontSize: 11, color: C.textMuted, margin: 0 }}>{meta.desc}</p>
        </div>
        {done && (
          <span style={{
            fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 12,
            background: cfg.dim, color: cfg.color, border: `1px solid ${cfg.color}33`,
          }}>{risk}</span>
        )}
      </div>

      {done ? (
        <>
          <div style={{ display: "flex", alignItems: "baseline", gap: 4, marginBottom: 10 }}>
            <span style={{ fontSize: 36, fontWeight: 700, color: cfg.color, fontVariantNumeric: "tabular-nums" }}>{score}</span>
            <span style={{ fontSize: 13, color: C.textMuted }}>/100</span>
          </div>
          {/* Mini biomarker hints */}
          {testKey === "spiral" && result?.biomarkers && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {[
                ["Tremor",      Math.round(result.biomarkers.tremor_power_ratio * 100) + "%"],
                ["RMSE",        Math.round(result.biomarkers.rmse_from_ideal) + " px"],
                ["Regularity",  Math.round(result.biomarkers.spiral_regularity * 100) + "%"],
                ["Speed",       Math.round(result.biomarkers.avg_speed_px_per_s) + " px/s"],
              ].map(([k, v]) => (
                <div key={k}>
                  <p style={{ fontSize: 10, color: C.textMuted, margin: "0 0 1px" }}>{k}</p>
                  <p style={{ fontSize: 12, fontWeight: 500, color: C.textSub, margin: 0 }}>{v}</p>
                </div>
              ))}
            </div>
          )}
          {testKey === "tremor" && result && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {[
                ["Dominant side", result.dominant_side],
                ["Asymmetry",     Math.round((result.asymmetry_index ?? 0) * 100) + "%"],
                ["Left",          result.left?.stability_score + "/100"],
                ["Right",         result.right?.stability_score + "/100"],
              ].map(([k, v]) => (
                <div key={k}>
                  <p style={{ fontSize: 10, color: C.textMuted, margin: "0 0 1px", textTransform: "capitalize" }}>{k}</p>
                  <p style={{ fontSize: 12, fontWeight: 500, color: C.textSub, margin: 0, textTransform: "capitalize" }}>{v ?? "—"}</p>
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <div style={{ textAlign: "center", padding: "12px 0" }}>
          <p style={{ fontSize: 12, color: C.textMuted, margin: "0 0 12px" }}>Not completed</p>
          {onRun && (
            <button onClick={() => onRun(testKey)} style={{
              padding: "7px 18px", borderRadius: 6, border: `1px solid ${C.accent}`,
              background: "none", color: C.accent, fontSize: 12, cursor: "pointer",
            }}>
              Run test →
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ─── History sparkline (SVG) ──────────────────────────────────────────────────
function Sparkline({ data, color, width = 200, height = 48 }) {
  if (!data?.length) return null;
  const pad = 4;
  const w = width - pad * 2, h = height - pad * 2;
  const min = 0, max = 100;
  const pts = data.map((v, i) => {
    const x = pad + (i / Math.max(data.length - 1, 1)) * w;
    const y = pad + (1 - (v - min) / (max - min)) * h;
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      {data.map((v, i) => {
        const x = pad + (i / Math.max(data.length - 1, 1)) * w;
        const y = pad + (1 - (v - min) / (max - min)) * h;
        return <circle key={i} cx={x} cy={y} r={i === data.length - 1 ? 3.5 : 2} fill={color} />;
      })}
    </svg>
  );
}

// ─── History row ──────────────────────────────────────────────────────────────
function HistoryRow({ session, isLatest }) {
  const risk = session.risk_level || "Unknown";
  const cfg  = RISK[risk];
  const date = new Date(session.created_at + "Z").toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric",
  });
  const time = new Date(session.created_at + "Z").toLocaleTimeString("en-GB", {
    hour: "2-digit", minute: "2-digit",
  });

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 16, padding: "12px 16px",
      background: isLatest ? C.tealDim : "none",
      border: `1px solid ${isLatest ? C.teal + "44" : C.borderSoft}`,
      borderRadius: 8, marginBottom: 8,
    }}>
      <div style={{ minWidth: 72, textAlign: "right" }}>
        <p style={{ fontSize: 11, color: C.textMuted, margin: 0 }}>{date}</p>
        <p style={{ fontSize: 10, color: C.textMuted, margin: 0 }}>{time}</p>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 80 }}>
        <span style={{
          fontSize: 22, fontWeight: 700, color: cfg.color, fontVariantNumeric: "tabular-nums"
        }}>{session.composite ?? "—"}</span>
        <span style={{ fontSize: 10, color: C.textMuted }}>/100</span>
      </div>

      <div style={{ display: "flex", gap: 10, flex: 1 }}>
        {[["F", session.finger_score, C.accent], ["S", session.spiral_score, C.yellow], ["T", session.tremor_score, C.green]].map(([k, v, c]) => (
          <div key={k} style={{ textAlign: "center" }}>
            <p style={{ fontSize: 9, color: C.textMuted, margin: "0 0 2px" }}>{k}</p>
            <p style={{ fontSize: 12, fontWeight: 600, color: v != null ? c : C.textMuted, margin: 0, fontVariantNumeric: "tabular-nums" }}>
              {v ?? "—"}
            </p>
          </div>
        ))}
      </div>

      <span style={{
        fontSize: 10, fontWeight: 500, padding: "2px 8px", borderRadius: 10,
        background: cfg.dim, color: cfg.color, border: `1px solid ${cfg.color}33`, whiteSpace: "nowrap",
      }}>
        {cfg.label}
      </span>

      {isLatest && (
        <span style={{ fontSize: 9, color: C.teal, fontWeight: 600 }}>LATEST</span>
      )}
    </div>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────
export default function Dashboard({
  patientId    = "PT001",
  apiBase      = "",
  fingerResult = null,
  spiralResult = null,
  tremorResult = null,
  onNavigate   = null,   // (testKey) => void — called when "Run test →" clicked
}) {
  const [history,  setHistory]  = useState([]);
  const [saving,   setSaving]   = useState(false);
  const [saved,    setSaved]    = useState(false);
  const [loadingH, setLoadingH] = useState(true);

  const fingerScore = fingerResult?.overall_score ?? null;
  const spiralScore = spiralResult?.overall_score ?? null;
  const tremorScore = tremorResult?.overall_score ?? null;
  const comp        = composite(fingerScore, spiralScore, tremorScore);
  const risk        = riskLevel(comp);
  const riskCfg     = RISK[risk];
  const testsComplete = [fingerScore, spiralScore, tremorScore].filter(v => v !== null).length;

  // ── Fetch patient history ────────────────────────────────────────────────
  useEffect(() => {
    setLoadingH(true);
    fetch(`${apiBase}/api/patient/${patientId}/history?limit=10`)
      .then(r => r.ok ? r.json() : [])
      .then(data => { setHistory(data); setLoadingH(false); })
      .catch(() => setLoadingH(false));
  }, [patientId, apiBase, saved]);

  // ── Save current session ─────────────────────────────────────────────────
  async function saveSession() {
    if (testsComplete === 0 || saving) return;
    setSaving(true);
    try {
      await fetch(`${apiBase}/api/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          finger: fingerResult,
          spiral: spiralResult,
          tremor: tremorResult,
        }),
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } finally {
      setSaving(false);
    }
  }

  // ── Trend data for sparklines (oldest first) ────────────────────────────
  const historyAsc = [...history].reverse();
  const trendComp   = historyAsc.map(h => h.composite).filter(Boolean);
  const trendFinger = historyAsc.map(h => h.finger_score).filter(Boolean);
  const trendSpiral = historyAsc.map(h => h.spiral_score).filter(Boolean);
  const trendTremor = historyAsc.map(h => h.tremor_score).filter(Boolean);

  // ── Insight text ─────────────────────────────────────────────────────────
  function insight() {
    if (testsComplete === 0) return "Run all three tests to get a composite PD risk assessment.";
    if (testsComplete < 3)   return `${3 - testsComplete} test${3 - testsComplete > 1 ? "s" : ""} remaining for a complete assessment.`;
    if (comp >= 80) return "All three motor domains appear within normal range. Continue monitoring regularly.";
    if (comp >= 60) return "Mild motor impairment detected in one or more domains. Consider clinical follow-up.";
    if (comp >= 40) return "Moderate motor impairment across domains. Clinical evaluation is recommended.";
    return "Significant motor impairment detected. Urgent clinical evaluation is strongly recommended.";
  }


  return (
    <div style={{ fontFamily: "system-ui,-apple-system,sans-serif", background: C.bg, minHeight: "100vh", padding: "24px 20px", color: C.text }}>
      <div style={{ maxWidth: 760, margin: "0 auto" }}>

        {/* ── Header ── */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
          <div>
            <p style={{ fontSize: 11, color: C.textMuted, margin: "0 0 4px", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              NeuroTrack AI
            </p>
            <h1 style={{ fontSize: 24, fontWeight: 700, margin: "0 0 4px", color: C.text }}>
              Patient dashboard
            </h1>
            <p style={{ fontSize: 13, color: C.textMuted, margin: 0 }}>
              ID: <span style={{ color: C.textSub, fontFamily: "monospace" }}>{patientId}</span>
              &nbsp;·&nbsp;
              {testsComplete}/3 tests completed
            </p>
          </div>
          <button
            onClick={saveSession}
            disabled={testsComplete === 0 || saving}
            style={{
              padding: "9px 18px", borderRadius: 8, border: "none", fontSize: 13, fontWeight: 600, cursor: "pointer",
              background: saved ? C.greenDim : testsComplete > 0 ? C.teal : C.surfaceHigh,
              color: saved ? C.green : testsComplete > 0 ? "#fff" : C.textMuted,
              border: `1px solid ${saved ? C.green : testsComplete > 0 ? C.teal : C.border}`,
              transition: "all 0.2s",
            }}
          >
            {saved ? "✓ Saved" : saving ? "Saving…" : "Save session"}
          </button>
        </div>

        {/* ── Composite score + test cards ── */}
        <div style={{ display: "flex", gap: 16, marginBottom: 20, flexWrap: "wrap" }}>

          {/* Tests completed panel */}
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 12, padding: "20px 24px", display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", minWidth: 220,
          }}>
            <TestsCompleted done={testsComplete} total={3} comp={comp} risk={risk} />
          </div>

          {/* Test cards */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10, minWidth: 260 }}>
            {["finger","spiral","tremor"].map(key => (
              <TestCard
                key={key}
                testKey={key}
                score={key === "finger" ? fingerScore : key === "spiral" ? spiralScore : tremorScore}
                result={key === "finger" ? fingerResult : key === "spiral" ? spiralResult : tremorResult}
                onRun={onNavigate}
              />
            ))}
          </div>
        </div>

        {/* ── Insight banner ── */}
        <div style={{
          background: comp !== null ? riskCfg.dim : C.surfaceHigh,
          border: `1px solid ${comp !== null ? riskCfg.color + "44" : C.border}`,
          borderRadius: 10, padding: "14px 18px", marginBottom: 20,
          display: "flex", alignItems: "flex-start", gap: 12,
        }}>
          <span style={{ fontSize: 16, marginTop: 1 }}>
            {comp === null ? "ℹ️" : comp >= 80 ? "✅" : comp >= 60 ? "⚠️" : "🔴"}
          </span>
          <p style={{ fontSize: 13, color: comp !== null ? riskCfg.color : C.textSub, margin: 0, lineHeight: 1.6 }}>
            {insight()}
          </p>
        </div>

        {/* ── Trend sparklines ── */}
        {trendComp.length > 1 && (
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "18px 20px", marginBottom: 20 }}>
            <p style={{ fontSize: 12, color: C.textSub, margin: "0 0 16px", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Score trends  ({trendComp.length} sessions)
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 16 }}>
              {[
                ["Composite",       trendComp,   C.teal  ],
                ["Finger tap",      trendFinger, C.accent ],
                ["Spiral drawing",  trendSpiral, C.yellow ],
                ["Postural tremor", trendTremor, C.green  ],
              ].map(([label, data, color]) => data.length > 1 ? (
                <div key={label}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 11, color: C.textSub }}>{label}</span>
                    <span style={{ fontSize: 11, fontWeight: 600, color, fontVariantNumeric: "tabular-nums" }}>
                      {data[data.length - 1]}
                    </span>
                  </div>
                  <Sparkline data={data} color={color} width={280} height={44} />
                </div>
              ) : null)}
            </div>
          </div>
        )}

        {/* ── Session history ── */}
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "18px 20px" }}>
          <p style={{ fontSize: 12, color: C.textSub, margin: "0 0 14px", textTransform: "uppercase", letterSpacing: "0.04em" }}>
            Session history
          </p>

          {loadingH ? (
            <p style={{ fontSize: 13, color: C.textMuted, textAlign: "center", padding: "20px 0" }}>Loading…</p>
          ) : history.length === 0 ? (
            <p style={{ fontSize: 13, color: C.textMuted, textAlign: "center", padding: "20px 0" }}>
              No sessions saved yet. Complete tests and tap "Save session".
            </p>
          ) : (
            <>
              {/* Legend */}
              <div style={{ display: "flex", gap: 14, marginBottom: 10, fontSize: 10, color: C.textMuted }}>
                <span><span style={{ color: C.accent }}>F</span> = Finger tap</span>
                <span><span style={{ color: C.yellow }}>S</span> = Spiral</span>
                <span><span style={{ color: C.green }}>T</span> = Tremor</span>
              </div>
              {history.map((h, i) => (
                <HistoryRow key={h.id} session={h} isLatest={i === 0} />
              ))}
            </>
          )}
        </div>

        {/* Disclaimer */}
        <div style={{ background: C.yellowDim, border: `1px solid ${C.yellow}33`, borderRadius: 8, padding: "10px 14px", marginTop: 16 }}>
          <p style={{ fontSize: 12, color: C.yellow, margin: 0, lineHeight: 1.6 }}>
            ⚠️ NeuroTrack AI is a research and screening tool only. Composite scores are not a clinical diagnosis. Always consult a qualified neurologist for medical decisions.
          </p>
        </div>

      </div>
    </div>
  );
}