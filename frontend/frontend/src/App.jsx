import { useState } from "react";
import Dashboard    from "./components/Dashboard";
import SpiralTest   from "./components/SpiralTest";
import TremorTest   from "./components/TremorTest";
import FingerTapTest from "./components/FingerTapTest";

const API_BASE   = "";
const PATIENT_ID = "PT001";

const TABS = [
  { key: "dashboard", label: "Dashboard", icon: "📊" },
  { key: "finger",    label: "Finger tap", icon: "👆" },
  { key: "spiral",    label: "Spiral",     icon: "🌀" },
  { key: "tremor",    label: "Tremor",     icon: "✋" },
];

const C = {
  bg: "#0d1117", surface: "#161b22", border: "#30363d",
  text: "#e6edf3", textSub: "#8b949e", teal: "#39c5bb",
};

export default function App() {
  const [tab, setTab] = useState("dashboard");

  // Results lifted here so Dashboard always sees latest — even after tab switches
  const [fingerResult, setFingerResult] = useState(null);
  const [spiralResult, setSpiralResult] = useState(null);
  const [tremorResult, setTremorResult] = useState(null);

  function handleNavigate(testKey) { setTab(testKey); }

  return (
    <div style={{ fontFamily: "system-ui,-apple-system,sans-serif", background: C.bg, minHeight: "100vh" }}>

      {/* Sticky nav */}
      <nav style={{
        background: C.surface, borderBottom: `1px solid ${C.border}`,
        padding: "0 20px", display: "flex", alignItems: "center", gap: 4,
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ marginRight: 16, padding: "12px 0" }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: C.teal, letterSpacing: "-0.02em" }}>NeuroTrack</span>
          <span style={{ fontSize: 13, color: C.textSub }}> AI</span>
        </div>
        {TABS.map(t => {
          const active = tab === t.key;
          const done   = (t.key === "finger" && fingerResult) ||
                         (t.key === "spiral" && spiralResult) ||
                         (t.key === "tremor" && tremorResult);
          return (
            <button key={t.key} onClick={() => setTab(t.key)} style={{
              padding: "14px 14px", border: "none", background: "none", cursor: "pointer",
              fontSize: 13, fontWeight: active ? 600 : 400,
              color: active ? C.teal : C.textSub,
              borderBottom: active ? `2px solid ${C.teal}` : "2px solid transparent",
              position: "relative", transition: "color 0.15s",
            }}>
              <span style={{ marginRight: 5 }}>{t.icon}</span>
              {t.label}
              {/* Green dot when test is done */}
              {done && (
                <span style={{
                  position: "absolute", top: 10, right: 6,
                  width: 5, height: 5, borderRadius: "50%",
                  background: active ? "transparent" : C.teal,
                }} />
              )}
            </button>
          );
        })}
      </nav>

      {/*
        All four tabs stay mounted (only visibility toggles) so results survive tab
        switches. `active` tells a test whether it is visible: hidden tests stop the
        shared camera/GPU tracker instead of running in the background.
      */}
      <main>
        <div style={{ display: tab === "dashboard" ? "block" : "none" }}>
          <Dashboard
            patientId={PATIENT_ID}
            apiBase={API_BASE}
            fingerResult={fingerResult}
            spiralResult={spiralResult}
            tremorResult={tremorResult}
            onNavigate={handleNavigate}
          />
        </div>

        <div style={{ display: tab === "finger" ? "block" : "none" }}>
          <FingerTapTest
            active={tab === "finger"}
            patientId={PATIENT_ID}
            apiBase={API_BASE}
            onResult={setFingerResult}
          />
        </div>

        <div style={{ display: tab === "spiral" ? "block" : "none" }}>
          <SpiralTest
            active={tab === "spiral"}
            patientId={PATIENT_ID}
            apiBase={API_BASE}
            onResult={setSpiralResult}
          />
        </div>

        <div style={{ display: tab === "tremor" ? "block" : "none" }}>
          <TremorTest
            active={tab === "tremor"}
            patientId={PATIENT_ID}
            apiBase={API_BASE}
            onResult={setTremorResult}
          />
        </div>
      </main>
    </div>
  );
}