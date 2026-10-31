import cv2
import numpy as np
import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, windows
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import IsolationForest
from collections import deque
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw
from datetime import datetime

# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------
TEST_DURATION       = 30
SENSITIVITY         = 0.04
MIN_TAP_INTERVAL    = 0.08      # fast-tap floor
BASELINE_SECONDS    = 5         # first N seconds = personal baseline
FFT_BUFFER_LEN      = 128       # power of 2

# Persistent history file — lives next to this script
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pd_history.json")

# Minimum past sessions needed before the IF model is useful
IF_MIN_SESSIONS = 3             # model only fits once we have ≥ 3 complete hand-sessions
IF_CONTAMINATION = 0.1          # assumed fraction of anomalous sessions in history

hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.8, min_tracking_confidence=0.8,
)

active_mode  = 1
mode_names   = {1: "FINGER TAPPING", 2: "FIST OPEN/CLOSE", 3: "PRONATION/SUPINATION"}

# Rolling signal buffers (windowed FFT + Hilbert)
L_BUFFER, R_BUFFER = deque(maxlen=FFT_BUFFER_LEN), deque(maxlen=FFT_BUFFER_LEN)

# Real fps tracking
frame_times = deque(maxlen=60)

# Full-session arrays (for end-of-run report)
timestamps, raw_l, raw_r = [], [], []

# Tap counting state
count_l, count_r = 0, 0
prev_val_l, prev_val_r = 0.0, 0.0
is_opening_l, is_opening_r = False, False
last_tap_l, last_tap_r = 0.0, 0.0
tap_times_l, tap_times_r = [], []

# Personal baseline (set once per mode)
baseline     = {"Left": None, "Right": None}
mode_start_time = time.time()

# Live display values
l_tremor, l_freq, l_hilb_freq, l_hilb_amp = 0.0, 0.0, 0.0, 0.0
r_tremor, r_freq, r_hilb_freq, r_hilb_amp = 0.0, 0.0, 0.0, 0.0


# -----------------------------------
# 2. PERSISTENT HISTORY + ISOLATION FOREST
# -----------------------------------

def load_history() -> list[dict]:
    """Load all past session records from disk. Returns [] if file absent/corrupt."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, IOError):
        pass
    return []


def save_history(history: list[dict]):
    """Persist the full history list back to disk (atomic write via temp file)."""
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, HISTORY_FILE)   # atomic on POSIX; best-effort on Windows


def history_to_feature_matrix(history: list[dict]) -> np.ndarray | None:
    """
    Convert history records to a 2-D feature matrix for Isolation Forest.

    Features per record (7 total):
        0  tap_count          – raw count
        1  avg_speed          – Hz
        2  speed_decay        – fraction [0, 1]
        3  amp_decay          – fraction [0, 1]
        4  tremor_power       – FFT band power in 4-9 Hz
        5  hilbert_freq       – mean instantaneous freq (Hz)
        6  hilbert_freq_dev   – % deviation from that session's personal baseline
                                (0 if no baseline was available)
    """
    rows = []
    for rec in history:
        try:
            rows.append([
                float(rec["tap_count"]),
                float(rec["avg_speed"]),
                float(rec["speed_decay"]),
                float(rec["amp_decay"]),
                float(rec["tremor_power"]),
                float(rec["hilbert_freq"]),
                float(rec.get("hilbert_freq_dev", 0.0)),
            ])
        except (KeyError, TypeError, ValueError):
            continue   # skip malformed records silently
    if len(rows) < IF_MIN_SESSIONS:
        return None
    return np.array(rows, dtype=float)


def build_if_model(history: list[dict]) -> IsolationForest | None:
    """Fit and return an IsolationForest on all valid historical records, or None."""
    X = history_to_feature_matrix(history)
    if X is None:
        return None
    contamination = min(IF_CONTAMINATION, max(0.01, (len(X) - IF_MIN_SESSIONS) / len(X)))
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X)
    return clf


def score_session(clf: IsolationForest | None, record: dict) -> tuple[float | None, bool]:
    """
    Return (anomaly_score, is_anomaly).
    anomaly_score: raw IF decision function value (lower = more anomalous)
    is_anomaly: True if IF classifies this point as an outlier
    Returns (None, False) when the model isn't ready yet.
    """
    if clf is None:
        return None, False
    X = history_to_feature_matrix([record])
    if X is None:
        return None, False
    score = float(clf.decision_function(X)[0])
    label = clf.predict(X)[0]   # -1 = anomaly, 1 = normal
    return score, label == -1


# Load history and pre-train the model at startup
session_history = load_history()
if_model        = build_if_model(session_history)
print(f"📂 Loaded {len(session_history)} past session(s) from history.")
if if_model is not None:
    print(f"🧠 Isolation Forest ready — trained on {len(session_history)} session(s).")
else:
    need = IF_MIN_SESSIONS - len(session_history)
    print(f"⏳ Need {need} more complete session(s) before Isolation Forest activates.")


# -----------------------------------
# 3. SIGNAL ANALYSIS UTILITIES
# -----------------------------------

def estimate_fps():
    if len(frame_times) < 2:
        return 30.0
    return (len(frame_times) - 1) / (frame_times[-1] - frame_times[0] + 1e-6)


def analyze_window(buffer, fs):
    """Windowed FFT + Hilbert on the rolling buffer.
    Returns: tremor_power, dom_freq, hilbert_inst_freq, hilbert_inst_amp
    """
    if len(buffer) < 32:
        return 0.0, 0.0, 0.0, 0.0

    sig = np.array(buffer, dtype=float)
    sig = sig - np.mean(sig)

    win      = windows.hann(len(sig))
    windowed = sig * win
    fft_res  = np.abs(np.fft.rfft(windowed, n=FFT_BUFFER_LEN))
    freqs    = np.fft.rfftfreq(FFT_BUFFER_LEN, d=1 / fs)

    band_mask    = (freqs >= 4) & (freqs <= 9)
    tremor_power = np.sum(fft_res[band_mask]) if band_mask.any() else 0.0
    dom_freq     = freqs[np.argmax(fft_res)] if np.max(fft_res) > 1e-3 else 0.0

    analytic   = hilbert(sig)
    envelope   = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq  = np.diff(inst_phase) / (2.0 * np.pi) * fs

    hilb_amp  = float(np.mean(envelope))
    hilb_freq = float(np.mean(np.abs(inst_freq))) if len(inst_freq) > 0 else 0.0

    return float(tremor_power), float(dom_freq), hilb_freq, hilb_amp


def update_baseline(side, tremor, hilb_freq, hilb_amp, mode_elapsed):
    if baseline[side] is None and mode_elapsed > BASELINE_SECONDS:
        baseline[side] = {
            "tremor": max(tremor, 1e-4),
            "freq":   max(hilb_freq, 1e-4),
            "amp":    max(hilb_amp, 1e-4),
        }


def deviation_pct(value, base):
    if base is None:
        return None
    return (value - base) / base * 100.0


# -----------------------------------
# 4. CAMERA LOOP
# -----------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("PRO ENGINE v4. Modes: [1] Tap [2] Fist [3] Rotate | [E] Export | [Q] Quit")
start_time      = time.time()
mode_start_time = start_time

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_times.append(time.time())
    fps_est = estimate_fps()

    frame = cv2.flip(cv2.resize(frame, (1280, 720)), 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    elapsed      = time.time() - start_time
    rem          = max(0, TEST_DURATION - elapsed)
    mode_elapsed = time.time() - mode_start_time

    cv2.rectangle(frame, (0, 0), (w, 140), (15, 15, 15), -1)
    cv2.line(frame, (0, 140), (w, 140), (0, 255, 0), 2)

    cur_l, cur_r = 0.0, 0.0

    if res.multi_hand_landmarks:
        for idx, landmarks in enumerate(res.multi_hand_landmarks):
            side = res.multi_handedness[idx].classification[0].label

            wrist   = landmarks.landmark[0]
            thumb   = landmarks.landmark[4]
            index   = landmarks.landmark[8]
            mcp_mid = landmarks.landmark[9]
            scale   = np.sqrt((wrist.x - mcp_mid.x) ** 2 + (wrist.y - mcp_mid.y) ** 2) + 0.001

            if active_mode == 1:
                val = np.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) / scale
            elif active_mode == 2:
                val = sum(
                    np.sqrt((landmarks.landmark[i].x - wrist.x) ** 2 +
                            (landmarks.landmark[i].y - wrist.y) ** 2)
                    for i in [8, 12, 16, 20]
                ) / (4 * scale)
            else:
                val = (landmarks.landmark[5].x - landmarks.landmark[17].x) / scale

            if side == "Left":
                cur_l = val
                L_BUFFER.append(val)
                now   = time.time()
                delta = cur_l - prev_val_l
                if delta > SENSITIVITY:
                    is_opening_l = True
                elif delta < -SENSITIVITY and is_opening_l and (now - last_tap_l) > MIN_TAP_INTERVAL:
                    count_l += 1
                    tap_times_l.append(now)
                    last_tap_l   = now
                    is_opening_l = False
                prev_val_l = cur_l
            else:
                cur_r = val
                R_BUFFER.append(val)
                now   = time.time()
                delta = cur_r - prev_val_r
                if delta > SENSITIVITY:
                    is_opening_r = True
                elif delta < -SENSITIVITY and is_opening_r and (now - last_tap_r) > MIN_TAP_INTERVAL:
                    count_r += 1
                    tap_times_r.append(now)
                    last_tap_r   = now
                    is_opening_r = False
                prev_val_r = cur_r

            color = (0, 255, 0) if side == "Left" else (0, 100, 255)
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=color, thickness=2))

    l_tremor, l_freq, l_hilb_freq, l_hilb_amp = analyze_window(L_BUFFER, fps_est)
    r_tremor, r_freq, r_hilb_freq, r_hilb_amp = analyze_window(R_BUFFER, fps_est)

    update_baseline("Left",  l_tremor, l_hilb_freq, l_hilb_amp, mode_elapsed)
    update_baseline("Right", r_tremor, r_hilb_freq, r_hilb_amp, mode_elapsed)

    l_dev = deviation_pct(l_hilb_freq, baseline["Left"]["freq"]  if baseline["Left"]  else None)
    r_dev = deviation_pct(r_hilb_freq, baseline["Right"]["freq"] if baseline["Right"] else None)

    # --- IF model: live session snapshot score (updates every frame, shown in UI) ---
    def live_record(count, avg_sp, s_decay, a_decay, tremor, hfreq, dev):
        return {
            "tap_count":      count,
            "avg_speed":      avg_sp,
            "speed_decay":    s_decay,
            "amp_decay":      a_decay,
            "tremor_power":   tremor,
            "hilbert_freq":   hfreq,
            "hilbert_freq_dev": dev if dev is not None else 0.0,
        }

    # Build approximate live records from current running stats
    # (speeds computed simply from count / elapsed for live display)
    l_live_speed = count_l / max(elapsed, 1.0)
    r_live_speed = count_r / max(elapsed, 1.0)
    l_live_rec = live_record(count_l, l_live_speed, 0, 0, l_tremor, l_hilb_freq, l_dev)
    r_live_rec = live_record(count_r, r_live_speed, 0, 0, r_tremor, r_hilb_freq, r_dev)
    l_if_score, l_if_anom = score_session(if_model, l_live_rec)
    r_if_score, r_if_anom = score_session(if_model, r_live_rec)

    # --- UI ---
    cv2.putText(frame, f"TASK: {mode_names[active_mode]}", (20, 45),  1, 2,   (255, 255, 255), 2)
    cv2.putText(frame, f"TIME: {rem:.1f}s | FPS: {fps_est:.0f}",
                (20, 85), 1, 1.3, (0, 255, 255), 2)

    # IF status bar (top right corner)
    if_label = f"IF: {len(session_history)} sessions" if if_model is None else \
               f"IF: trained ({len(session_history)} sess)"
    cv2.putText(frame, if_label, (w - 350, 45), 1, 1.1, (180, 120, 255), 2)

    def draw_hand_panel(x0, side_label, count, tremor, hilb_freq, dev, if_score, if_anom, color):
        x1 = x0 + 300
        cv2.rectangle(frame, (x0, 150), (x1, 370), (30, 30, 30), -1)
        cv2.putText(frame, side_label, (x0 + 10, 180), 1, 1.2, color, 2)
        cv2.putText(frame, f"Taps: {count}",              (x0 + 10, 210), 1, 1, (255, 255, 255), 1)
        cv2.putText(frame, f"Tremor pwr: {tremor:.3f}",   (x0 + 10, 235), 1, 1, (200, 200, 200), 1)
        cv2.putText(frame, f"Hilbert: {hilb_freq:.2f} Hz",(x0 + 10, 260), 1, 1, (200, 200, 200), 1)
        base_txt = f"vs baseline: {dev:+.0f}%" if dev is not None else "baselining..."
        cv2.putText(frame, base_txt, (x0 + 10, 285), 1, 1, (0, 200, 255), 1)

        # Isolation Forest result
        if if_score is not None:
            if_color = (0, 0, 255) if if_anom else (0, 200, 100)
            if_txt   = f"IF: {'ANOMALY' if if_anom else 'normal'} ({if_score:+.3f})"
        else:
            if_color = (130, 130, 130)
            sessions_left = max(0, IF_MIN_SESSIONS - len(session_history))
            if_txt = f"IF: need {sessions_left} more sess"
        cv2.putText(frame, if_txt, (x0 + 10, 315), 1, 1, if_color, 1)

        bar = int(np.clip(tremor * 500, 0, 280))
        cv2.rectangle(frame, (x0 + 10, 340), (x0 + 10 + bar, 360), (0, 0, 255), -1)

    draw_hand_panel(20,      "LEFT HAND",  count_l, l_tremor, l_hilb_freq, l_dev,
                    l_if_score, l_if_anom, (0, 255, 0))
    draw_hand_panel(w - 320, "RIGHT HAND", count_r, r_tremor, r_hilb_freq, r_dev,
                    r_if_score, r_if_anom, (0, 100, 255))

    timestamps.append(elapsed)
    raw_l.append(cur_l)
    raw_r.append(cur_r)

    cv2.imshow("PD-PRO DIAGNOSTIC ENGINE v4", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or rem <= 0:
        break
    elif key in [ord('1'), ord('2'), ord('3')]:
        active_mode = int(chr(key))
        L_BUFFER.clear(); R_BUFFER.clear()
        count_l = count_r = 0
        tap_times_l.clear(); tap_times_r.clear()
        baseline["Left"] = baseline["Right"] = None
        mode_start_time = time.time()
    elif key == ord('a'):
        SENSITIVITY += 0.01
    elif key == ord('s'):
        SENSITIVITY = max(0.005, SENSITIVITY - 0.01)
    elif key == ord('e'):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame({"time": timestamps, "left": raw_l, "right": raw_r}) \
          .to_csv(f"pd_session_{ts}.csv", index=False)
        print(f"✅ Exported pd_session_{ts}.csv")

cap.release()
cv2.destroyAllWindows()


# -----------------------------------
# 5. POST-SESSION REPORT
# -----------------------------------
print("\n" + "=" * 60)
print("POST-SESSION CLINICAL REPORT")
print("=" * 60)
print(f"Mode: {mode_names[active_mode]}")


def full_session_report(t, data, tap_times_list, side):
    data = np.array(data)
    if len(data) < 30:
        print(f"\n{side} HAND: insufficient data")
        return None

    fs     = estimate_fps()
    smooth = gaussian_filter1d(data, sigma=2)
    peaks, _ = find_peaks(smooth, distance=5, prominence=np.std(smooth) * 0.5)

    report = {"tap_count": len(tap_times_list)}

    if len(tap_times_list) > 3:
        intervals   = np.diff(tap_times_list)
        speeds      = 1.0 / np.clip(intervals, 1e-3, None)
        half        = len(speeds) // 2
        first_speed = np.mean(speeds[:half]) if half > 0 else np.mean(speeds)
        last_speed  = np.mean(speeds[half:])
        speed_decay = (first_speed - last_speed) / first_speed if first_speed > 0 else 0
        report["speed_decay"] = speed_decay
        report["avg_speed"]   = float(np.mean(speeds))
    else:
        report["speed_decay"] = 0
        report["avg_speed"]   = 0

    if len(peaks) > 3:
        amps       = smooth[peaks]
        half       = len(amps) // 2
        first_amp  = np.mean(amps[:half]) if half > 0 else np.mean(amps)
        last_amp   = np.mean(amps[half:])
        report["amp_decay"] = (first_amp - last_amp) / first_amp if first_amp > 0 else 0
    else:
        report["amp_decay"] = 0

    chunk = FFT_BUFFER_LEN
    tremors, hilb_freqs = [], []
    for i in range(0, len(data) - chunk, chunk // 2):
        tp, _, hf, _ = analyze_window(data[i:i + chunk], fs)
        tremors.append(tp)
        hilb_freqs.append(hf)
    report["tremor_power"]  = float(np.mean(tremors)    if tremors    else 0)
    report["hilbert_freq"]  = float(np.mean(hilb_freqs) if hilb_freqs else 0)
    report["smooth"]        = smooth
    report["peaks"]         = peaks
    report["baseline"]      = baseline[side]

    base = baseline[side]
    dev  = deviation_pct(report["hilbert_freq"], base["freq"] if base else None)
    report["hilbert_freq_dev"] = dev if dev is not None else 0.0

    return report


rep_l = full_session_report(np.array(timestamps), raw_l, tap_times_l, "Left")
rep_r = full_session_report(np.array(timestamps), raw_r, tap_times_r, "Right")

# -----------------------------------
# 6. SAVE TO HISTORY + RETRAIN IF
# -----------------------------------
new_records = []
ts_str = datetime.now().isoformat()

for rep, side in [(rep_l, "Left"), (rep_r, "Right")]:
    if rep is None:
        continue
    record = {
        "timestamp":        ts_str,
        "mode":             mode_names[active_mode],
        "side":             side,
        "tap_count":        rep["tap_count"],
        "avg_speed":        rep["avg_speed"],
        "speed_decay":      float(rep["speed_decay"]),
        "amp_decay":        float(rep["amp_decay"]),
        "tremor_power":     rep["tremor_power"],
        "hilbert_freq":     rep["hilbert_freq"],
        "hilbert_freq_dev": rep["hilbert_freq_dev"],
    }
    session_history.append(record)
    new_records.append((rep, side, record))

save_history(session_history)
print(f"\n💾 History saved — {len(session_history)} total session(s) in {HISTORY_FILE}")

# Retrain model on the now-updated history
if_model = build_if_model(session_history)
if if_model is not None:
    print(f"🧠 Isolation Forest retrained on {len(session_history)} session(s).")
else:
    need = IF_MIN_SESSIONS - len(session_history)
    print(f"⏳ {need} more session(s) needed to activate Isolation Forest.")

# -----------------------------------
# 7. PRINT REPORT + IF VERDICT
# -----------------------------------
for rep, side, record in new_records:
    upper = side.upper()
    print(f"\n{upper} HAND:")
    print(f"  Taps: {rep['tap_count']} | Avg speed: {rep['avg_speed']:.2f} Hz")
    print(f"  Speed decay (bradykinesia):    {rep['speed_decay']*100:.1f}%")
    print(f"  Amplitude decay (hypokinesia): {rep['amp_decay']*100:.1f}%")
    print(f"  Tremor power:  {rep['tremor_power']:.4f} | Hilbert freq: {rep['hilbert_freq']:.2f} Hz")
    if rep["baseline"]:
        print(f"  Deviation from personal baseline: {rep['hilbert_freq_dev']:+.1f}%")

    # Rule-based flags (same as v3)
    flags = []
    if rep["speed_decay"] > 0.25:
        flags.append("SIGNIFICANT SPEED DECAY")
    if rep["amp_decay"] > 0.3:
        flags.append("SIGNIFICANT AMPLITUDE DECAY")
    if rep["tremor_power"] > 0.08:
        flags.append("ELEVATED TREMOR BAND POWER")
    print(f"  Rule flags: {', '.join(flags) if flags else 'none'}")

    # Isolation Forest verdict
    if if_model is not None:
        score, is_anom = score_session(if_model, record)
        verdict = "⚠️  ANOMALY" if is_anom else "✅ Normal"
        print(f"  Isolation Forest: {verdict}  (score {score:+.4f})")
        print(f"    (score < 0 → more anomalous vs your personal history)")
    else:
        need = IF_MIN_SESSIONS - (len(session_history) - len(new_records))
        print(f"  Isolation Forest: not yet active ({need} more session(s) needed)")

print("=" * 60)

# -----------------------------------
# 8. DASHBOARD
# -----------------------------------
if new_records:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"PD Motor Assessment v4: {mode_names[active_mode]}", fontsize=16, fontweight='bold')

    color_map = {"Left": "blue", "Right": "red"}

    for i, (rep, side, record) in enumerate(new_records[:2]):
        col = color_map.get(side, "purple")

        axes[0, i].plot(timestamps, rep["smooth"], color=col, linewidth=1.5, label="Signal")
        axes[0, i].scatter(
            np.array(timestamps)[rep["peaks"]], rep["smooth"][rep["peaks"]],
            color="gold", s=60, zorder=5, label="Peaks",
        )
        title_str = f"{side.upper()} — Tremor: {rep['tremor_power']:.4f}"
        if if_model is not None:
            sc, an = score_session(if_model, record)
            title_str += f" | IF: {'ANOM' if an else 'norm'} ({sc:+.3f})"
        axes[0, i].set_title(title_str, fontweight="bold")
        axes[0, i].legend(); axes[0, i].grid(alpha=0.3)

        tap_src  = tap_times_l if side == "Left" else tap_times_r
        speeds   = 1.0 / np.clip(np.diff(tap_src), 1e-3, None) if len(tap_src) > 1 else np.array([])
        if len(speeds) > 1:
            axes[1, i].plot(speeds, "o-", color=col, label="Tap speed")
            z = np.polyfit(np.arange(len(speeds)), speeds, 1)
            axes[1, i].plot(np.poly1d(z)(np.arange(len(speeds))), "k--", label="Trend")
        axes[1, i].set_title(f"{side.upper()} speed decay: {rep['speed_decay']*100:.1f}%",
                             fontweight="bold")
        axes[1, i].set_xlabel("Tap #"); axes[1, i].set_ylabel("Hz")
        axes[1, i].legend(); axes[1, i].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- History trend plot (shown when enough data exists) ---
    if len(session_history) >= IF_MIN_SESSIONS:
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        fig2.suptitle("Isolation Forest Anomaly Scores — All Sessions", fontsize=14)
        for side_filt, color in [("Left", "blue"), ("Right", "red")]:
            sub = [r for r in session_history if r.get("side") == side_filt]
            if len(sub) < 2:
                continue
            scores = [score_session(if_model, r)[0] for r in sub]
            ax2.plot(range(len(scores)), scores, "o-", color=color,
                     label=f"{side_filt} (score)", alpha=0.8)
            ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.set_xlabel("Session #")
        ax2.set_ylabel("IF Score (more negative = more anomalous)")
        ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
