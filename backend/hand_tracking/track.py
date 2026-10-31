import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Store data
timestamps = []
distances = []

start_time = time.time()

cap = cv2.VideoCapture(0)

print("Press 'q' to stop recording")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape

        # Thumb tip (4)
        thumb = hand_landmarks.landmark[4]

        # Index tip (8)
        index = hand_landmarks.landmark[8]

        x1 = int(thumb.x * w)
        y1 = int(thumb.y * h)

        x2 = int(index.x * w)
        y2 = int(index.y * h)

        # Palm landmarks for normalization
        index_mcp = hand_landmarks.landmark[5]
        pinky_mcp = hand_landmarks.landmark[17]

        # Thumb-index distance
        finger_distance = np.sqrt(
            (index.x - thumb.x)**2 +
            (index.y - thumb.y)**2
        )

        # Palm width
        palm_width = np.sqrt(
            (pinky_mcp.x - index_mcp.x)**2 +
            (pinky_mcp.y - index_mcp.y)**2
        )

        # Scale-independent signal
        if palm_width > 0:
            distance = finger_distance / palm_width
        else:
            continue

        current_time = time.time() - start_time

        timestamps.append(current_time)
        distances.append(distance)

        # Draw landmarks
        cv2.circle(frame, (x1, y1), 8, (0,255,0), -1)
        cv2.circle(frame, (x2, y2), 8, (0,255,0), -1)

        cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)

        cv2.putText(
            frame,
            f"Normalized Distance: {distance:.2f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow("Finger Tapping Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if len(timestamps) < 20:
    print("Not enough data collected.")
    exit()

# Plot Distance vs Time
plt.figure(figsize=(10,5))
plt.plot(timestamps, distances)
plt.xlabel("Time (s)")
plt.ylabel("Thumb-Index Distance (pixels)")
plt.title("Finger Tapping Signal")
plt.grid(True)
plt.show()
t = np.array(timestamps)
d = np.array(distances)
# Remove first 2 seconds
mask = t > 2

t = t[mask]
d = d[mask]
# Smooth signal
smooth = gaussian_filter1d(d, sigma=2)

# Detect peaks (fingers open)
fs = len(t) / (t[-1] - t[0])

min_peak_distance = int(0.15 * fs)

peaks, _ = find_peaks(
    smooth,
    prominence=np.std(smooth) * 0.3,
    distance=min_peak_distance
)

peak_times = t[peaks]
peak_values = smooth[peaks]

# -----------------------------------
# FEATURE 1: Average Amplitude
# -----------------------------------
avg_amplitude = np.mean(peak_values)

# -----------------------------------
# FEATURE 2: Amplitude Decay
# -----------------------------------
n = len(peak_values)
duration = t[-1] - t[0]

tap_frequency = len(peaks) / duration

if n >= 6:

    first_amp = np.mean(peak_values[:n//3])
    last_amp = np.mean(peak_values[-n//3:])

    amplitude_decay = (
        first_amp - last_amp
    ) / first_amp

else:
    amplitude_decay = 0

# -----------------------------------
# FEATURE 3: Speed Decay
# -----------------------------------
if len(peak_times) > 2:

    iti = np.diff(peak_times)

    speed = 1 / iti

    first_speed = np.mean(speed[:len(speed)//2])
    last_speed = np.mean(speed[len(speed)//2:])

    speed_decay = (
        first_speed - last_speed
    ) / first_speed

else:
    speed_decay = 0

# -----------------------------------
# FEATURE 4: Rhythm Variability
# -----------------------------------
# -----------------------------------
# FEATURE 4: Rhythm Variability
# -----------------------------------
if len(peak_times) > 2:

    iti = np.diff(peak_times)

    hesitation_count = np.sum(iti > 1.0)

    valid_iti = iti[
        (iti > 0.08) &
        (iti < 3.0)
    ]

    if len(valid_iti) > 1:
        rhythm_variability = np.std(valid_iti, ddof=1)
    else:
        rhythm_variability = 0

else:
    hesitation_count = 0
    rhythm_variability = 0


# -----------------------------------
# PRINT RESULTS
# -----------------------------------
print("\n========== FEATURES ==========")

print(f"Average Amplitude      : {avg_amplitude:.2f}")

print(f"Amplitude Decay        : {amplitude_decay:.3f}")

print(f"Speed Decay            : {speed_decay:.3f}")

print(f"Rhythm Variability     : {rhythm_variability:.3f}")

print(f"Detected Peaks         : {len(peaks)}")
movement_drop = amplitude_decay * 100
speed_drop = speed_decay * 100

print("\n========== CLINICAL SUMMARY ==========")

print(f"Tap Frequency      : {tap_frequency:.2f} taps/sec")
print(f"Movement Size Drop : {movement_drop:.1f}%")
print(f"Speed Reduction    : {speed_drop:.1f}%")
print(f"Hesitations        : {hesitation_count}")
# -----------------------------------
# VISUALIZATION
# -----------------------------------
plt.figure(figsize=(12,5))

plt.plot(
    t,
    smooth,
    label="Distance Signal"
)

plt.scatter(
    peak_times,
    peak_values,
    color="red",
    label="Detected Peaks"
)

plt.xlabel("Time (s)")
plt.ylabel("Normalized Thumb-Index Distance")

plt.title(
    f"Finger Tapping Analysis\n"
    f"Movement Drop: {movement_drop:.1f}% | "
    f"Speed Drop: {speed_drop:.1f}%"
)
plt.legend()

plt.grid(True)

plt.show()