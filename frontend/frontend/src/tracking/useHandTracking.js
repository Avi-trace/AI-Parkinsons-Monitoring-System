/**
 * React hook around the shared GPU hand tracker.
 *
 *   const tracking = useHandTracking({ numHands: 1, onFrame });
 *   <video ref={tracking.videoRef} ... />
 *   await tracking.start();       // user clicked "Enable camera"
 *   tracking.lock(true);          // freeze resolution while recording
 *   tracking.stop();              // e.g. when the user leaves the test
 *
 * `onFrame(result, frameTimeMs, video)` runs on every processed frame and is read
 * through a ref, so passing a new function each render never restarts tracking.
 * `stats` (fps, cameraFps, inferenceMs, width, height, delegate) updates ~1x/second,
 * so the component does not re-render on every frame.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { acquireCamera, getHandLandmarker, releaseCamera, startTracking } from "./handTracker";

export function useHandTracking({ numHands = 1, onFrame } = {}) {
  const videoRef = useRef(null);
  const onFrameRef = useRef(onFrame);
  const loopRef = useRef(null);
  const ownsCameraRef = useRef(false);
  const statsRef = useRef(null);
  const [status, setStatus] = useState("idle"); // idle | loading | ready | error
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => { onFrameRef.current = onFrame; }, [onFrame]);

  const stop = useCallback(() => {
    loopRef.current?.stop();
    loopRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if (ownsCameraRef.current) {
      releaseCamera();
      ownsCameraRef.current = false;
    }
    setStatus("idle");
  }, []);

  const start = useCallback(async () => {
    if (loopRef.current) return true;
    setStatus("loading");
    setError(null);
    try {
      const [{ landmarker, delegate }, cam] = await Promise.all([getHandLandmarker(numHands), acquireCamera()]);
      ownsCameraRef.current = true;
      const video = videoRef.current;
      if (!video) throw new Error("video element not mounted");
      video.srcObject = cam.stream;
      video.muted = true;
      video.playsInline = true;
      await video.play();
      loopRef.current = startTracking(video, landmarker, {
        onFrame: (result, ts, v) => onFrameRef.current?.(result, ts, v),
        onStats: (s) => {
          statsRef.current = { ...s, delegate };
          setStats(statsRef.current);
        },
      });
      setStats({ fps: 0, cameraFps: 0, inferenceMs: 0, delegate });
      setStatus("ready");
      return true;
    } catch (err) {
      console.error("[useHandTracking] start failed:", err);
      if (ownsCameraRef.current) {
        releaseCamera();
        ownsCameraRef.current = false;
      }
      const denied = err?.name === "NotAllowedError" || err?.name === "SecurityError";
      setError(denied
        ? "Camera permission was denied. Allow camera access in the browser and try again."
        : err?.name === "NotFoundError"
          ? "No camera was found. Connect a webcam and try again."
          : "Could not start the camera or hand tracker. See the browser console for details.");
      setStatus("error");
      return false;
    }
  }, [numHands]);

  const lock = useCallback((value) => loopRef.current?.lock(value), []);

  useEffect(() => stop, [stop]); // release camera on unmount

  return { videoRef, status, error, stats, statsRef, start, stop, lock };
}

/** Pixel coordinates of landmark `i`, using the real video resolution. */
export function px(lm, i, video) {
  return { x: lm[i].x * video.videoWidth, y: lm[i].y * video.videoHeight };
}
