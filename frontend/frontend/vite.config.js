import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  optimizeDeps: {
    // MediaPipe packages are UMD bundles — Vite can't pre-bundle them as ES modules.
    // We load them from CDN instead (see TremorTest.jsx), so exclude them here
    // to prevent the "does not provide an export named 'default'" crash.
    
    exclude: ["@mediapipe/hands", "@mediapipe/camera_utils"],
  },

  server: {
    proxy: {
      // Proxy all /api/* calls to Flask during development — no CORS needed
      "/api": "http://localhost:5000",
    },
  },
});