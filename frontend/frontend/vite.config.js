import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    proxy: {
      // Proxy all /api/* calls to the FastAPI backend during development — no CORS needed
      "/api": "http://localhost:5000",
    },
  },
});
