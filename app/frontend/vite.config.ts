import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Single-origin in production (FastAPI serves dist/). In dev, proxy /api to the
// backend so the frontend and API share an origin from the browser's view.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1200,
  },
});
