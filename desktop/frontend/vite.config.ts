import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  base: command === "build" ? "/static/app/" : "/",
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:8000",
      "/run": "http://localhost:8000",
      "/videos": "http://localhost:8000",
      "/reports": "http://localhost:8000",
      "/appeal": "http://localhost:8000",
      "/budget": "http://localhost:8000",
      "/settings": "http://localhost:8000"
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true
  }
}));
