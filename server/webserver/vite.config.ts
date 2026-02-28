import { defineConfig } from 'vite';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default defineConfig({
  root: '.',
  envDir: resolve(__dirname, '../..'),
  optimizeDeps: {
    include: [
      'three',
      'socket.io-client',
      '@mediapipe/tasks-vision',
    ],
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        landing: resolve(__dirname, 'landing.html'),
        plan: resolve(__dirname, 'plan.html'),
        viewer: resolve(__dirname, 'viewer.html'),
        sender: resolve(__dirname, 'sender.html'),
        detectionDebug: resolve(__dirname, 'detection-debug.html'),
        summary: resolve(__dirname, 'summary.html'),
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    https: {
      key: fs.readFileSync(resolve(__dirname, 'server.key')),
      cert: fs.readFileSync(resolve(__dirname, 'server.cert')),
    },
    warmup: {
      clientFiles: [
        './src/landing.ts',
        './src/main.ts',
        './src/plan.ts',
        './src/services/**/*.ts',
        './src/components/**/*.ts',
      ],
    },
  },
});
