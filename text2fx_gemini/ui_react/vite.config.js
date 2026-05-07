import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': resolve(__dirname, 'src') },
  },
  build: {
    outDir: resolve(__dirname, '../ui'),
    emptyOutDir: false,
    sourcemap: true,
    rollupOptions: {
      input: resolve(__dirname, 'src/App.jsx'),
      output: {
        entryFileNames: 'v1.bundle.js',
        chunkFileNames: 'v1.[name].js',
        assetFileNames: 'v1.[name][extname]',
      },
    },
  },
});
