import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    rolldownOptions: {
      output: {
        // 将稳定的框架依赖拆开，降低单文件体积并改善浏览器缓存命中率。
        codeSplitting: {
          groups: [
            {
              name: "react-vendor",
              test: /node_modules[\\/](react|react-dom)[\\/]/,
              priority: 30,
            },
            {
              name: "antd-icons",
              test: /node_modules[\\/]@ant-design[\\/]icons/,
              priority: 20,
            },
            {
              name: "antd-vendor",
              test: /node_modules[\\/](antd|@rc-component|rc-)/,
              priority: 10,
            },
          ],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    css: true,
  },
});
