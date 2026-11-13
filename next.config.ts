import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server is bound on every interface, so a browser that opens
  // 127.0.0.1 is treated as cross-origin and the page never hydrates.
  allowedDevOrigins: ["127.0.0.1"],
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
