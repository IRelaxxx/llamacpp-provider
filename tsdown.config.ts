import { defineConfig } from "tsdown";

export default defineConfig({
  entry: "src/index.ts",
  outDir: "dist",
  target: "node20",
  format: ["esm"],
  dts: true,
  clean: true,
  external: ["@ai-sdk/provider", "@ai-sdk/provider-utils", "zod", "zod/v4"],
});
