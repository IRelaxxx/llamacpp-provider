import { defineConfig } from "tsdown";

export default defineConfig({
  entry: "src/index.ts",
  outDir: "dist",
  target: "node22",
  format: ["esm"],
  dts: true,
  clean: true,
  external: ["@ai-sdk/provider", "@ai-sdk/provider-utils", "zod", "zod/v4"],
});
