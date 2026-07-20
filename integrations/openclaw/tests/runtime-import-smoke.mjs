const module = await import("../dist/index.mjs");

if (!module.default || typeof module.default.register !== "function") {
  throw new Error("Built OpenClaw plugin does not expose a register function");
}

console.log("OpenClaw plugin runtime import passed");
