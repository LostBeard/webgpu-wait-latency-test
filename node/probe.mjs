// The page's seven cases run against Google Dawn directly, through npm `webgpu` (dawn-gpu/node-webgpu): the same WebGPU
// calls on the same GPU, with no browser process between the app and Dawn. Run: npm install && node probe.mjs
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);

const ROUNDS = 2, WARMUP = 5, SAMPLES = 40;
const gpu = create([]);
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
const device = await adapter.requestDevice();
const q = device.queue;
const N = 1000 * 4;
const src = device.createBuffer({ size: N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
const staging = device.createBuffer({ size: N, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const module = device.createShaderModule({ code: `@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < 1000u) { b[g.x] = b[g.x] + 1.0; } }` });
const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: src } }] });
const copy = () => { const e = device.createCommandEncoder(); e.copyBufferToBuffer(src, 0, staging, 0, N); q.submit([e.finish()]); };
const dispatch = () => { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pipeline); p.setBindGroup(0, bindGroup); p.dispatchWorkgroups(16); p.end(); q.submit([e.finish()]); };
const map = async () => { await staging.mapAsync(GPUMapMode.READ); staging.getMappedRange(); staging.unmap(); };
const cases = {
  A_emptyDone: async () => { await q.onSubmittedWorkDone(); },
  B_dispatchDone: async () => { dispatch(); await q.onSubmittedWorkDone(); },
  C_copyDone: async () => { copy(); await q.onSubmittedWorkDone(); },
  D_copyMap: async () => { copy(); await map(); },
  E_doneThenMap: async () => { copy(); await q.onSubmittedWorkDone(); await map(); },
  F_dispatchCopyMap: async () => { dispatch(); copy(); await map(); },
  G_emptySubmitDone: async () => { q.submit([device.createCommandEncoder().finish()]); await q.onSubmittedWorkDone(); },
};
const samples = Object.fromEntries(Object.keys(cases).map(k => [k, []]));
for (let round = 0; round < ROUNDS; round++) {
  for (const [name, fn] of Object.entries(cases)) {
    for (let w = 0; w < WARMUP; w++) await fn();
    for (let i = 0; i < SAMPLES; i++) {
      await q.onSubmittedWorkDone();
      const t0 = performance.now();
      await fn();
      samples[name].push(performance.now() - t0);
    }
  }
}
const info = adapter.info || {};
console.log(`Node ${process.version}, webgpu (Dawn), adapter: ${[info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' / ')}`);
console.log('| Case | Median (ms) | Min (ms) | p90 (ms) |\n|---|---|---|---|');
for (const [name, xs] of Object.entries(samples)) {
  const s = [...xs].sort((a, b) => a - b), at = p => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  console.log(`| ${name} | ${at(0.5).toFixed(3)} | ${s[0].toFixed(3)} | ${at(0.9).toFixed(3)} |`);
}
device.destroy();
process.exit(0);
