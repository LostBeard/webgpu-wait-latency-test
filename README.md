# webgpu-wait-latency-test

**Test your browser: https://lostbeard.github.io/webgpu-wait-latency-test/**

In Chrome (and Edge), waiting for a WebGPU submit that contains any work to complete costs about **3 ms** (median), however
little the work is. A 4 KB copy or a 1,000-thread dispatch that the GPU finishes in microseconds is reported done about 3 ms
later, whether the page waits with `queue.onSubmittedWorkDone()` or reads the result with `mapAsync()`. The same WebGPU calls
through Google Dawn directly (Node, npm `webgpu`, same GPU, same machine) complete in about **0.1 ms**. So the delay is not the
GPU and not Dawn: it is in how the browser learns that submitted work finished.

It matters because every result a WebGPU app reads back pays it once: an inference that ends with a readback, a physics step
that reads a value back, any loop that waits on the GPU. SpawnDev.ILGPU.ML measured it on MobileNetV2 in Chrome: a captured
inference replay plus its readback waited twice and took 8.0 ms; removing one of the two waits made it 5.1 ms with identical
outputs.

This repository holds the test page (plain WebGPU, no libraries: `index.html`) and the same test for Node (`node/probe.mjs`),
so anyone can measure their own browser and GPU and compare.

## Results so far

Median of 80 samples per case (40 x 2 rounds, each starting from an idle queue), milliseconds.

Windows 11, NVIDIA GeForce RTX 4070 (driver 32.0.16.2114), D3D12, 12 CPU threads, 2026-11-26:

| Case | Chrome 159 | Edge 148 | Node 24 + Dawn (npm `webgpu` 0.6.2) |
|---|---|---|---|
| `A_emptyDone` | 0.1 | 0.1 | 0.006 |
| `B_dispatchDone` | **2.9** | **2.9** | 0.114 |
| `C_copyDone` | **2.9** | **2.9** | 0.088 |
| `D_copyMap` | **3.0** | **2.9** | 0.099 |
| `E_doneThenMap` | **2.9** | **2.9** | 0.106 |
| `F_dispatchCopyMap` | **2.9** | **2.9** | 0.155 |
| `G_emptySubmitDone` | 0.1 | 0.1 | 0.015 |

The browser timings are in steps of 0.1 ms (`performance.now()` on a page that is not cross-origin isolated). The spread is
wide: in Chrome, C's minimum was 0.2 ms and its 90th percentile 3.8 ms; most samples sit near 3 ms. Firefox (its own WebGPU
implementation) is not measured yet: results welcome.

## Known upstream

This is a long-known Chromium issue, still open: [Chromium 40286576](https://issues.chromium.org/issues/40286576),
"Accelerate and decrease latency of GPU->CPU readbacks" (P1). Its 2022 predecessor,
[Chromium 42240322](https://issues.chromium.org/issues/42240322) "mapAsync API is very slow" (about 3 ms per `await mapAsync`,
closed as a duplicate of 40286576), traced the cause: Chrome's GPU process checks whether submitted work finished on delayed
tasks (nominally 1 to 2 ms apart, 4 to 7 ms measured) instead of being woken when the GPU finishes. This repository adds a
current measurement (Chrome 159, 2026), the comparison with Dawn outside the browser, and a page anyone can run.

## What each case does

| Case | Submitted and awaited |
|---|---|
| `A_emptyDone` | `onSubmittedWorkDone()` with nothing submitted: the round trip alone |
| `B_dispatchDone` | a 1,000-thread compute dispatch, then `onSubmittedWorkDone()` |
| `C_copyDone` | a 4 KB `copyBufferToBuffer`, then `onSubmittedWorkDone()` |
| `D_copyMap` | a 4 KB copy, then `mapAsync(READ)` of the destination: what an app's readback does |
| `E_doneThenMap` | a 4 KB copy, `onSubmittedWorkDone()`, then `mapAsync(READ)` |
| `F_dispatchCopyMap` | a dispatch and a 4 KB copy, then `mapAsync(READ)` |
| `G_emptySubmitDone` | an empty command buffer submitted, then `onSubmittedWorkDone()` |

What the cases separate:

- **A and G vs the rest**: waiting with nothing submitted, or after submitting an EMPTY command buffer, is fast (0.1 ms). As
  soon as the submit holds a command (one copy, one dispatch) the wait is about 3 ms.
- **B, C, D, F**: a dispatch, a copy, a mapped readback, or both together all cost the same. The work itself is negligible.
- **D vs E**: `mapAsync` adds nothing once the copy is known to be done, so the cost is not in mapping: it is in learning that
  the submit completed.
- **Chrome vs Node**: the same calls through Dawn without a browser process take about 0.1 ms.
- **The spread**: the same tiny copy sometimes completes in 0.2 ms, usually near 3 ms, up to about 4 ms. That looks like
  completion being noticed on a periodic check rather than when the GPU signals it, but this page only shows the timing, not
  the mechanism.

`performance.now()` is coarsened to 0.1 ms on a page that is not cross-origin isolated (the page reports the step it sees),
which is far below the effect.

## Run it

- **Browser:** open https://lostbeard.github.io/webgpu-wait-latency-test/ and press *Run the test* (a few seconds). *Copy the
  report* gives a Markdown summary with your browser, adapter and the numbers; *Open an issue with your result* fills it into a
  new issue. Results from any browser and GPU help, fast or slow.
- **Node (Dawn directly):** `cd node && npm install && node probe.mjs` (Node 22 or later).

## Notes

- Measured with installed Chrome, not a headless or automation build with a software adapter: the page prints the adapter, so
  a SwiftShader result is easy to spot.
- Nothing here depends on the GPU being busy: every sample starts after `onSubmittedWorkDone()` with an idle queue.
- If you know the cause, or a Chrome flag that changes it, please open an issue.

## License

MIT, see [LICENSE](LICENSE).
