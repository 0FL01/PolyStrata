# Goal: Fix the RDNA2 prefill stalls in expert uploads

Status: complete (targeted mitigation deployed; exact production outlier not reproduced)
Source: Production incident of 2026-11-28; user requested profiling, a root-cause fix, testing, push, build and production deployment.

## Required outcomes

- R1 (verified for the replayed mechanism): Live stacks and targeted RDNA2 traces locate the stalls in HIP waits and correlate replay stalls with KFD queue eviction and restore retries. The exact trigger of the minutes-long production outlier remains untraced.
- R2 (verified for the targeted mitigation): The opt-in host-memory policy removes the observed USERPTR allocation path and reduces KFD evictions. The HIP build, host tests, and fixed-placement greedy ID and state comparisons pass. Other causes of KFD evictions are not fixed by this change.
- R3 (verified): Pushed `352beccd` to `origin/main`, installed the qualified build, and passed the production smoke checks below.

Stay on the Inkling RDNA2 engine and the implicated expert upload path. Leave the hot-cache policy, serving auth and model settings alone unless needed to fix the trigger. Use guarded runs on the RX 6700 XT with separate writable artifacts. Do not treat a watchdog timeout as a fix.

## Evidence

Baseline `c62b3324`, batch 2048, direct staging and early expert prefetch: on production the 60,566-token cold prefill took 571,709 ms. Live stacks stopped at `hipStreamSynchronize(up_)` before SWA capture and at the direct-staging fallback's pinned-buffer `hipEventSynchronize` inside expert prefetch. GPU telemetry was 0–1% and no new kernel fault was logged. The HIP event order has no source-level cycle in the normal path.

Scratch runs, reproduction driver and live stacks: `~/.cache/opencode-tmp/opencode/polystrata-runs/prefill-stuck.oqhuSJ/`. The host-handoff prototype was discarded: it does not address queue eviction during decode.

The production journal also shows stalled decode once the preceding reply reached its 9,634th token, including a 45-second pause. An instrumented stock long-decode replay saw a 4.55 s wait on the decode router event (no speculative prompt upload involved). The next stock replay captured KFD queue evictions with `USERPTR` and `SVM` triggers and restore retries during waits on both prefill and decode; one prefill wait spent 531 of 660 ms under eviction, and one checkpoint wait spent 244 of 246 ms under eviction. No kernel fault is needed for this mechanism. The minutes-long outlier was not reproduced with the synthetic prompts, so there is no driver trace of that exact request.

On RX 6700 XT, Linux 7.2.15 and ROCm 7.13.0a20260515, the short A/B (`pin-stock`, `pin-min`, `pin-paged`, `pin-both`, `pin-stock2`) used cold 9.2K, cold 12.3K, and a parked return with 129 fresh tokens. Both stock runs recorded 62 USERPTR allocations (281 MiB cumulative) and 8–10 queue evictions. Raising only the ROCclr transfer threshold left those allocations but reduced the observed evictions to one. Changing only the ROCr allocation path removed the USERPTR allocations but still allowed eight SVM evictions. With both settings there were zero USERPTR allocations and one SVM eviction; the longest observed eviction interval was 61 ms versus 313–355 ms on stock. These runs varied by up to three VRAM expert slots and are not parity evidence.

The candidate exposes the two settings together as `STRATA_INKLING_RECLAIM_SAFE=1`, before HIP initialization, off by default with explicit runtime overrides. It does not change the stream handoff or cache layout. HIP build and the three conversation CTests passed. In the frozen-placement A/B (`exact-stock`, `exact-off`, `exact-safe`, 1,151 resident experts, greedy, 24 tokens per turn), all 72 token IDs, reuse positions `[0, 0, 2698]` and all bytes of the returned conversation's STRSESS file matched (SHA-256 `22582e9481d651e84569113ab0be897dbff8791613daf3ef2e20fa92f2b457c7`). KFD tracing verified 58 USERPTR allocations in each of the first two arms and zero with the flag. A larger unfrozen run diverged on token 9 of the parked return; it allows timing-based selection and is not a parity test.

On the production-size synthetic replay (`kfd-stock` and `kfd-safe`, five requests, 200K context, swaps 64, 4 GiB hot cache), KFD reported **299 → 11** queue evictions: 41 → 0 USERPTR and 258 → 11 SVM. Restore retries went **461 → 13**; the longest logged HIP wait went **676 → 177 ms**. Cold prefill at 60,566 tokens was effectively unchanged (**71.05 → 70.98 s**); the earlier 45,955-token cold read was 54.62 → 50.02 s. One 768-token decode grew from 17.84 to 18.12 s. The live expert pool started at 1,372 vs. 1,371 and was free to move, so these timings and output differences (first comparable divergences at tokens 10, 1, and 7 on three turns) are not fixed-placement parity. Both guarded runs had no stop or warning. This establishes fewer pauses from the measured mechanism, not a reproduction of the minutes-long live request.

## Production verification

The installed `strata-poly-352beccd` matched the qualified executable (SHA-256 `246f94d30d55efe5e3ad801afae0cbbcb534f055ffb8895f12b6eba2c6f7b26e`). The only config changes were the executable and `STRATA_INKLING_RECLAIM_SAFE=1`. Startup printed both effective ROCm values. The service was active with no systemd restarts, listening on `127.0.0.1:8080`; an unauthenticated model request returned `401`. The retained-image prompt reused 2,847 of 2,864 rows and read the 17 remaining rows in 309 ms. An edited image reused only the earlier text prefix and the answer changed from red to blue. The production guard reported no stop or warning. Rollback to the previous executable and settings uses the saved private config `before-reclaim-20261128-131309.json.bak`.
