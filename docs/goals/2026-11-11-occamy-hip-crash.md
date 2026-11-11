# Goal: reproduce and fix Occamy HIP/HSA crashes

Status: blocked
Source: user-approved diagnostic plan of 2026-11-11; subsequent instruction to copy the plan into a goal, reproduce, fix, retest, commit and push. Production `strata` downtime explicitly approved; nobody uses it during this work.
Last updated: 2026-11-11

## Objective

Reproduce the reported `strata-poly` HIP/HSA crash, implement the smallest evidence-backed correction, retest and deploy it, then commit and push this task's changes. Keep Occamy IQ4_XS, its matching image projector, 200000 context capacity, Q8_0 KV and CPU/RAM-resident vision. A non-reproducing run alone does not establish a fix.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work on the smallest unresolved outcome. Do not add requirements from reviews, tests, tools, speculative risks, or optional source text. Finish when every required outcome is resolved and affected constraints remain satisfied. Stop substantive work at a proven external blocker, an approved budget boundary, or when no remaining in-scope action has a falsifiable expected result; record the exact evidence and smallest unlock.

## Frozen Contract

### Required Outcomes

- R1: Preserve the approved plan in this repository-local goal.
  - Source: user, "делай копию плана в цель".
  - Acceptance: approved sequence, constraints and evidence are resumable without chat history.
  - Primary evidence: this document.
  - Status: verified.
  - Evidence: created 2026-11-11 before implementation.
- R2: Diagnose and reproduce the reported failure.
  - Source: user, "воспроизводи дефект"; approved core-first plan.
  - Acceptance: saved crash artifacts analysed with explicit binary provenance; a controlled baseline exhibits the same failure class, or a concrete blocker is recorded without claiming success.
  - Primary evidence: sanitized core-analysis and failing-reproducer results, with exact commands and artifact references.
  - Status: blocked.
  - Evidence: saved cores analysed; baseline and the authorized static-queue variant each completed 3 passes without the original signal-dispatch crash. No original-crash reproducer has been established within the approved exposure. This is not R2 verification; the configuration workaround below addresses a source-backed path, not a locally repeated root-cause failure.
- R3: Correct the demonstrated defect and retest the correction.
  - Source: user, "чини, тестируй по новой"; approved minimal-fix plan.
  - Acceptance: owner-local correction passes the reproducer, affected regression checks and required backend gates; independent Sol review covers the final substantive diff. A workaround must be labelled as such, not as a proven root-cause fix.
  - Primary evidence: baseline/corrected comparison, affected test/build results and final review.
  - Status: blocked.
  - Evidence: the explicitly permitted configuration workaround is verified and deployed below, but a root-cause correction passing an original-crash reproducer is not established. This item is not closed by relabelling a successful bounded trial as a root fix. No SDMA fix is claimed; coherent runtime replacement remains outside current authorization.
- R4: Return the production user service to the requested working configuration.
  - Source: approved rollout plan; latest user authorizes production downtime and manipulation.
  - Acceptance: service serves Occamy with 200000 capacity, Q8_0 KV and matching CPU-resident vision; authenticated text and image smoke checks pass; rollback remains available.
  - Primary evidence: actual child command/configuration, health/model results and smoke responses.
  - Status: verified.
  - Evidence: guarded rollout completed with production authenticated text/image checks, actual engine/vision children and unchanged model/context/KV/reserve; only local env.DEBUG_HIP_DYNAMIC_QUEUES=0 added. Final kernel window was empty and service active/running. Versioned original-config backup remains available; see deployment evidence below.
- R5: Commit and push the completed task changes.
  - Source: user, "потом коммит пуш"; subsequent parent instruction permits publication of checked workaround results with explicit unresolved status, not an unfinished root-cause fix.
  - Acceptance: task changes committed and the intended remote ref contains them, without unrelated worktree edits or accidental publication of unrelated local commits.
  - Primary evidence: commit hashes, push result and remote-ref verification.
  - Status: verified.
  - Evidence: task-only commits including `50148343b6f0492c58ea79c7b61286937899b9e9` and `757326a9ec52bcd9571fe8735c41cefe5b074f13` were published over SSH to `0FL01/PolyStrata`, branch `docs/occamy-hip-queue-workaround-20261111`, followed by this final status commit. `git ls-remote` verifies the remote branch at the final pushed HEAD. No unrelated local commit, foreign worktree edit, core, secret or private prompt was published.

### Constraints

- Model: `/home/stfu/.lmstudio/models/unsloth/Qwen-3.6-35B-A3B/occamy-1.0-IQ4_XS.gguf`; projector: matching `mmproj-occamy-1.0-F16.gguf` in the same directory. Final context `200000`, `--kv q8_0`, vision `gpu: false`, `elastic: false`; keep the 3072 MiB VRAM reserve.
- Production `strata` may be stopped/restarted/reconfigured for this work. This supersedes the old R&D goal's no-restart restriction for this task only. Do not stop unrelated services. No concurrent model instances competing with production on the GPU; tests are sequential.
- Preserve existing worktree changes: `.gitignore` additions; deleted `rdna2-rnd/pp-tg-2026-11-03.md`, `scripts/probe_budget.py`, `scripts/run_forced.py`, and `test-{flagoff,flagon,stage128,window4,window8}.json`; untracked `DAMN_PLAN.md`. No reset/restore/staging of those changes.
- Baseline checkout HEAD: `317912bb9ae19da40f1d0907057a626ef9565a78`, branch `main`, upstream `origin/main`. At preflight the local tracking ref is two commits behind HEAD: `b366e934` and `317912bb` are pre-existing local commits. Inspect the live remote and publication range before push; no force push, implicit publication of unrelated history, or broad pull/rebase into the dirty worktree.
- Follow `AGENTS.md`: build each backend a changed file affects; missing tools and unrun checks are unverified, not PASS. No changes to defaults justified by unmeasured performance claims.
- Secrets, private prompts and core dumps stay local, outside Git. Use closed temporary directories under `$TMPDIR` (currently `/home/stfu/.cache/opencode-tmp`; approved subdirectory `opencode`), with extraction/disk budgets. Test/build projects outside the worktree. Do not overwrite original backups.
- Keep authenticated loopback access. No arbitrary driver installation, mixed ROCm library sets or unapproved public API changes.

### Non-goals

- Lowering the requested context, replacing Q8_0 KV with FP16, removing vision or changing models as the final solution.
- General performance tuning, multiple simultaneous diagnostic switches, new recovery/retry frameworks, unrelated cleanup or broad project/runtime upgrades.
- Claiming that short smoke checks validate full 200k occupancy or long-term stability.

## Approved Plan

### 1. Offline core analysis first

Analyse the saved SIGSEGV core for PID `2853869` and one SIGABRT core for PID `2858523` (PID `2865470` is another available ABRT). Check mapped ELF/libraries, signal information, registers, disassembly and the HSA signal/object/vtable pointer chain. Process cores sequentially, within a disk budget: compressed size does not bound the uncompressed size. Do not attach to the live engine. Disable GDB auto-load and debuginfod for this local inspection. Do not publish core contents.

Relevant commands: `coredumpctl info 2853869`, `coredumpctl info 2858523`; bounded `coredumpctl --output="$D/engine.core" dump 2853869`; offline GDB `info proc mappings`, `p $_siginfo`, `info registers`, `thread apply all bt`, and targeted memory/instruction reads. Expected result is either a demonstrated defect or a narrower hypothesis, not a mandatory speculative patch.

### 2. Reproduce, then one controlled comparison if needed

If core analysis does not establish the cause, first compare the current deployed ELF/config with that same ELF/config plus only `--no-graphs`. Keep context, KV, reserve, CPU vision, environment switches and all other options unchanged. Temporary test configurations stay outside the model registry, use isolated logs and identical fresh copies of the initial expert profile, and an available authenticated loopback port. Stop production while exercising the test engine; downtime is now approved.

Replay one fixed ordered history: approximately 44–47k prompt tokens, tool-result continuations and cache reuse. Use an original local history only with appropriate access/authorization; otherwise use a sanitized synthetic fixture. Never execute tools generated by the model. Bound this initial experiment to at most three identical passes per arm. If baseline does not reproduce, do not call a non-crashing candidate a fix: record the limitation and next concrete evidence-dependent checkpoint, or stop at a proven blocker. No blind flag matrix. Stop GPU experiments on fresh amdgpu fault/reset, OOM, or sampled free VRAM below 1536 MiB.

`--no-graphs` is a diagnostic/workaround candidate, not a proven root cause. It does not directly bypass the checkpoint synchronization where the reported read fault surfaced. Measure any prompt/decode speed cost; do not promise a speedup. Configured capacity 200k is distinct from testing a filled 200k context.

### 3. Fix only the demonstrated owner

Expected owners: `src/poly/engine.cpp` (`Engine::read`, `run`, `record`, `follow_loop`, `cover`) and directly related stream/event/copy lifetimes; `src/poly/qwen35.cpp` if KV involvement is demonstrated. Add the closest regression for the actual defect, not a speculative audit suite.

The A/B uses the installed ELF, not a rebuild of current HEAD. Before a source correction, recover the corresponding deployed source/build provenance in an isolated working copy. Do not substitute checkout HEAD for the deployed baseline. If correspondence cannot be established, record the precise provenance blocker instead of attributing differences to a patch.

If evidence isolates a ROCm defect, propose the smallest coherent pinned runtime replacement separately; do not upgrade `.venv` or drivers blindly. If only `--no-graphs` avoids a reproduced crash, clearly distinguish a measured configuration workaround from a root-cause correction. Material scope or runtime changes require escalation; ordinary in-scope implementation steps do not.

### 4. Retest, review, deploy, commit and push

Repeat the failing baseline scenario against the correction with comparable exposure. Run only affected existing checks and the direct new regression. Closest tests include `glm_conversations_test` (fake-engine cache logic), `poly_kv_q8_test` (Q8 layout/store), and `poly_hip_value_split_test` (HIP attention/kernel replay); these do not by themselves prove Engine lifetime correctness. Build affected backends per repository rules; record unavailable toolchains honestly. Obtain independent Review-Sol-Global review of the final substantive diff and resolve confirmed findings.

Deploy only a validated solution with versioned backups and rollback. Check health/model identity, actual `--max-context 200000 --kv q8_0`, CPU/resident vision, authenticated text and image requests, and fresh errors. On an unsuccessful deployment restore the prior configuration/binary where safe; on driver faults stop and report rather than retry indefinitely.

Commit only this task's artifacts and inspect the complete proposed push range. Use a scoped branch if needed to isolate task changes, validating the exact published tree; do not publish the two pre-existing local commits implicitly. If authentication, remote permissions, history or a required check blocks publication, report the exact blocker, not a successful push. Record completion evidence in this goal and the final report.

## Change Envelope

- Target: reported Occamy HIP/HSA crash and the local service deployment of its demonstrated correction.
- Expected repository changes: this goal, owner-local engine code if justified, nearest regression/reproduction harness and sanitized evidence. Relevant build wiring/documentation only as directly needed.
- Allowed local deployment artifacts: active config `/home/stfu/.config/strata/strata-qwen36-qwen3.6-35b-a3b-ud-q3_k_s.json`, `server.env`, task backups and engine artifacts. Keep credentials out of committed evidence.
- No new services, public APIs, broad dependency migration, retry infrastructure, unrelated models or other services. Expand scope only for a documented concrete blocker to a required outcome.

### Parent-approved recovery checkpoint (after first safety stop)

- First perform read-only journal/driver/process checks. Any ongoing fault/reset/hang or need for reboot, driver reset, privilege escalation or system changes is a hard blocker; none of those actions is authorized.
- Verify `DEBUG_HIP_DYNAMIC_QUEUES=0` against the installed runtime's matching source and disassembly, including parser support, the idle-check bypass and queue/resource consequences. A generic small HIP stress test is not a faithful reproducer; use a small reproducer only if this precise stale-signal path can be reproduced credibly.
- Otherwise run **only the graph-on installed ELF plus this one environment setting**, with the existing fixed history, at most three passes. Do not repeat the first baseline/no-graphs series or change swaps, SDMA, cache, model, context, KV, vision or other environment flags. Use the corrected fail-closed monitor/cleanup. Any new GPU fault stops the checkpoint with no automatic production restart.
- If the path is excluded mechanistically and bounded inference plus authenticated text/image and actual-config checks pass without fresh faults, a **narrow configuration workaround** may be deployed and presented with measured tradeoffs and final Sol review. This is not permission to call an unreproduced failure fixed, mark R2 verified, change ROCm packages, or publish unfinished work as a root-cause fix.
- Runtime replacement remains a separate material scope decision. Commit/push remain subject to checked results, explicit unresolved status and task-only publication.

## Current State

- Approved plan independently reviewed by Sol with no blockers; this is plan review, not implementation approval or a test PASS.
- Offline inspection and the bounded first comparison are complete. Extracted core sizes were 1,150,976,000 and 865,021,952 bytes, below the per-core 24 GiB limit. Private artifacts are under `$TMPDIR/opencode/occamy-core.DqJ1r2wu` (0700 directory; credentials, core and log files kept private).
- Crash timeline: Nov11 16:24:40 SIGSEGV/SEGV_MAPERR in `hsa_signal_load_relaxed`; Python reported exit -11 at 16:24:46. Prior request 44376/reused43956/read420/output525 completed successfully and is not the failing request. New engine later served 46525 tokens. Nov8 has the same HSA chain via `run/eval`; Nov11 16:34 and 16:37 have `__cxa_pure_virtual` through the same HSA call.
- No OOM/GPU-reset/systemd-kill evidence in the inspected journal window; direct `dmesg` access denied. This is not proof of the complete historical absence of such events.
- SIGSEGV core: `/var/lib/systemd/coredump/core.strata-poly.1000.85e7b7ae107b4520a450f7fb3e7ff7b3.2853869.1794403480000000.zst`, about 494 MiB compressed, raw size unknown. GDB/zstd/coredumpctl available; last disk sample had 113 GiB free.
- Current ELF SHA-256 `523a0cb7f4c4cd964c7b9f577ef8d22228047da2734ff8cbfffafa457635c004` matches `engine-poly/BUILD.json`; version 0.1.45, gfx1031, deployed Nov3, `main_commit=b366e934d6603b5fdd288c4466f3e0807c680b39`, `build_base_commit=f472d85a30927301663401f7dc892e668578a90e`, source patch SHA-256 `0f384a760ab02cfe95127110c7068cdb4944d6315a159ca3c8b1c7528f96c024`. Build-base object not locally present at recon. Suitable Engine DWARF absent.
- Provenance recon recovered the exact setup source fingerprint `fe8a21baecab8ba4` from local Git tree `b366e934`; HEAD instead produces `65875c3632643a93`. After fetch, the build-base object is locally available. Its engine-source difference from `b366e934` is only the two IQ4 files. An isolated `git archive b366e934` in the private directory reproduces the fingerprint over 463 engine files and pinned llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The original build directory, manifest and patch with the recorded SHA are no longer present; this limits exact build reconstruction. `build-poly-hip/strata-poly` is a different Nov2 binary and is not the deployed baseline.
- Core NT_FILE mappings match the installed engine and ROCm library paths; each captured ELF first page matches its current file byte-for-byte. Most executable pages were omitted from the cores, so this is not a full historical SHA-256 verification. Current disassembly places return address `0x13bb793` after checkpoint stream synchronization, equivalent to checkout `engine.cpp:4433`. Known `h2d`/`zero_vram` Async-on-`st_` fix is already in this ELF.
- SEGV: `si_addr=0x55cc00000000`; the indirect call at HSA `+0xdc768` reads slot zero of this invalid vtable pointer from the heap signal object at `0x55c90cfe8320`. ABRT: `__cxa_pure_virtual` returns to HSA `+0xdc76a`, reached through `Engine::settle_swaps()` synchronizing the upload stream, not the same read checkpoint. Both show invalid signal-object dispatch, but the cores do not identify the writer/destructor or prove an Engine versus ROCm cause. Some HSA mappings are not captured; zero-filled omitted pages cannot establish signal contents.
- Additional ABRT evidence: the HIP caller's current handle `0x7f51f3fed000` matches a captured heap object's `O+8`; that object has the base abstract `Signal` vtable and both reference counters zero, consistent with destruction. The identity of the object actually dispatched at the earlier call is not recoverable from caller-saved registers, so this correspondence is evidence, not a proved complete temporal chain.
- ROCm wheels currently `7.13.0a20260515`; HSA `+0xdc768` is an indirect call compatible with vtable dispatch. HIP internal frames are stripped: nearest exported symbols are not valid function attribution. Do not declare UAF or graph fault proven from these facts.
- Targeted source/disassembly comparison subsequently identified both HIP stacks as `VirtualGPU::ReleaseHwQueue()` → `IsQueueIdle()`: offsets `+0x3e8/+0x3f0` are the queue write indices and `+0x3f8` is a copied raw completion-signal handle. The installed HIP version filename embeds official source revision `d342d97fde`. Upstream [22af9e98, July 6](https://github.com/ROCm/rocm-systems/commit/22af9e98a3dad737d0a7b153c712622139e5a2e6) explicitly fixes this stale-handle UAF path by using a tracker-owned signal. Independent Sol review confirmed this close mechanism match. This is a high-confidence runtime-ownership hypothesis for the historical cores, not a locally verified fix or proof that the new SDMA fault has the same cause.
- Existing recovery in `serve/server.py` restarts the engine on the next request. No new recovery framework is needed for this scope.
- Open issue search completed: origin redirects to `simplexai-labs/PolyStrata`. Upstream `Niko1221/Strata#5559` has similar HSA symptoms on a different engine/GPU; it is not a proven matching fix. No issue/PR publication is requested.
- Live remote preflight: `git fetch --no-tags origin main` found `origin/main=e4ea22a3c5ac6d20f763fb8d73b2b8032db80e9b`, ahead on a diverged history. Neither local pre-existing commit is its ancestor. A direct push of local main is not an acceptable publication path; use a task-only branch from the verified remote baseline if a completed correction can be published.

## Completed First Comparison

- Closes: R2–R4, blocked at the approved GPU safety boundary.
- Completed experiment: `$D` below denotes `$TMPDIR/opencode/occamy-core.DqJ1r2wu`. `.venv/bin/python "$D/replay.py" > "$D/replay.log" 2>&1` used the installed ELF, fixed synthetic tool history, seed 1701, temperature 0, 640 output tokens/request, fresh identical initial expert-profile copies, authenticated loopback port 40673, and isolated configs/logs. The executed runner is preserved as `replay.executed.py`; the current `replay.py` has an offline-tested safety-cleanup correction and has not been re-run on GPU.
- Actual server prompt counts were **44614 → 44969 → 45324**, with cache reuse **0 → 44609 → 44964** (the initial direct-template estimates were slightly higher). Fixture SHA-256: `44b2a6a039fd407f0a135e0225a95854934971b6d3928de8954af75a7a4a1e1e`; initial profile SHA-256: `b00c3e9d79c4fb8691050f1f72c4290caa77c44b578a162e468a20f19ed4590b`. Generated tools were never executed.
- Baseline: **3 passes × 3 requests completed**, each output 640 tokens, no reported native crash. Full-prefill rates were 644.3/656.4/667.4 tok/s; decode ranged 34.7–46.7 tok/s. This bounded non-reproduction does not validate stability or a fix.
- `--no-graphs`: pass 1 requests 1–2 completed (655.6 tok/s full prefill; 47.3/46.9 tok/s decode). Request 3 returned **HTTP 503**, zero tokens, server-reported engine exit **-13/SIGPIPE**, with `Memory access fault by GPU node-1 ... Page not present or supervisor privilege`. No core exists for PID 2905942. This is not established as the original SIGSEGV/SIGABRT class. No further passes ran; no performance conclusion is justified from the incomplete arm.
- At **17:23:44 +03**, the kernel journal identifies that same PID and **SDMA1 page faults** at `0x7f12f5400000` and subsequent pages. The 233 periodic VRAM samples had minimum free **2969 MiB**, above the 1536 MiB stop threshold; sampled headroom does not rule out other memory/lifetime defects. Private evidence: `kernel-after-replay.jsonl`, `results.json`, `metrics.json`, `vram.jsonl`, `no-graphs-1/{server,engine}.log`, `no-graphs-crash-info.txt`.
- Safety-cleanup defect: the executed runner's daemon watcher was not joined and had no final journal sample. It restored production at 17:23:45 before seeing the last fault; its empty `stop-reasons.json` is therefore not safety evidence. Sol independently found the join race. After reading the completed logs, the executor stopped `strata` at **17:28:41**. The corrected runner now joins the watcher, fails closed on monitor errors/nontermination, and checks the complete kernel window after teardown before deciding whether to restart. Offline regression: `.venv/bin/python "$D/test_replay_cleanup.py"` **3 tests PASS**; `py_compile` of replay/smoke/cleanup-test scripts PASS. No GPU rerun was attempted.
- Current service: `ActiveState=inactive`, `SubState=dead`, `MainPID=0`; no test/model processes remain. The deployed ELF SHA remains `523a0cb7...c004`; active config, `server.env`, wrapper and existing backups were not edited. `--no-graphs` was not deployed. Prepared `smoke.py` was **not run** because of the GPU safety stop.
- Blocker/unlock: the first A/B budget is exhausted for baseline and stopped by a fresh GPU fault for the other arm. Further GPU execution requires a recovery/replan decision; silently restarting or expanding to a flag matrix violates the approved stop rule. Separately, using a coherent pinned ROCm runtime containing `22af9e98` requires the explicit runtime-scope decision in plan step 3. No compatible package set has been selected or tested; replacing one HIP library is not proposed. `DEBUG_HIP_DYNAMIC_QUEUES=0` is a source-backed possible diagnostic/workaround that excludes the old idle-check path, but it has not been run and cannot be called a root-cause correction.

## Current Checkpoint

- Closes: R3/R4 as a possible measured workaround; R2 remains unverified unless the original failure is actually reproduced.
- Read-only preflight at **17:40:25 +03**: `journalctl -k --since '2026-11-11 17:28:41'` returned no entries; `strata` remained inactive/dead/PID 0, no model/test processes remained, and VRAM used was 2,025,144,320 of 12,868,124,672 bytes. No ongoing fault/reset/hang was observed and no system recovery action is needed for the authorized new-process trial. This is not a whole-driver health guarantee.
- Installed-ELF proof (HIP SHA `b59071cc...c88`): flag table binds `DEBUG_HIP_DYNAMIC_QUEUES` to uint storage at ELF `+0x1a23748`, default 1; environment parser and uint setter at `+0x55da0b…+0x55db2c` / `+0x55e23f…+0x55e252` accept literal `0`. Settings load `+0x573b8d…+0x573bb4` packs the low two bits into settings byte `+0x25`, mask `0x06`. `ReleaseHwQueue` at `+0x575586…+0x575598` tests that mask and returns on zero before the historical idle-check call `+0x5755e2`. This excludes the identified path, not all HSA signal operations.
- Source/resource consequences: matching `d342d97fde` `rocvirtual.cpp:2180–2204` retains queues instead of opportunistic release in mode zero; destructor still releases them. Existing pooling/reuse remains, with ordinary default `GPU_MAX_HW_QUEUES=4` per priority, not a universal hard cap. Mode zero uses reference-count selection rather than dynamic depth. Extra retained resources/stream-creation and scheduling costs must be measured rather than assumed absent.
- No credible small deterministic reproducer for this exact failure was found in the matching fix; upstream explicitly could not unit-test it, and the related queue-swap test requires eight GPUs. Selected scenario: the existing fixed model history, **only** `DEBUG_HIP_DYNAMIC_QUEUES=0`, graphs on, three passes, followed by required authenticated text/image checks on the isolated server. New artifacts are in `$D/queue-static`; it preserves the old fixture/profile and never automatically restarts production. Each child will have its environment and the exact installed HIP parsed uint value read non-invasively from `/proc/PID/mem` (no attach or writes).
- The selected single supported variant is complete below; no further model-soak or flag matrix is authorized by this checkpoint.
- Stop: any fresh fault, unverified monitor state, ongoing driver problem or need for forbidden system/runtime changes. No repeated baseline, no no-graphs recovery, no flag matrix.

### Static-queue checkpoint results

- Command: `.venv/bin/python "$D/queue-static/replay.py" > "$D/queue-static/replay.log" 2>&1` (runner SHA `89beb39c3eba…`). **3 passes × 3 requests completed**, all 640 output tokens; same actual prompt/cache counts and fixture/profile hashes as the baseline. No original SIGSEGV/SIGABRT or new GPU fault was observed. Each engine's actual env was `DEBUG_HIP_DYNAMIC_QUEUES=0`, actual args retained graphs, and its parsed installed HIP uint at `bias+0x1a23748` read **0**. No live attach or memory write was used.
- Required isolated smoke also passed under the monitor: 200000 context, Occamy model, images enabled, key required; unauthenticated `/v1/models` **401**; authenticated text **`STRATA_OK`** and red PNG classified **`Red`**. CPU/resident vision config unchanged. Artifacts: `queue-static/{isolated-smoke.json,results.json,flag-proof.txt,static-queues-*/actual-engine.txt}`.
- Tradeoff observations, same-day baseline versus this variant: full-prefill medians **656.4 → 660.4 tok/s** (individual variant rates 654.0/665.9/660.4); decode median **45.4 → 45.4 tok/s** (variant range 44.2–47.0). Aggregate server request time **348650 → 343237 ms**. These small noisy samples do not establish a speedup, slowdown bound or long-term stability. The 181 variant guard samples had minimum free **2991 MiB** versus prior baseline/no-graphs-series minimum 2969 MiB; queue-specific allocation cost/count was not isolated or measured.
- Cleanup completed with empty reasons **after** watcher join and final check; no automatic production restart. `journalctl -k --since '2026-11-11 17:40:25'` remained empty. The prior SDMA no-graphs failure is not reclassified as fixed.
- Offline checks before deployment: `py_compile` and three cleanup regressions PASS for the new runner. `queue-static/deploy.py` uses an exclusive versioned config backup, an atomic config replacement adding only the one env key, required production smoke/actual child checks, and rollback-to-original-config-with-service-stopped on any failure/fault. Its `check_safety`, `monitor`, and `finish_monitor` ASTs are identical to the tested/reviewed runner; `py_compile` PASS.
- Sol reviewed the actual trial and prepared guarded rollout (`deploy.py` SHA `950ba8a6951f…`), found no blocking issues and recommended one deployment as a narrow local workaround, explicitly not a root-cause fix or R2 completion.

### Guarded production deployment

- Command: `.venv/bin/python "$D/queue-static/deploy.py" > "$D/queue-static/deployment.log" 2>&1` **PASS**. Only semantic active-config delta is `env.DEBUG_HIP_DYNAMIC_QUEUES="0"`; direct JSON comparison against the exclusive pre-change backup confirmed every other field unchanged. No source, deployed ELF, ROCm package/library, unit, wrapper, `server.env`, projector or existing backup was changed.
- Backup: `/home/stfu/.config/strata/strata-qwen36-qwen3.6-35b-a3b-ud-q3_k_s.json.bak.static-queues-20261111T180235` (0600). Rollback is stop `strata`, copy this backup back to the original config path, then restart only if the current driver/error state permits. Rollback code/path was reviewed, not exercised live; original configuration may still have the historical crash risk.
- Production checks **PASS**: authenticated `/health` and `/v1/models`, unauthenticated `/v1/models` **401**, text **`STRATA_OK`**, image **`Red`**. Actual engine PID **2933357** has Occamy, `--max-context 200000 --kv q8_0 --vram-reserve-mib 3072`, original remaining args, graphs on, and environment flag **0**. Actual vision PID **2933351** remains present after image processing, has the matching projector/model and no GPU option; config remains `gpu:false`, `elastic:false`. Main PID **2933345**, `ActiveState=active`, `SubState=running`.
- Final guard reasons empty after join/final journal check. `kernel-after-deploy.jsonl` for the full recovery window since 17:40:25 contains **0 entries**. Current ELF SHA remains `523a0cb7...c004`, HIP SHA `b59071cc...c88`. Artifacts: `queue-static/{deployment.log,deployment-smoke.log,production-smoke.json,deployment-actual.json,deployment-stop-reasons.json,deployment-backup-path.txt}`. This is bounded recovery evidence, not a long-term or full-200k-occupancy qualification.
- Final Sol review (`ses_e350213e9ffeDjpHsOP3sSbr4q`) confirmed the bounded workaround, actual production recovery and the exact docs-only result tree, with no blocking findings. It explicitly kept R2/root-fix R3 unresolved and did not qualify long-term stability, full-200k occupancy or the old SDMA fault as fixed. No engine/backend source was changed, so backend builds/regression kernels are not applicable to this configuration/documentation-only change; no backend build PASS is claimed.

### Task-only commit and publication

- Isolated worktree: `$D/publish`; branch **`docs/occamy-hip-queue-workaround-20261111`**, based on live verified remote main **`e4ea22a3c5ac6d20f763fb8d73b2b8032db80e9b`**. Only `docs/goals/2026-11-11-occamy-hip-crash.md` differs. Neither pre-existing local commit is an ancestor; no foreign dirty change, core, secret, private prompt, config or test artifact is in the publication tree.
- Result commits: **`50148343b6f0492c58ea79c7b61286937899b9e9`** (`docs: record bounded Occamy HIP queue workaround`) and **`757326a9ec52bcd9571fe8735c41cefe5b074f13`** (`docs: record publication access blocker`). The final checkpoint metadata is retained on the same docs branch, not on main. `git diff --cached --check` passed before commit.
- The earlier HTTPS push failed with HTTP 403 and no usable noninteractive credential. The user then supplied the writable SSH destination **`git@github.com:0FL01/PolyStrata.git`** and requested another push. `origin` now has that SSH URL for fetch and push, with no HTTPS remote left in the main or publication worktree.
- `git push --set-upstream origin docs/occamy-hip-queue-workaround-20261111` succeeded and created the remote branch at **`757326a9ec52bcd9571fe8735c41cefe5b074f13`**. This status update is a later docs-only commit on the same branch; `git ls-remote` verifies the remote branch equals that final local HEAD after the push. No force push, main push, issue/PR or alternate-repository publication was attempted; no unrelated local commit or foreign worktree change was published.
- The original reproduce/root-fix finish line separately needs a credible original-workflow reproducer with a newly approved evidence-dependent checkpoint, or explicit user acceptance/waiver making this measured workaround the finish line. Neither can be inferred from successful smoke checks. No further GPU trials or runtime replacement are authorized by this completed checkpoint.

## Material Decisions

- 2026-11-11: user approved the plan and authorized the full reproduce/fix/retest cycle, production downtime, commit and push. Old performance R&D remains a separate objective.
- 2026-11-11: the fresh SDMA fault activates the approved GPU stop boundary. Do not conflate it with the historical stale-signal hypothesis, apply an unvalidated Engine patch, deploy `--no-graphs`, or publish unfinished work as a completed correction.
- 2026-11-11: parent explicitly authorized read-only recovery preflight and one cause-directed `DEBUG_HIP_DYNAMIC_QUEUES=0` checkpoint, maximum three graph-on passes, with unchanged model settings and renewed hard stop on any fresh fault. No ROCm replacement, reboot/reset, elevated system changes or other-service modifications are authorized.
- 2026-11-11: supported-flag proof, the bounded variant and isolated smoke passed without fresh faults; Sol recommended the guarded rollout. Production recovery then passed with only the one local env setting added. Publish these checked workaround results and limits as documentation, not as a root-cause fix; the original reproduce/fix finish line remains unresolved.
- 2026-11-11: user supplied the writable SSH origin `git@github.com:0FL01/PolyStrata.git` and requested another push. HTTPS origin URLs were removed from the main and publication worktrees; the task-only docs branch was pushed successfully over SSH.

## Checkpoint History

- 2026-11-11: R1 verified; frozen contract and reviewed plan recorded. R2 starts from existing read-only recon, not repeated architectural discovery.
- 2026-11-11: R2 offline checkpoint narrowed the failure to invalid C++ HSA signal dispatch in both cores, at two Engine synchronization sites. No owner-local fix is justified yet; proceed to the approved single-switch A/B rather than speculative code changes.
- 2026-11-11: bounded A/B completed 9 baseline requests without the original crash; no-graphs failed on its third request with a new GPU page fault. GPU work stopped; service left stopped. Diagnostic runner safety cleanup corrected and tested offline after a review finding. R2–R5 remain blocked, not complete.
- 2026-11-11: final Review-Sol-Global (`ses_e350213e9ffeDjpHsOP3sSbr4q`) read the corrected runner, regression tests, final goal and actual experiment/journal evidence. The safety-cleanup finding is resolved; no new blocking findings for the corrected offline code or blocked-checkpoint record. Sol explicitly did not qualify a runtime fix or authorize resuming GPU work.
- 2026-11-11: parent-authorized static-queue checkpoint passed 3×3 requests and isolated auth/text/image checks; installed parsed flag was 0 in each process. Sol reviewed and recommended the guarded local workaround rollout, which passed production checks and restored `strata`. R4 verified; R2/root-fix R3 remain blocked; R5 starts docs-only publication.
- 2026-11-11: final Sol review found no blockers for the bounded deployed workaround and exact task-only docs tree. Local result commit created; the first authenticated HTTPS push was rejected with HTTP 403.
- 2026-11-11: R5 verified after the user supplied SSH origin `0FL01/PolyStrata`; the docs branch was pushed and its remote HEAD matched. R1/R4/R5 verified; R2 and root-fix R3 remain blocked because the original failure was not reproduced. Full objective remains blocked despite the successful publication.

## Completion

R1, R4 and R5 verified. The parent-authorized configuration workaround is qualified, independently reviewed and deployed, not a root-cause correction; R2 and root-fix R3 remain blocked because the original failure was not reproduced. The task-only docs branch is published to `0FL01/PolyStrata` over SSH, with no unrelated local commit or foreign worktree change included. Closure does not pass the frozen finish line because reproduction and root-cause correction remain unresolved; status is **blocked**, not complete. Preserve the running service/workaround and rollback backup; the remaining unlock is a credible original-workflow reproducer and an approved evidence-dependent root-cause checkpoint, or explicit user acceptance of the measured workaround as the finish line.
