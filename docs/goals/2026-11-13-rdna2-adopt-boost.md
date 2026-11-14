# Goal: RDNA2 boost via adopting fork engine work (0.1.45-poly.9)

Status: active
Source: user request of 2026-11-13 ("изучи текущий расклад и надо подумать как ещё буст можно получить"), plan approved same day; RECON of `origin/main` `e4ea22a3` (two Explore reports) and prior cycles' evidence.
Last updated: 2026-11-14

## Objective

Raise pp and tg of the production `strata-poly` configuration (user-selected Qwen3.6 UD mixed IQ4_NL, gfx1031, 200000
context, `--kv q8_0`, reserve 3072 MiB, graphs on) by porting our local IQ4 patch onto the fork's
`0.1.45-poly.9` engine work, measuring that fork's opt-in HIP paths on this card, and enabling the
winning flag set. No new self-written kernels in this cycle: adopt, measure, validate, configure.
Commit and push the result. Deployment to the user's service is a separate later approval.

Revised finish line (user decision 2026-11-14): publish the confirmed B/C measurements,
opt-in port, tests and harnesses. Freeze the interrupted cycle; remaining confirmation and
GPU qualification are **waived_by_user**, not passed. This closes a measured research cycle,
not production-readiness or the original full optimization/qualification finish line.

## Execution Directive

Complete the frozen Required Outcomes using the listed Change Envelope and Primary Evidence. Work
on the smallest unresolved outcome. Do not add requirements from reviews, tests, tools, speculative
risks, or optional source text. Finish when every required outcome is resolved and affected
constraints remain satisfied.

## Frozen Contract

### Required Outcomes

- R1 — Port and build. A branch based on `origin/main` `e4ea22a3` carries the IQ4 row-geometry
  opt-in (from local `b366e934`; `STRATA_RDNA2_IQ4_ROWS`) and builds for HIP gfx1031 with tests.
  The phase-timer commit `317912bb` is optional to port; if skipped, record it.
  Evidence: branch name, cherry-pick/merge result, build log, binary SHA-256, passing test list
  (`iq_multi_parity`, `poly_hip_split_attend_test`, `poly_hip_prefill_fp32_test`,
  `poly_mmq_tiles_test`, `expert_parity`, `poly_workers_test`, `poly_hip_value_split_test`,
  `poly_kv_q8_test` as available).
  Status: verified. Port `c6b2efcc` and all nine build targets succeeded; seven CTests,
  IQ4_ROWS=1 parity, new-model/tokenizer load and actual-model expert parity passed.
- R2 — Screening and paired measurement of the fork's opt-ins on the production contract
  (200000/q8_0/threads 8/graphs on/`DEBUG_HIP_DYNAMIC_QUEUES=0`):
  `STRATA_HIP_SPLIT_ATTEND=1` (mode `2` only as a clearly separated extra arm),
  `STRATA_HIP_PREFILL_FP32=1` and `=2`, `STRATA_POLY_STAGED_TILES=8,16,32`.
  One screening pass per arm on the realistic 44–47k replay workload, then 5 AB/BA pairs plus one
  A/A control for each arm that screens positive; strict teacher-forced pairs (adapted forced
  runner, `--follow`) where trajectories must be fixed. Frozen statistics rule (below) decides
   confirmed / not confirmed / insufficient data. A negative result is acceptable and final.
  Status: waived_by_user for the unfinished scope; partially verified overall. B verified:
  tg+5.26%, CI[+2.67%,+7.92%]; C verified: pp+28.54%, CI[+27.73%,+29.36%], each five pairs+A/A.
  D's remaining pairs/control and strict forced pairs waived after two D pairs. E/F remain
  screening-only, not confirmed; F screen860.1pp /44.1–44.4tg. IQ4_ROWS excluded.
- R3 — Speculation series (`--window 4` and `--window 8` with `--lookup 3`, `--no-draft` removed
  or window semantics per current CLI) as a settings series on the same workload: tg change,
  guess-acceptance share from engine logs, and a needle/retrieval check. Never mixed into the
  kernel-flag statistics; percentages not added.
  User clarification: baseline, lookup-only (--no-draft) and MTP+window (draft enabled) are
  distinct families. Screen windows4/8 once in each speculation family; five pairs plus A/A
  for each family's best. If MTP is clearly stronger, screen final F+MTP and require its needle.
  Status: waived_by_user for confirmation/needle/joint-screen scope. Both window screens
  verified as observations only: lookup-only rejected for recommendation (-4.18%/-2.65% tg),
  MTP8+24.81% with1082/1232 accepted, **not confirmed**; no needle or R3 pairs ran.
- R4 — Correctness gates for every flag recommended on: needle/retrieval over long context
  (>= 44k input), flag-off runs byte-identical to the fork default path, existing parity tests
  green, and for `STRATA_POLY_STAGED_TILES` explicit staged-tiles parity for the selected model's
  expert types IQ4_NL/IQ3_S/Q4_K/Q3_K (replaces the original IQ4_XS gate by user decision).
  Status: waived_by_user for unfinished qualification; partially verified overall. Code/kernel
  flag-off checks, IQ4 graph parity, actual-model parity and57-case actual-type staged parity
  passed before the fault. Model-level forced equivalence, needle and final GPU-test repeat were
  not run and are waived for this cycle, not validated. No additional GPU check is authorized.
- R5 — Report and recommended configuration: `rdna2-rnd/boost-2026-11-13.md` (raw pair IDs,
  geomeans with intervals, acceptance shares, what is unverified, rollback = unset flags and old
   binary) plus a ready-to-apply production config delta (not applied).
  Status: verified under the revised finish line. Final report records B/C intervals/A-A,
  unconfirmed screens, fault, waivers, future candidate flags and pre-deployment gates. A
  qualified ready-to-deploy delta is waived; no configuration is applied.
- R6 — Commit and push: task-only commits on the branch; push to `origin`
  (`git@github.com:0FL01/PolyStrata.git`) — fast-forward `main` if the branch is a clean
   descendant of `e4ea22a3`, otherwise a topic branch; no unrelated local commits published.
  Status: in_progress. User explicitly authorized finalization/commit/push of the frozen result;
  no new GPU runs, recovery, deployment or default changes.

### Constraints

- C1: The user service stays untouched: `strata.service` remains inactive; no restart, no config
  write, no binary replacement under `engine-poly/`. Deployment is out of scope for this cycle.
- C2: Every engine process runs with `DEBUG_HIP_DYNAMIC_QUEUES=0` and graphs enabled;
  `--no-graphs` is forbidden (fresh SDMA fault on 2026-11-11). Stop the series on any new
  amdgpu/KWin fault or sampled free VRAM below 1536 MiB; report minimum sampled headroom.
- C3: One engine at a time; the desktop keeps its light background load (accepted, recorded);
  runs are not pooled with older series (different binary base and contract).
- C4: Preserve the pre-existing dirty state of the main worktree (`.gitignore` modification,
  deleted `rdna2-rnd/*` tracked files, untracked `DAMN_PLAN.md`, goal docs). Build and benchmark
  in a separate linked worktree/branch; nothing of the old dirty state is staged, reset or pushed.
- C5: No default behaviour changes: all measured paths stay opt-in env/CLI; flag-off must equal
  the fork default path. `--vram-pool` is not deployed (Linux untested, desktop VRAM risk); it may
  appear only as an explicitly labelled non-deployable screening arm if time permits.
- C6: Private fixtures, prompts, logs and cores stay outside Git; benchmark artifacts under
  `$TMPDIR/opencode` or ignored `rdna2-rnd/` paths; secrets never committed.
- C7: `AGENTS.md` build rules: HIP gfx1031 build required; CUDA/SYCL recorded as unverified if
  toolchains are absent. Pinned llama.cpp `3cf03257f` is unchanged by the fork; a pin bump is out
  of scope.

### Non-goals

- Writing new GPU kernels, porting WMMA/MFMA paths, hybrid GPU routing, Q8 KV default changes,
  model/quant changes, `--vram-pool` deployment, service deployment, further crash-root-cause work
  (owned by `2026-11-11-occamy-hip-crash.md`).

## Change Envelope

- Target: engine binary from `e4ea22a3` + ported IQ4 opt-in; measured env/CLI opt-ins; report and
  goal docs.
- Expected paths: `src/kernels/cuda/native_mmvq.cu`, `src/kernels/iq_multi_parity.cpp` (port),
  possibly `rdna2-rnd/scripts/*.py` (restored/adapted runners), `rdna2-rnd/*.md`, `docs/goals/*`.
  Engine source changes beyond the port only if a fork defect blocks a required outcome, with
  evidence and minimal diff.
- Forbidden: serve/protocol rewrites, dependency pin changes, public API changes, service files.
- Budget: iterative; GPU series bounded (screening ≤ 1 pass per arm; confirmation ≤ 5 pairs per
  winner + 1 A/A; stop earlier on gates/faults).

## Frozen statistics rule

Geometric mean of paired ratios with an approximate two-sided 95% t interval on the logs.
Confirmed gain: interval entirely above 1.0. Target: lower bound above 1.10. Not confirmed:
interval includes 1.0. Insufficient data: interval wider than ±5% around the estimate. No
confirmed regression of the other metric: its upper bound ≤ 1.02 (pp for tg arms, tg for pp arms;
the speculation series reports tg and acceptance only). Extensions capped at 10 pairs, announced
before reading results.

## Current Checkpoint

Closes: R6 and the user-revised frozen research cycle.
Smallest next action: offline checks, task-only commits, final Sol review, then non-force push.
Check live origin/main before push; use main only if still e4ea22a3, otherwise the topic branch.
Expected evidence: reviewed commit range, successful push and matching remote SHA.
Recovery is cancelled for this cycle. No new GPU run, reset, ROCm change, deployment or service start.

## Current State

- Original RECON: plan approved; two reports found dispatch conditions on our shape pass for
  SPLIT_ATTEND (single-sequence decode only), PREFILL_FP32 (attend_batch path), STAGED_TILES
  (IQ4_XS allowed, runtime `Mmq::fits` unproven); all published numbers are gfx1010/Windows).
- Historical Occamy only: 2026-11-11 old-binary pp644–667, tg34.7–46.7; not a comparison
  reference for the selected UD-IQ4_NL file. Fresh new-model baselines and B/C pairs are below.
- Available assets: private replay fixture + guarded runner (fail-closed monitor, VRAM stop,
  journal check) under `$TMPDIR/opencode/occamy-core.DqJ1r2wu`; committed R&D runners in
  `317912bb:rdna2-rnd/scripts/` (restore via `git show`, they hard-check 65536/fp16 and need
  adaptation to 200000/q8_0).
- Key facts: fork ggml pin unchanged; `--vram-pool` needs VMM and is Linux-untested;
  `STRATA_HIP_SPLIT_ATTEND` early-returns before `STRATA_HIP_VALUE_SPLIT` (not combinable on one
  call); mode dispatch accepts any non-zero int (only `2` = int8) — use exact `1`/`2` values;
  attn_rows/block-pass never take the split path; scratch for both attention paths fits the
  existing `score_`/`awork_` at 200000 (arithmetic in RECON).
- Worktree: `/home/stfu/ai/agent-frameworks/PolyStrata-boost`, branch
  `perf/rdna2-adopt-poly9`, based on `e4ea22a3c5ac6d20f763fb8d73b2b8032db80e9b`.
  Clean cherry-pick `c6b2efcc` contains only the two IQ4 files; NVFP4 changes remain intact.
- Build/artifact root: `$TMPDIR/opencode/boost-20261113` (0700, outside Git).
  Ordinary FetchContent succeeded, resolved pinned llama.cpp
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`; Release HIP gfx1031 configured successfully.
- Guard harness: adapted watcher joins before its final safety decision, samples VRAM from
  before launch through teardown, checks kernel and user/KWin journals, never touches services.
  Three offline cleanup regressions and all runner syntax checks passed.
- Build: all nine requested targets succeeded (`build.log`); engine SHA-256
  `d0d08867bf0a684949a85ed5e480a40a6832f2430645fd1cd86ba621356acc90`.
  CUDA `nvcc` and SYCL `icpx` are absent, hence those builds are unverified.
- Tests: seven named CTests passed in 34.32 s; a separate IQ4_ROWS=1 run reported zero failures
  and eight bitwise-equal IQ4 row-geometry graph checks. Guard reasons empty after join/final
  journal check; minimum sampled free VRAM9733.34MiB at that first unit-test checkpoint,
  before new-model inference. Cycle minimum was later2878.29MiB.
- A fork-default comparator was linked outside Git from the exact `e4ea22a3` native source and
  otherwise identical engine objects (SHA-256 `b6a194f3a11beb157bf074267081cac593f0f006603ac91a874ae12521069927`).
  It is prepared, not run; no actual-model flag-off equivalence is claimed.
- Superseded blocker: `poly_expert_parity` reported `cannot open` the original
  `/home/stfu/.lmstudio/models/unsloth/Qwen-3.6-35B-A3B/occamy-1.0-IQ4_XS.gguf`.
  The directory instead contains only `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` and `mmproj-F16.gguf`;
  bounded search under `~/.lmstudio/models` found no Occamy file. A different quant was not an
  authorized substitute at that time; the user then explicitly authorized the new model.
- New model SHA-256: `d0ad78047bd0a583bfcb484e9defc63112c5f1f856e310dcf88dc48f2ebeae5b`.
  Existing tokenizer pack loaded; 14-token prompt generated 32 decodable tokens; actual-model
  parity PASS. Expert pairs in its log: IQ3_S/IQ4_NL, IQ3_S/Q6_K, IQ4_NL/Q6_K, Q3_K/Q4_K.
  All actual expert types had a working MMQ tile and batch output versus FP64 reference.
- Runtime clarification: the ordinary `--no-draft` load reported 10240 routed experts (40 layers).
  Engine derives main-layer count from block_count minus nextn_predict_layers; do not infer 41
  executed layers from tensor names or claim +2.5% inference work. No architecture change made.
- Screen A: cold pp 625.1 tok/s; per-request tg 38.7/40.8/39.1 tok/s. B split: pp642.9,
  tg43.8/44.4/42.9 (tg geomean ratio1.10558). C prefill1: pp827.7 (ratio1.32411),
  tg41.4/41.0/41.3. D prefill2: pp777.4 (ratio1.24364), tg41.3/41.1/39.0.
  E tiles: pp660.7 (ratio1.05695), tg41.5/41.5/41.5. All single-pass positive screens, not
  confirmed effects or confidence intervals. Native prompt counts44613/44968/45323; 640 output
  each; cache reuse0/44608/44963. Minimum sampled free3021.04MiB; guard reasons empty.
- Actual-type staged parity: extended nearest `poly_mmq_tiles_test`, 57 cases PASS including
  nine actual expert shape/type cases with fits=1, split_exact=1 and zero differing rows.
  No stream-k tolerance was used; these cases are bitwise equal and finite.
- F combined split+prefill1+tiles: cold pp860.1 (ratio1.37594 vs screen A), tg44.1/44.2/44.4
  (geomean ratio1.11918); passes screening only. Combined/spec-screen min free2926.52MiB,
  no safety reasons. Its separate baseline was pp646.1 and tg41.5/41.6/41.3.
- Initial MTP+window4/8 screen (without --no-draft) loaded10496 rather than10240 routed experts,
  proving a draft layer was actually active. Tg ratios1.23807/1.24808 and accepted/offered
  1062/1215,1082/1232 cannot be attributed to suffix-only lookup and will not be pooled with it.
  User explicitly authorized this MTP lever as a distinct R3 family; these one-pass screens
  remain valid and are not repeated. Current engine summary aggregates MTP and suffix guesses,
  so source-specific acceptance cannot be recovered from that summary.
- Lookup-only screens: window4 pp645.5, tg39.3/39.9/40.0, ratio0.95817 and55/112 guesses
  accepted; window8 pp647.5, tg40.1/40.4/40.6, ratio0.97346 and116/199 accepted, relative to
  separate spec baseline. Both negative screens; window8 is the better diagnostic comparison,
  not an opt-in recommendation. Min free2947.96MiB; empty final guard reasons.
- Completed R2 B: tg ratio1.05263067 [1.02673234,1.07918226], confirmed and precise but below
  the+10% lower-bound target. Prefill-time upper1.01556820 passes the1.02 budget. C: pp
  ratio1.28544049 [1.27732826,1.29360424], confirmed/precise and exceeds target; decode-time
  upper0.99811136 passes budget. Each has five pairs+A/A; no forced/retrieval qualification.
- D completed only two pairs (pp ratios1.23537/1.22274); no frozen CI. Third baseline D-p3-A
  failed on request3 with HTTP503/exit-13 and GPU page fault. E/F confirmations and all R3
  pairs/joint final screen never started. No failed/incomplete series is labelled a no-gain result.
- **Unresolved safety risk, further work waived:** at2026-11-14 00:00:59.784+01:00, journal names PID495248 and
  SDMA1 page faults from0x7f71eb200000. Failing baseline had graphs on, actual static-queue
  env0, VALUE_SPLIT=1, all fork candidate flags unset and IQ4_ROWS absent. Minimum sampled
  free VRAM2878.28515625MiB across1836 pair-window samples, above1536; fault still occurred.
- Cleanup: watcher joined; final check recorded the fault; no further pass ran. All model/vision
  children are gone; strata remains inactive/dead/PID0. No driver/system/runtime action taken.
  Private evidence: `$TMPDIR/opencode/boost-20261113/kernel-pairs/{kernel.jsonl,vram.jsonl,
  stop-reasons.json,D-p3-A/}` and `kernel-after-stop.jsonl`.
- Updated Sol review (`ses_e2931e416ffegROXJgrjJbb35Z`) read the actual final diff and all seven
  new files, independently matched all28 completed passes to native summaries, recomputed B/C
  intervals/A-A, checked speculation argv/acceptance and the actual safety stop. No blocking or
  nonblocking findings for this stopped checkpoint. SDMA1 blocker remains; no recovery,
  publication/main, deployment or qualification approval was implied. The user subsequently
  authorized publication of the frozen partial result; final publication review is next.

## Material Decisions

- 2026-11-13: user approved this cycle ("делай копию плана в цель и начинай итеративную
  реализацию, жду результат, потом коммит пуш"). Commit and push are in scope; service deployment
  is not (the service stays inactive; a later user request deploys the winning config).
- 2026-11-13: phase-timer commit `317912bb` is not ported; this cycle needs no instrumentation
  runtime change. Fork defaults (including RAM expert autobudget and picture-prompt text read)
  remain unchanged. This goal is updated only in the new worktree; the original untracked copy
  and all other foreign dirty state are left untouched.
- 2026-11-13: **model scope changed by explicit user choice.** The frozen Occamy IQ4_XS file was
  deleted from disk on 2026-11-11 and the user chose to continue on
  `/home/stfu/.lmstudio/models/unsloth/Qwen-3.6-35B-A3B/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`
  (18,536,192,288 B; same qwen35moe family: 41 blocks, 2048 embd, 256 experts/8, 16q/2kv/256dim,
  ctx 262144; mixed dynamic quant F32/Q8_0/IQ3_S/IQ4_NL/Q6_K/Q3_K/Q4_K/BF16; no MTP/draft
  tensors). Consequences recorded: the IQ4 row-geometry opt-in is dormant on this model (port
  stays in the branch); attention opt-ins (SPLIT_ATTEND, PREFILL_FP32) are quant-agnostic and
  unaffected; STAGED_TILES applicability must be verified per actual expert MMQ types; the
  2026-11-11 Occamy baseline (644–667 pp / 34.7–46.7 tg) is NOT a comparison reference for this
  model — a fresh flag-off baseline on the new binary and new model is the reference arm. A new
  `mmproj-F16.gguf` (899,283,584 B) replaced the old projector file; any deployment-time config
  updates (model path, projector, tokenizer pack) belong to the separate deployment approval.
- 2026-11-13: actual window-run logs supersede the inventory's assumption of no draft block:
  removing --no-draft loads an additional256 experts (10496 total), whereas baseline loads10240.
  `Run::answer` (`main.cpp:503–525`) enables suffix repeats independently of `engine.drafts()`.
  The safe minimum satisfying the latest **suffix/lookup-only** instruction is retaining
  --no-draft while setting window4/8 and lookup3. Initial MTP-present screens stay separate;
  no engine/model mutation or MTP performance recommendation is made.
- 2026-11-13: user clarified that detected MTP is a legitimate existing-engine lever in R3,
  not an out-of-scope model change. Premise "no draft tensors" applied to Occamy, not the
  selected UD-IQ4_NL file: engine loads its 41st block as draft when allowed. Keep baseline,
  lookup-only and MTP+window statistics separate, confirm each best speculation family after
  kernel pairs, and screen F+best MTP if its effect exceeds lookup-only. Needle for MTP is
  mandatory. Existing MTP screens are reused exactly once, not rerun or relabelled suffix-only.
- 2026-11-14: **cycle frozen on confirmed results by explicit user choice** ("берём эти правки,
  вноси и коммит пуш") after the safety stop. The recovery checkpoint (remaining D/E/F pairs,
  strict forced qualification, MTP confirmation, needle, final GPU retest) is **waived_by_user**,
  not completed. R2 retains verified B/C, partial D and unconfirmed E/F screens; R3 retains
  unconfirmed MTP and negative lookup screens; unexecuted R4 qualification stays unverified.
  Finalize report/goal, commit task artifacts and push R6. No deployment, defaults change or
  new GPU run. Completion means the user-revised publication contract, not full GPU qualification.
  Before future deployment: resolve safety in a separately approved cycle, F/joint-config pairs,
  MTP confirmation, >=44k needle, forced equivalence, final retest/stability and model/mmproj
  config transition. The old Occamy configuration is not a validated rollback for the new file.

## Checkpoint History

- 2026-11-13: goal created from the approved plan; RECON evidence recorded above.
- 2026-11-13: R1 port/configure checkpoint passed (clean cherry-pick, unchanged ggml pin,
  no missing dependency cache blocker).
- 2026-11-13: R1 build completed; seven CTests passed, IQ4_ROWS=1 parity passed. Actual-model
  parity failed before model loading because the required GGUF is absent; R1–R4 blocked, no
  screening/confirmation/speculation/retrieval trial launched. Parent unlock required.
- 2026-11-13: final blocked-checkpoint Sol review found no blocking issues; independently
  checked source/default path, prepared runners, evidence hashes, 7/7 tests, eight bitwise checks
  and missing model. No additional GPU exposure, report commit, push or deployment authorized.
- 2026-11-13: user explicitly changed model scope, unblocking inference. New-model load with the
  existing tokenizer and actual-model expert parity passed; R1 verified. MMQ covers every observed
  expert type including Q6_K; native log shows 40 main MoE layers. A–E screen started with only
  VALUE_SPLIT fixed and dormant IQ4_ROWS removed; old Occamy numbers are historical only.
- 2026-11-13: A–E screening complete; B/C/D/E all pass the preliminary continuation gate,
  without a gain claim. Actual-type staged parity57/57 PASS, all new cases bitwise exact.
  Combined F and separate R3 screen started; no new faults and min screen headroom3021.04MiB.
- 2026-11-13: F screening positive (pp860.1, tg44.1/44.2/44.4). Initial window runs revealed
  a real draft layer; separated from the requested suffix-only series. Correct suffix-only
  screen started with --no-draft; all R2 positive arms remain queued for five pairs plus A/A.
- 2026-11-13: lookup-only screen negative (tg ratios0.95817/0.97346); MTP8 remains strongest
  speculation screen. R2 confirmation60-pass schedule started before any further R3 work;
  no extension beyond the frozen five pairs and one A/A per kernel arm is planned.
- 2026-11-14: mandatory safety stop during D's third baseline. B/C five-pair+A/A series completed,
  D only two pairs, E/F and R3 confirmation never began. Fresh SDMA1 fault occurred with graphs
  on and static queues; final guard caught it and left no model/service running. All subsequent
  GPU gates stopped. Offline B/C intervals computed; full goal blocked, not complete.
- 2026-11-14: updated Sol review of final stopped checkpoint found no findings and independently
  confirmed all28 completed passes, B/C statistics, speculation separation and SDMA1 stop.
  Closure fails remaining qualification outcomes and safety boundary; no further substantive
  action or GPU run without parent recovery/replan, no push/deployment.
- 2026-11-14: user cancelled recovery and froze the cycle on confirmed partial results,
  explicitly waiving unfinished measurement/GPU gates and authorizing final commit/push only.
  R1 verified, R2–R4 partially verified with remaining scope waived, R5 final report verified;
  R6 publication in progress. No new GPU execution.

## Completion

Pending R6 publication only under the user-revised finish line. R1 and B/C measurements verified;
unfinished R2–R4 confirmation/qualification explicitly waived_by_user, not passed. R5 verified.
After checked publication, this goal may be complete **only as a frozen partial research result**.
F/MTP/joint-config confirmation, needle, forced equivalence, final GPU retest, stability/safety
resolution and new-model/mmproj config transition remain gates for a separately approved future
deployment. No defaults, service, installed engine or foreign main-worktree state were changed.
