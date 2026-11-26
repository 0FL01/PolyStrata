# Short interactive qualification

`run.py` uses the existing private billing fixture and its AST interpreter, not a
written transcript or execution of generated Python. Every arm starts each turn
by rendering its own conversation and starting a fresh native `--tokens` process.
It dispatches one parsed, allowlisted read after each of the first two turns and
requires natural EOS and the existing 118-check grader on turn three. A failed
grade has no time to correct. Do not use `--follow` for these runs.

Keep logs, prompts, tool replies and IDs under `$R/trajectory`, where `$R` is
`~/.cache/opencode-tmp/opencode/polystrata-runs/dflash2-recon`. The service must
already be stopped by the coordinating GPU owner. The runner holds `$R/gpu.lock`
and uses the ordinary guard with `--warn-amdgpu-faults`.

Default primary seed 42, optional held-out 79; low reasoning, T=1, unrestricted
top-k/top-p/min-p, 768 IDs per turn, lookup off. Use the installed ELF and settings
as the no-draft control, with 2,048 context, 256 prompt batch, fresh cloned profile,
no prompt/checkpoint reuse and the same 2,048 MiB physical reserve for each arm.
`--cap 3072` is a diagnostic frozen-placement option, not the deployed policy.
Keep the installed staging size: the first variant with a 128 MiB stage failed
in prefill on gfx1031, before the first emitted ID. Equal resident counts
without a map dump are not a measured map fingerprint.

```sh
PY=.venv/bin/python
AGENT=rdna2-rnd/dflash2/trajectory
$PY -B "$AGENT/run.py" production prod-natural-42
$PY -B "$AGENT/run.py" taps taps-natural-42
$PY -B "$AGENT/run.py" dspark dspark-natural-42
WEIGHTS=/path/to/Inkling-2-Mini-DFlash2-Q4_0.gguf
$PY -B "$AGENT/run.py" live q4-k0-42 --weights "$WEIGHTS" --k 0
$PY -B "$AGENT/run.py" live q4-k3-serial-42 --weights "$WEIGHTS" --k 3 --serial-verify
$PY -B "$AGENT/run.py" live q4-k3-batched-42 --weights "$WEIGHTS" --k 3
$PY -B "$AGENT/report.py" prod-natural-42 dspark-natural-42 q4-k0-42 q4-k3-serial-42 q4-k3-batched-42
```

Set `--k` to the selected width from live target testing, not greedy replay. The
live arm uses the target's one-shot DFlash2 runner, `--window 8`, and the optional
`--serial-verify` diagnostic; `--k 0` is the drafter-loaded W1 control. The `taps`
arm is an independent free-running target trace with a probe executable, never
a performance result. Its three per-turn DFT1 files cover prompt and generated
IDs except the unconsumed final displayed ID. The JSON includes that final ID
and checks every DFT1 row's position, token and prompt/decode flag before the
trajectory can continue. The first post-read and final answer turns are full
canonical prompts, not replay of the first turn's verifier state.

The installed one-shot CLI buffers its output IDs. All measured arms therefore
use `--top-probs 1` as an identical per-token timing marker. That entails a
vocabulary scan, so rates reflect instrumented one-shot runs, not HTTP serving;
the marker reports a sampling event, not a streamed ID. That overhead can also
affect the elapsed-time part of DSpark's adaptive window policy. Startup ends at the
engine's `loaded` line; TTFT and compute exclude startup; task wall includes
all startups, process exits, tool reads and grading. Inter-token p90 pools
only within-turn intervals, including short intra-window intervals. Neither
different prompts nor different output lengths establish a same-work speedup.

## Replay on the captured billing trajectory

With the private `taps-natural-42/manifest.json` and the native replay binary
available, run:

```sh
.venv/bin/python -B rdna2-rnd/dflash2/trajectory/replay.py billing-genuine-k7-42
```

This checks the archived tool links and DFT1 alignment, then runs the five
native GGUFs and DSpark under the shared GPU lock and guard. DSpark uses CLI
`--follow` only for this proposal comparison, and the runner verifies both the
emitted transcript and the logged anchors. All variants use seven proposals and
the same captured positions. Short trailing windows and the final displayed
ID (which has no captured row) are excluded; a turn with no full window is
marked skipped. The report separates the tool-read turns, the answer turn, and
the phase of the first proposed ID. Replay acceptance is not live target speed
or a new quality grade. Keep the output under the private run directory.
