# AGENTS.md

This fork targets **AMD RDNA2 only**, using HIP and system RAM. Develop, optimize, and validate for RDNA2; other GPU
architectures and backends are out of scope. Inherited backends may remain, but do not add compatibility work or
require builds and tests for them.

The primary focus is developing and optimizing [Inkling 2 Mini](docs/INKLING2.md) inference in `strata-poly` (`src/poly/`).
Qwen-specific work is deferred and out of scope.

The Qwen engine (`src/` and `include/`), installer (`setup.py`), and API server and web app (`serve/`) come from
[Strata](https://github.com/Niko1221/Strata).

## Installing PolyStrata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, PolyStrata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- During R&D, you may stop the production server at any time, even while people are using it.
- Run guarded RDNA2 benchmarks with `rdna2-rnd/scripts/run_guarded.py --warn-amdgpu-faults`.
  An AMDGPU memory/page fault alone, earlier in the boot or during a run, is a warning, not an error or blocker.
  Continue using the GPU; resets, hangs, timeouts, OOM, failed commands and the free-VRAM limit still stop the run.
- On the Btrfs profiling host, keep each A/B series in `~/.cache/opencode-tmp/opencode/polystrata-runs/<run>/`.
  Reuse read-only inputs or copy with `cp --reflink=always`. Keep SAVE outputs independent until validated;
  once writers and benchmarks stop, run `~/.local/bin/polystrata-dedupe <run>` without sudo.
- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- Build the affected HIP target and run the closest relevant checks on RDNA2; report the GPU tested. See
  [docs/AMD_HIP.md](docs/AMD_HIP.md) for build and validation.
- For each inference optimization, compare generated token IDs before and after on the same GPU and model,
  using the same prompts that exercise the optimized path and the same greedy decoding settings.
  Report whether the sequences match and where they first differ if they don't.
- Work on `main` and push task commits to `origin/main` unless the user directs otherwise.
- Give every commit a clear subject and a body separated by a blank line. In the body, explain what changed
  and why, naming the affected code paths or flags where relevant. Use bullets for distinct changes.
  Say what was checked and what stayed unchanged. For performance changes, give before/after numbers with
  the baseline, hardware, and test conditions; include any measured regressions.
- Text taken from the Strata project names that project. After merging from it,
  `python tools/polystrata_names.py` writes PolyStrata again; its first lines say what it leaves alone.
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
