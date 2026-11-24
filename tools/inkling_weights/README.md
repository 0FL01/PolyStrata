# Inkling weight experiment: frozen P0 corpus

The [manifest](../../docs/goals/inkling2-weights/corpus-manifest.json) is the frozen input contract. It is not a model result. The artifacts default to `$HOME/.cache/opencode-tmp/opencode/inkling-weights/corpus` and are not checked into Git.

```sh
python tools/inkling_weights/corpus.py verify
python tools/inkling_weights/corpus.py build
python -m unittest discover -s tools/inkling_weights -p test_corpus.py -v
```

`verify` checks the materialized artifacts without the GGUF or a network connection. `build` requires Python 3.11+, the repo tokenizer's existing `regex` package, Git with commit `0051297f6d250f24fca26f68ee8f8ca6acb9f7e8`, and the exact H0 GGUF. By default it reads `$HOME/.lmstudio/models/unsloth/Inkling-2-Mini/Inkling-2-Mini-Q4_0.gguf`; `--model` may point to another copy of the same file. Build checks all 15,064,977,088 bytes against SHA-256 `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37`, checks the tokenizer code against the pinned commit, and requires the complete regenerated contract to match the manifest before writing corpus outputs. It won't overwrite a file with different bytes. Neither command runs inference.

The two Russian prose sources are Tolstoy's *После бала* and *Смерть Ивана Ильича* from Russian Wikisource. The manifest credits and pins both transcriptions by revision URL and hashes their extracted UTF-8 text. To rebuild offline, retain the two `sources/*.txt` files listed in the manifest. To recover missing copies, use `build --fetch-public` to fetch only those pinned public pages, extract article paragraphs, and check the extracted bytes against the manifest. The remaining sources come from Git blobs at the pinned commit, not the working tree. `--root` accepts a separate directory outside the checkout for reproducibility checks.

## NLL and forced decode

- For each of `2k` (2,048 tokens) and `16k` (16,384 tokens), there is one Russian prose excerpt, one English documentation excerpt, one Python source excerpt, and one C++ source excerpt. Each 256-token target comes directly after its prefix in a single source tokenization. The builder checks UTF-8 and the tokenization of the prefix and joined window; a target can start or end within a word and must not be tokenized on its own.
- Every `prefix`, `target`, and `forced` record contains relative `text_path` and `ids_path` paths, per-file SHA-256, and an exact token count. The ID files contain unsigned decimal IDs in ASCII, separated by commas, with no header, spaces, or trailing newline. Text files are exact UTF-8 with no added newline. These are plain IDs, without a chat template, BOS, EOS, or padding. All are in the 200,058-token active vocabulary. The builder excludes every non-normal token, including stop ID `200006` (H0's EOS/EOT and the Inkling server end-of-sampling marker).
- `perf[split].nll_prefix_id` selects that split's English prefix; `perf[split].forced` is its 512-token continuation, including the corresponding 256-token NLL target at its start. Pass the two ID files as `--tokens "$(cat PREFIX.ids.csv)" --follow "$(cat FORCED.ids.csv)" --max-new 512` with the separately locked benchmark settings. Check the actual emitted count and IDs against the frozen suffix. The current CLI emits the last token without feeding it back, so 512 emitted tokens do not imply 512 post-prefill single-token evals; use the same accounting for every model. The two lengths use different text, so compare versions at matched length/prefix, not speed by length in isolation.
- For NLL, take the logits after the prefix, score the first target ID in those logits, then feed that ID and continue. Never use chat or retokenize the saved snippets for the timed or NLL inputs. CLI parity, an exact-NLL observer, and model results are outside P0.

## Short tasks and Orca diagnostics

`tasks` holds one set of 32 short prompts: 6 Python, 6 Rust, 4 JSON, 4 simulated tool records, 8 Russian instruction-following, and 4 logic prompts. The stable `2k-` case IDs do not imply padding or a context length. Use the inline `prompt` as a single user message with the separately locked template and effort; the prompt ID artifact describes *unwrapped* text, not a formatted chat prompt. Apply each case's `max_new_tokens` as a budget for the full completion, including any reasoning. Pass only final-answer content (not reasoning, chat markers, or transport envelopes) to the checker:

```sh
printf '%s' 'abs(x-y)' | python tools/inkling_weights/validate_tasks.py 2k-py-01
```

The checker returns JSON with `pass` and exits 0 on pass or 1 on failure. Python checks use a bounded integer/boolean AST interpreter and fixed vectors. Rust checks match explicit token skeletons, not compiler results. Neither checker executes or compiles the generated answer. JSON checks require exact values and types, reject extra or duplicate keys, and do not dispatch tools. Literal answers are exact apart from surrounding whitespace. The unit tests use hand-written/reference answers, not model outputs. `smoke_candidate` marks cases eligible for the later H0-passing critical smoke; no task is yet known to pass on H0.

`orca.cases` freezes 12 benign diagnostic candidates and 12 paired ordinary controls. At `low` effort, use the same template/system/sampler for H0/H3 and any permitted later comparison. First check whether H0 or H3 actually shows a refusal on a benign request. Record the reason with the raw final answer, separating a refusal from a wrong answer, format failure, or truncation. The automated checker measures correct completion; its failure alone is **not** a refusal label. If no such symptom is observed, do not claim there is a baseline problem or that a donor might have fixed it. This pilot does not estimate a general refusal rate.

## Reflink weight builds

`transplant build` uses Linux reflinks by default. Put the output on the same Btrfs filesystem as `--base`; a failed clone stops rather than materializing a full copy. The clone keeps the parent's blocks and offsets for every unchanged tensor, appends just the replacement payloads, and updates their descriptors. `du` and file size can count shared data more than once; use `btrfs filesystem du -s` to check actual sharing. Every clone is still reread and fully hashed before publication. Reuse the resulting file path for all GPU runs; only the small expert profile and logs are per-run copies.

`--reuse-verified OLD_MANIFEST` reuses the old artifact's exact replacement tensors without reading the donor from the network. It verifies their hashes and checks all non-selected hashes against the new parent. The reflink layout can have a different whole-file SHA and a larger logical size than the old packed layout despite identical tensor payloads and typed metadata. Use `--compact` only to reproduce the historical packed builds, not for normal staging.

### Fetch the A1 replacement tensors

After H3 is verified, pipe the active authorized Hugging Face token to the A1 writer. The HF CLI's Python can be separate from the repository's Python. Use new private output paths:

```sh
"$HOME/.hf-cli/venv/bin/python" -I -B -c \
  'from huggingface_hub import get_token; print(get_token() or "")' |
  python3 -B -m tools.inkling_weights.transplant build --variant A1 \
    --base "$H3" --parent-manifest "$H3_MANIFEST" --output "$A1" --manifest "$A1_MANIFEST"
```

This fetches only the pinned header and 24 selected projections. The sidecar records the exact copied payload hashes and distinguishes the advertised donor digest from a whole-file verification. It inherits H3's architecture/tokenizer metadata and every other payload. Successful build alone does not establish a reduction in false refusals, qualify the model, or run the full donor control.

## Guarded runtime pilots

Use these from the checkout with production inactive. Run them serially on the one GPU. Each engine process gets its own scratch directory and writable seed profile; none reads production sessions. The binaries, corpus, template and runtime flags come from the [experiment lock](../../docs/goals/inkling2-weights/experiment-lock.json).

For these pilots only, AMDGPU memory/page-fault journal entries are warnings, whether historical or live. They are preserved in `guard/warnings.json` and do not require a reboot or by themselves fail a run. This does not mask GPU resets, hangs, OOM, low VRAM, invalid outputs, nonzero exits or timeouts. `run_guarded.py` remains strict by default for other callers.

```sh
WORK="$HOME/.cache/opencode-tmp/opencode/inkling-weights"
.venv/bin/python -B -m tools.inkling_weights.chat_pilot --smoke --model "$MODEL" --output "$WORK/candidate-chat"
.venv/bin/python -B -m tools.inkling_weights.nll_pilot --model "$MODEL" --output "$WORK/candidate-nll" \
  --observer "$OBSERVER" --case 2k-en
python3 -B -m tools.inkling_weights.timing --quick --model-b "$MODEL" --sha256-b "$MODEL_SHA256"
```

`--smoke` runs the four H0-passing critical tasks instead of all 32. Chat uses the shipped Inkling output parser and passes only final content to the validators. `--suite orca` selects the separate low-effort diagnostic; its validator cannot classify refusals. Both quality pilots accept repeated `--case`. For a full paired NLL comparison, use `--reference` with a result from the same documents.

Use `--quick` timing on a finalist: after the two warm-ups, one measured pair per length (`AB` at 2K, `BA` at 16K), eight launches instead of 20. It is a screen, not a noise estimate. Run without `--quick` only when a decision needs the full `ABBABAAB` comparison or to reproduce P5. Both modes check all 512 output IDs (511 timed decode steps), retain sampled VRAM and keep observer/logit printing off. With no model flags both arms use H0; set `--model-a` and `--sha256-a` for a different parent.
