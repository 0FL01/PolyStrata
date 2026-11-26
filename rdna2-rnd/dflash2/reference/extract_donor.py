#!/usr/bin/env python3
"""Extract the vendored llama.cpp graph functions verbatim, plus the Inkling selector patch."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

# Exact added expression from ggml-org/llama.cpp#33049 (src/models/dflash2.cpp).
POSTCAP = """        if (hparams.dflash_selector_softcap_post && hparams.f_final_logit_softcapping > 0.0f) {
            const float cap = hparams.f_final_logit_softcapping;
            score = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, score, 1.0f / cap)), cap);
        }
"""


def extract(path: Path, signature: str, indent: str = "") -> str:
    matches = re.findall(r"^" + re.escape(indent + signature) + r"[\s\S]*?^" + indent + r"}\s*?$",
                         path.read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one {signature!r} in {path}, found {len(matches)}")
    return matches[0].rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tree", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--inkling-diff", type=Path, help="verify the fragment with a local copy of PR #33049")
    a = p.parse_args()
    if a.inkling_diff is not None:
        added = "\n".join("+" + line for line in POSTCAP.splitlines())
        if a.inkling_diff.read_text().count(added) != 1:
            raise ValueError("the local diff does not contain the exact postcap fragment")

    draft = a.tree / "src/models/dflash.cpp"
    conv = extract(draft, "static ggml_tensor * build_dflash2_conv(")
    selector = extract(draft, "static void build_dflash2_selector(")
    join = "        score = ggml_add(ctx0, score, ggml_repeat(ctx0, unary_bcast, score));\n"
    if selector.count(join) != 1:
        raise ValueError("cannot uniquely locate the unary addition before the postcap fragment")
    mha = extract(a.tree / "src/llama-graph.cpp", "ggml_tensor * llm_graph_context::build_attn_mha(")
    swa = extract(a.tree / "src/llama-hparams.h", "static bool is_masked_swa(", "    ")

    fragments = {"conv": conv, "selector": selector, "mha": mha, "swa": swa, "postcap": POSTCAP}
    hashes = "".join(f'{k} {hashlib.sha256(v.encode()).hexdigest()}\\n' for k, v in fragments.items())
    a.output.write_text("// Generated; see extract_donor.py.\n"
                        f'constexpr char literal_donor_hashes[] = "{hashes}";\n\n'
                        + conv + selector.replace(join, join + POSTCAP) + mha + swa)


if __name__ == "__main__":
    main()
