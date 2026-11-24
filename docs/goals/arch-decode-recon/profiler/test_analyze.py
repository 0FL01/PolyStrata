"""CPU-only tests: python -B docs/goals/arch-decode-recon/profiler/test_analyze.py"""

from pathlib import Path
import unittest
from unittest.mock import patch

from analyze import Event, category, decode_windows, inspect, per_window_stats, symbol, tag_stream, union


def event(name, start, end, stream="1", **kwargs):
    return Event(start, end, name, symbol(name), stream, agent="Agent 1", **kwargs)


class AnalysisTests(unittest.TestCase):
    def test_union_and_overlap_across_token_boundary(self):
        self.assertEqual(union([(6, 10), (1, 9), (10, 12), (13, 13), (15, 20)]), [(1, 12), (15, 20)])
        k = [event("a", 4, 9), event("b", 6, 8, "2"), event("c", 12, 17)]
        c = [event("copy", 7, 13, "2")]
        a, b = per_window_stats(k, c, [(0, 10), (10, 20)])
        self.assertEqual((a["kernel_busy_ns"], a["traced_device_busy_ns"], a["kernel_copy_overlap_ns"]), (5, 6, 2))
        self.assertEqual((b["kernel_busy_ns"], b["traced_device_busy_ns"], b["kernel_copy_overlap_ns"]), (5, 7, 1))

    def test_copy_and_blit_boundaries_exclude_prefill(self):
        stream = [event("softcap_kernel", 10, 12), event("__amd_rocclr_copyBuffer", 13, 15),
                  event("init", 30, 32), event("softcap_kernel", 40, 42)]
        copies = [event("MEMORY_COPY_DEVICE_TO_HOST", 43, 45)]
        w, methods, boundary = decode_windows(stream, copies, 1)
        self.assertEqual((w, methods, boundary), ([(15, 45)], {"rocclr_copybuffer": 1, "d2h_record": 1}, "logits_copy_end"))
        self.assertEqual(copies[0].category, "copy_logits_d2h")
        self.assertEqual(decode_windows(stream, [], 1)[0], [(12, 42)])
        with self.assertRaisesRegex(ValueError, "expected 3"):
            decode_windows(stream, copies, 2)

    def test_fused_controller_and_logits_scoping(self):
        names = ["mix3_kernel", "control_in3_kernel", "gemv_rows_segments_kernel", "control_out3_kernel",
                 "mix_kernel", "rms_norm_kernel", "native_quantize_q8_1_kernel", "native_q6_k_warps_kernel<2>",
                 "softcap_kernel"]
        k = [event(n, i * 2, i * 2 + 1) for i, n in enumerate(names)]
        for e in k:
            e.category = category(e)
        matched, unmatched = tag_stream(k)
        self.assertEqual((dict(matched), dict(unmatched)), ({"control_in3_kernel": 1, "softcap_kernel": 1}, {}))
        self.assertEqual([e.category for e in k], ["norm_quant_mix_residual", "controller_input_fused", "controller_rank",
                                                 "controller_output_fused", "lm_head_input", "lm_head_input",
                                                 "lm_head_input", "lm_head_projection", "lm_head_softcap"])

    def test_unfused_path_and_attention_names(self):
        names = ["mix_kernel", "rms_norm_kernel", "rms_norm_kernel", "gemv_kernel", "gemv_kernel",
                 "gemv_kernel", "control_kernel"]
        k = [event(n, i * 2, i * 2 + 1) for i, n in enumerate(names)]
        self.assertEqual(dict(tag_stream(k)[0]), {"control_kernel": 1})
        self.assertEqual(k[0].category, "controller_bank_mix")
        self.assertEqual(k[3].category, "controller_rank")
        self.assertEqual(category(event("void strata::poly::inkling::(anonymous namespace)::decode_q8_kernel<8>(float*)",
                                        0, 1)), "attention")
        self.assertEqual(category(event("void strata::kernels::(anonymous namespace)::inkling_q4_0_slots_kernel<72, 4>()",
                                        0, 1)), "expert_q4_0")

    def test_host_api_is_separate_and_nested_calls_are_unioned(self):
        kernel, copies, api = map(Path, ("kernel", "copy", "api"))
        k = [event("softcap_kernel", 5, 10), event("control_in_kernel", 18, 25, correlation="2"),
             event("gemv_rows_kernel", 25, 30, correlation="2"), event("control_out_kernel", 32, 36, correlation="2"),
             event("native_q6_k_warps_kernel", 36, 40, correlation="3"), event("softcap_kernel", 42, 45)]
        c = [event("MEMORY_COPY_HOST_TO_DEVICE", 22, 27, "2", dst="Agent 1", src="Agent 0", size=64)]
        h = [event("hipGraphLaunch", 12, 23, correlation="2"), event("hipLaunchKernel", 14, 18, correlation="2"),
             event("hipEventSynchronize", 27, 37), event("hipMemcpyAsync", 38, 40)]
        with patch("analyze.read_events", side_effect=lambda path, domain: {kernel: k, copies: c, api: h}[path]):
            out = inspect(kernel, copies, api, None, 1)
        self.assertAlmostEqual(out["hip_api"]["by_group"]["launch_api"]["union_ms_per_token"] * 1e6, 11)
        self.assertAlmostEqual(out["mean"]["traced_device_busy_ms_per_token"] * 1e6, 23)
        self.assertAlmostEqual(out["controller"]["hoist_input_dispatch_union_ms_per_token"] * 1e6, 7)
        self.assertAlmostEqual(out["controller"]["hoist_input_exclusive_device_ms_per_token"] * 1e6, 4)


if __name__ == "__main__":
    unittest.main()
