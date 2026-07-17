"""Unit tests for e2e memory-capacity log parsing and floor checks.

Guards regressions in the log/HTTP snapshot parsers that the CI memory
threshold update script and runtime checker rely on. A wrong regex or
fingerprint merge would silently drop SWA/Mamba/DSV4 floors.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.memory_threshold import (
    check_snapshot_against_floor,
    extract_snapshots_from_log,
    mean_floor,
    parse_memory_log_line,
    snapshot_from_server_info,
    threshold_key,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMemoryLogParsers(CustomTestCase):
    def test_parse_kv_combined_size(self):
        line = (
            "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 175135, "
            "KV size: 3.76 GB"
        )
        snap = parse_memory_log_line(line)
        self.assertEqual(snap["token_capacity"], 175135)
        self.assertAlmostEqual(snap["kv_cache_gb"], 3.76)

    def test_parse_kv_k_v_size(self):
        line = (
            "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 12000, "
            "K size: 0.07 GB, V size: 0.07 GB"
        )
        snap = parse_memory_log_line(line)
        self.assertEqual(snap["token_capacity"], 12000)
        self.assertAlmostEqual(snap["kv_cache_gb"], 0.14)

    def test_parse_swa(self):
        line = "SWAKVPool mem usage: 1.53 GB, swa size: 16384, full size: 81920"
        snap = parse_memory_log_line(line)
        self.assertEqual(snap["swa_size"], 16384)
        self.assertEqual(snap["full_size"], 81920)
        self.assertAlmostEqual(snap["swa_mem_gb"], 1.53)
        self.assertEqual(snap["token_capacity"], 81920)

    def test_parse_mamba(self):
        line = (
            "Mamba Cache is allocated. max_mamba_cache_size: 500, "
            "conv_state size: 0.21GB, ssm_state size: 8.81GB"
        )
        snap = parse_memory_log_line(line)
        self.assertEqual(snap["mamba_cache_size"], 500)
        self.assertAlmostEqual(snap["mamba_conv_gb"], 0.21)
        self.assertAlmostEqual(snap["mamba_ssm_gb"], 8.81)

    def test_parse_dsv4(self):
        line = (
            "DSV4 pool sizes: full=19968, swa=4864, c4=4992, c128=156, "
            "c4_state=608, c128_state=0"
        )
        snap = parse_memory_log_line(line)
        self.assertEqual(snap["dsv4_full"], 19968)
        self.assertEqual(snap["dsv4_swa"], 4864)
        self.assertEqual(snap["dsv4_c4"], 4992)
        self.assertEqual(snap["dsv4_c128"], 156)
        self.assertEqual(snap["dsv4_c4_state"], 608)
        self.assertEqual(snap["dsv4_c128_state"], 0)
        self.assertEqual(snap["token_capacity"], 19968)

    def test_tp_duplicates_collapsed_and_swa_merged(self):
        text = """
[TP0] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 774980, K size: 8.87 GB, V size: 8.87 GB
[TP1] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 774980, K size: 8.87 GB, V size: 8.87 GB
[TP0] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 968726, K size: 11.09 GB, V size: 11.09 GB
[TP1] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 968726, K size: 11.09 GB, V size: 11.09 GB
[TP0] SWAKVPool mem usage: 39.91 GB, swa size: 774980, full size: 968726
[TP1] SWAKVPool mem usage: 39.91 GB, swa size: 774980, full size: 968726
"""
        snaps = extract_snapshots_from_log(text)
        # One server start → one snapshot (sub-pool KV lines collapsed into SWA).
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].get("swa_size"), 774980)
        self.assertEqual(snaps[0].get("full_size"), 968726)
        self.assertEqual(snaps[0].get("token_capacity"), 968726)
        self.assertAlmostEqual(snaps[0].get("swa_mem_gb"), 39.91)
        # Larger of the two sub-pool KV sizes is kept.
        self.assertAlmostEqual(snaps[0].get("kv_cache_gb"), 22.18)

    def test_mamba_plus_kv_merge(self):
        text = """
Mamba Cache is allocated. max_mamba_cache_size: 500, conv_state size: 0.21GB, ssm_state size: 8.81GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 12000, K size: 0.07 GB, V size: 0.07 GB
"""
        snaps = extract_snapshots_from_log(text)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]["mamba_cache_size"], 500)
        self.assertEqual(snaps[0]["token_capacity"], 12000)
        self.assertAlmostEqual(snaps[0]["kv_cache_gb"], 0.14)

    def test_eagle_draft_target_kept_separate(self):
        """Draft and target are two server launches (same tokens, different GB)."""
        text = """
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 33767, K size: 4.12 GB, V size: 4.12 GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 33767, K size: 0.07 GB, V size: 0.07 GB
"""
        snaps = extract_snapshots_from_log(text)
        self.assertEqual(len(snaps), 2)
        self.assertAlmostEqual(snaps[0]["kv_cache_gb"], 8.24)
        self.assertAlmostEqual(snaps[1]["kv_cache_gb"], 0.14)


class TestServerInfoSnapshot(CustomTestCase):
    def test_snapshot_from_server_info_top_level_and_internal(self):
        info = {
            "max_total_num_tokens": 1000,
            "internal_states": [
                {
                    "memory_usage": {
                        "token_capacity": 999,
                        "kvcache": 3.5,
                        "swa_size": 100,
                        "full_size": 999,
                        "mamba_cache_size": 50,
                    }
                }
            ],
        }
        snap = snapshot_from_server_info(info)
        # internal_states overrides top-level capacity when present
        self.assertEqual(snap["token_capacity"], 999)
        self.assertAlmostEqual(snap["kv_cache_gb"], 3.5)
        self.assertEqual(snap["swa_size"], 100)
        self.assertEqual(snap["mamba_cache_size"], 50)


class TestFloorCheck(CustomTestCase):
    def test_mean_floor(self):
        self.assertAlmostEqual(mean_floor([100.0, 100.0], factor=0.99), 99.0)

    def test_check_passes_and_fails(self):
        floor = {"token_capacity": 990, "kv_cache_gb": 3.0}
        ok = check_snapshot_against_floor(
            {"token_capacity": 1000, "kv_cache_gb": 3.1},
            floor,
            key="t",
            launch_idx=0,
        )
        self.assertEqual(ok, [])
        bad = check_snapshot_against_floor(
            {"token_capacity": 900, "kv_cache_gb": 3.1},
            floor,
            key="t",
            launch_idx=0,
        )
        self.assertEqual(len(bad), 1)
        self.assertIn("token_capacity", bad[0])

    def test_missing_observed_field_skipped(self):
        # Floor has SWA but server only reported token_capacity — do not fail.
        failures = check_snapshot_against_floor(
            {"token_capacity": 1000},
            {"token_capacity": 990, "swa_size": 100},
            key="t",
            launch_idx=0,
        )
        self.assertEqual(failures, [])

    def test_threshold_key(self):
        self.assertEqual(
            threshold_key("base-b-test-1-gpu-small", "test/registered/foo.py"),
            "base-b-test-1-gpu-small::test/registered/foo.py",
        )


if __name__ == "__main__":
    unittest.main()
