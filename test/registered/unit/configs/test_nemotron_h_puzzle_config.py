import unittest

from sglang.srt.configs.nemotron_h import NemotronHPuzzleConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNemotronHPuzzleConfig(unittest.TestCase):
    def make_config(self, block_configs, n_routed_experts=512):
        config = object.__new__(NemotronHPuzzleConfig)
        config.block_configs = block_configs
        config.n_routed_experts = n_routed_experts
        return config

    def test_max_experts_falls_back_to_global_config(self):
        config = self.make_config(
            [
                {"block_type": "mamba"},
                {"block_type": "moe"},
                {"block_type": "moe", "n_routed_experts": 256},
            ]
        )

        self.assertEqual(config.max_n_routed_experts, 512)

    def test_max_experts_preserves_per_block_overrides(self):
        config = self.make_config(
            [
                {"block_type": "moe", "n_routed_experts": 128},
                {"block_type": "moe", "n_routed_experts": 768},
            ]
        )

        self.assertEqual(config.max_n_routed_experts, 768)


if __name__ == "__main__":
    unittest.main()
