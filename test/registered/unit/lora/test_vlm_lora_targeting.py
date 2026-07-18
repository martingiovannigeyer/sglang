"""CPU-only coverage for VLM LoRA module selection and GDN targets."""

import types
import unittest

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.utils import get_normalized_target_modules
from sglang.srt.models.minicpmv import MiniCPMV4_6
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


class TestVlmLoraTargeting(unittest.TestCase):
    def test_manager_honors_model_target_filter(self):
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = types.SimpleNamespace(
            should_apply_lora=lambda name: name.startswith("language.")
        )

        self.assertTrue(manager._should_apply_lora("language.layers.0.q_proj"))
        self.assertFalse(manager._should_apply_lora("vision.layers.0.q_proj"))

    def test_manager_keeps_models_without_filter_compatible(self):
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = types.SimpleNamespace()

        self.assertTrue(manager._should_apply_lora("model.layers.0.q_proj"))

    def test_minicpm_v46_only_targets_language_layers(self):
        model = MiniCPMV4_6.__new__(MiniCPMV4_6)

        self.assertTrue(
            model.should_apply_lora("llm.model.layers.0.linear_attn.out_proj")
        )
        self.assertTrue(model.should_apply_lora("llm.model.layers.23.mlp.down_proj"))
        self.assertFalse(model.should_apply_lora("vpm.encoder.layers.26.mlp.down_proj"))
        self.assertFalse(model.should_apply_lora("resampler.mlp.down_proj"))

    def test_split_gdn_targets_normalize_to_fused_runtime_modules(self):
        targets = get_normalized_target_modules(
            ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
        )

        self.assertEqual(targets, {"in_proj_qkvz", "in_proj_ba"})

    def test_split_gdn_weights_normalize_to_fused_runtime_modules(self):
        adapter = LoRAAdapter.__new__(LoRAAdapter)
        weights = {
            "layer.in_proj_qkv.lora_A.weight": torch.ones(2, 3),
            "layer.in_proj_z.lora_A.weight": torch.ones(2, 3),
            "layer.in_proj_qkv.lora_B.weight": torch.ones(6, 2),
            "layer.in_proj_z.lora_B.weight": torch.ones(2, 2),
            "layer.in_proj_b.lora_A.weight": torch.ones(2, 3),
            "layer.in_proj_a.lora_A.weight": torch.ones(2, 3),
            "layer.in_proj_b.lora_B.weight": torch.ones(4, 2),
            "layer.in_proj_a.lora_B.weight": torch.ones(4, 2),
        }

        adapter._normalize_in_proj_qkvz(weights)
        adapter._normalize_in_proj_ba(weights)

        self.assertEqual(weights["layer.in_proj_qkvz.lora_A.weight"].shape, (8, 3))
        self.assertEqual(weights["layer.in_proj_qkvz.lora_B.weight"].shape, (8, 2))
        self.assertEqual(weights["layer.in_proj_ba.lora_A.weight"].shape, (4, 3))
        self.assertEqual(weights["layer.in_proj_ba.lora_B.weight"].shape, (8, 2))


if __name__ == "__main__":
    unittest.main()
