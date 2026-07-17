"""CPU regressions for mixed-precision compressed-tensors configs."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import Mock, patch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
    should_ignore_layer,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.test.test_utils import CustomTestCase

FP8_TARGET = "re:.*(q_proj|lm_head)$"
NVFP4_TARGET = "re:.*mlp.experts.*"


def _mixed_precision_config():
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "fp8": {
                "format": "float-quantized",
                "targets": [FP8_TARGET],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
            },
            "nvfp4": {
                "format": "nvfp4-pack-quantized",
                "targets": [NVFP4_TARGET],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "dynamic": True,
                },
            },
        },
        "ignore": [],
    }


class TestCompressedTensorsMixedPrecision(CustomTestCase):
    def test_parses_per_group_activation_quantization(self):
        quant_config = CompressedTensorsConfig.from_config(_mixed_precision_config())

        fp8_input = quant_config.target_scheme_map[FP8_TARGET]["input_activations"]
        nvfp4_input = quant_config.target_scheme_map[NVFP4_TARGET]["input_activations"]

        self.assertIsNotNone(fp8_input)
        self.assertEqual(fp8_input.num_bits, 8)
        self.assertIsNotNone(nvfp4_input)
        self.assertEqual(nvfp4_input.num_bits, 4)

    def test_selects_activation_scheme_for_mixed_precision_groups(self):
        quant_config = CompressedTensorsConfig.from_config(_mixed_precision_config())

        with patch.object(quant_config, "_check_scheme_supported", return_value=True):
            fp8 = quant_config.target_scheme_map[FP8_TARGET]
            nvfp4 = quant_config.target_scheme_map[NVFP4_TARGET]

            self.assertIsInstance(
                quant_config._get_scheme_from_parts(
                    fp8["weights"], fp8["input_activations"]
                ),
                CompressedTensorsW8A8Fp8,
            )
            self.assertIsInstance(
                quant_config._get_scheme_from_parts(
                    nvfp4["weights"], nvfp4["input_activations"]
                ),
                CompressedTensorsW4A4Fp4,
            )

    def test_keeps_weight_only_pack_quantized_groups_valid(self):
        config = _mixed_precision_config()
        config["config_groups"] = {
            "weight_only": {
                "format": "pack-quantized",
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                    "dynamic": False,
                },
                "input_activations": None,
            }
        }

        quant_config = CompressedTensorsConfig.from_config(config)

        self.assertIsNone(quant_config.target_scheme_map["Linear"]["input_activations"])

    def test_ignore_entries_match_exact_names_or_regexes_only(self):
        parent = "model.layers.0.linear_attn"
        child = f"{parent}.in_proj_qkv"

        self.assertTrue(check_equal_or_regex_match(parent, [parent]))
        self.assertFalse(check_equal_or_regex_match(child, [parent]))
        self.assertTrue(check_equal_or_regex_match(child, [r"re:.*in_proj_qkv$"]))
        self.assertFalse(should_ignore_layer(child, ignore=[parent]))

    def test_quantizes_parallel_lm_head_when_targeted(self):
        quant_config = CompressedTensorsConfig.from_config(_mixed_precision_config())
        layer = Mock(spec=ParallelLMHead)
        scheme = object()

        with patch.object(
            quant_config, "get_linear_scheme", return_value=scheme
        ) as get_scheme:
            method = quant_config.get_quant_method(layer, prefix="model.lm_head")

        self.assertIsInstance(method, CompressedTensorsLinearMethod)
        self.assertIs(layer.scheme, scheme)
        get_scheme.assert_called_once_with(layer=layer, layer_name="model.lm_head")


if __name__ == "__main__":
    unittest.main()
