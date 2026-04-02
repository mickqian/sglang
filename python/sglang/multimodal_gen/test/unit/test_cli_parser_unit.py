import tempfile
import unittest
from pathlib import Path

import yaml

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.utils import FlexibleArgumentParser


class TestFlexibleArgumentParserConfig(unittest.TestCase):
    def test_parse_known_args_loads_sampling_params_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ltx2.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "prompt": "A beautiful sunset over the ocean",
                        "negative_prompt": "shaky, glitchy, low quality",
                        "seed": 1234,
                    }
                )
            )

            parser = FlexibleArgumentParser()
            subparsers = parser.add_subparsers(dest="subparser")
            generate_parser = subparsers.add_parser("generate")
            generate_parser.add_argument("--config", type=str, default="")
            SamplingParams.add_cli_args(generate_parser)

            args, unknown_args = parser.parse_known_args(
                ["generate", "--config", str(config_path)]
            )

        self.assertEqual(args.subparser, "generate")
        self.assertEqual(args.config, str(config_path))
        self.assertEqual(args.prompt, ["A beautiful sunset over the ocean"])
        self.assertEqual(args.negative_prompt, "shaky, glitchy, low quality")
        self.assertEqual(args.seed, 1234)
        self.assertEqual(unknown_args, [])


if __name__ == "__main__":
    unittest.main()
