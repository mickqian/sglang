"""DiffGenerator.generate builds its sampling-params template from one prompt."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


class _StopAfterTemplate(Exception):
    pass


def _template_kwargs(**sampling_params_kwargs):
    generator = object.__new__(DiffGenerator)
    generator.server_args = SimpleNamespace(model_path="dummy", prompt_file_path=None)
    with patch.object(
        SamplingParams,
        "from_user_sampling_params_args",
        side_effect=_StopAfterTemplate,
    ) as build:
        with pytest.raises(_StopAfterTemplate):
            generator.generate(sampling_params_kwargs=sampling_params_kwargs)
    return build.call_args.kwargs


def test_cli_prompt_list_builds_template_from_a_string():
    # `generate --prompt a b` passes a list; model validators (MiniMax-H3-Image)
    # require the single string each cloned request carries.
    kwargs = _template_kwargs(prompt=["a red fox", "a quiet street"])
    assert kwargs["prompt"] == "a red fox"


def test_prompt_file_builds_template_from_its_first_line(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first prompt\nsecond prompt\n")
    kwargs = _template_kwargs(prompt=None, prompt_path=str(path))
    assert kwargs["prompt"] == "first prompt"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
