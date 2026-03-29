import os

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LTX2TextConnectorStage(PipelineStage):
    """
    Stage for applying LTX-2 Text Connectors to split/transform text embeddings
    into video and audio contexts.
    """

    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors

    def _maybe_save_dump(
        self, batch: Req, file_name: str, tensor: torch.Tensor | None
    ) -> None:
        if tensor is None:
            return
        save_dir = None
        if os.environ.get("SAVE_INTERMEDIATE_TENSORS"):
            save_dir = os.environ.get("EXPERIMENTS_DIR", "/data/experiments")
        else:
            output_path = getattr(batch, "output_path", None)
            if output_path and "ltx2_example_stage1" in output_path:
                save_dir = output_path
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tensor.detach().cpu(), os.path.join(save_dir, file_name))

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Input: batch.prompt_embeds (from Gemma, [B, S, D])
        # Output: batch.prompt_embeds (Video Context), batch.audio_prompt_embeds (Audio Context)

        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask
        neg_prompt_embeds = batch.negative_prompt_embeds
        neg_prompt_attention_mask = batch.negative_attention_mask

        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0] if len(prompt_embeds) > 0 else None

        if isinstance(prompt_attention_mask, list):
            prompt_attention_mask = (
                prompt_attention_mask[0] if len(prompt_attention_mask) > 0 else None
            )

        if isinstance(neg_prompt_embeds, list):
            neg_prompt_embeds = (
                neg_prompt_embeds[0] if len(neg_prompt_embeds) > 0 else None
            )

        if isinstance(neg_prompt_attention_mask, list):
            neg_prompt_attention_mask = (
                neg_prompt_attention_mask[0]
                if len(neg_prompt_attention_mask) > 0
                else None
            )

        self._maybe_save_dump(batch, "sglang_text_prompt_embeds.pt", prompt_embeds)
        self._maybe_save_dump(
            batch, "sglang_text_prompt_attention_mask.pt", prompt_attention_mask
        )
        self._maybe_save_dump(
            batch, "sglang_text_negative_prompt_embeds.pt", neg_prompt_embeds
        )
        self._maybe_save_dump(
            batch,
            "sglang_text_negative_attention_mask.pt",
            neg_prompt_attention_mask,
        )

        # Handle CFG: Concatenate negative and positive inputs
        if batch.do_classifier_free_guidance:

            # Concatenate: [Negative, Positive]
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [neg_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        self._maybe_save_dump(
            batch,
            "sglang_connector_input_prompt_embeds.pt",
            prompt_embeds,
        )
        self._maybe_save_dump(
            batch,
            "sglang_connector_input_attention_mask.pt",
            prompt_attention_mask,
        )

        # Call connectors
        with set_forward_context(current_timestep=None, attn_metadata=None):
            if getattr(self.connectors, "per_modality_projections", False):
                (
                    connector_prompt_embeds,
                    connector_audio_prompt_embeds,
                    connector_mask,
                ) = self.connectors(
                    prompt_embeds,
                    prompt_attention_mask,
                )
            else:
                dtype = prompt_embeds.dtype
                additive_attention_mask = (1 - prompt_attention_mask.to(dtype)) * (
                    -1000000.0
                )
                (
                    connector_prompt_embeds,
                    connector_audio_prompt_embeds,
                    connector_mask,
                ) = self.connectors(
                    prompt_embeds, additive_attention_mask, additive_mask=True
                )

        self._maybe_save_dump(
            batch,
            "sglang_connector_prompt_embeds.pt",
            connector_prompt_embeds,
        )
        self._maybe_save_dump(
            batch,
            "sglang_connector_audio_prompt_embeds.pt",
            connector_audio_prompt_embeds,
        )
        self._maybe_save_dump(
            batch,
            "sglang_connector_attention_mask.pt",
            connector_mask,
        )

        # Split results if CFG was enabled
        if batch.do_classifier_free_guidance:
            neg_embeds, pos_embeds = connector_prompt_embeds.chunk(2, dim=0)
            neg_audio_embeds, pos_audio_embeds = connector_audio_prompt_embeds.chunk(
                2, dim=0
            )
            neg_mask, pos_mask = connector_mask.chunk(2, dim=0)

            batch.prompt_embeds = [pos_embeds]
            batch.audio_prompt_embeds = [pos_audio_embeds]
            batch.prompt_attention_mask = pos_mask

            batch.negative_prompt_embeds = [neg_embeds]
            batch.negative_audio_prompt_embeds = [neg_audio_embeds]
            batch.negative_attention_mask = neg_mask
        else:
            # Update positive fields
            batch.prompt_embeds = [connector_prompt_embeds]
            batch.audio_prompt_embeds = [connector_audio_prompt_embeds]
            batch.prompt_attention_mask = connector_mask

        return batch
