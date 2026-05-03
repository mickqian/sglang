import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    is_ltx2_two_stage_pipeline_name,
)


class LTX2TextConnectorStage(PipelineStage):
    """
    Stage for applying LTX-2 Text Connectors to split/transform text embeddings
    into video and audio contexts.
    """

    _CONNECTOR_CACHE_MAX_ENTRIES = 4

    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors
        self._connector_cache = {}

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

        if prompt_embeds is None or prompt_attention_mask is None:
            raise ValueError(
                "LTX2TextConnectorStage requires prompt embeddings and "
                "attention mask."
            )

        cache_key = self._connector_cache_key(
            batch,
            server_args,
            prompt_embeds,
            prompt_attention_mask,
            neg_prompt_embeds,
            neg_prompt_attention_mask,
        )
        use_cache = (not batch.is_warmup) and is_ltx2_two_stage_pipeline_name(
            server_args.pipeline_class_name
        )
        if use_cache and cache_key in self._connector_cache:
            cached = self._connector_cache[cache_key]
            batch.prompt_embeds = [cached[0]]
            batch.audio_prompt_embeds = [cached[1]]
            batch.prompt_attention_mask = cached[2]
            if batch.do_classifier_free_guidance:
                batch.negative_prompt_embeds = [cached[3]]
                batch.negative_audio_prompt_embeds = [cached[4]]
                batch.negative_attention_mask = cached[5]
            return batch

        if batch.do_classifier_free_guidance:
            if neg_prompt_embeds is None or neg_prompt_attention_mask is None:
                raise ValueError(
                    "LTX2TextConnectorStage requires negative prompt embeddings "
                    "and attention mask when classifier-free guidance is enabled."
                )

            # Official LTX-2.3 processes positive and negative prompts through
            # the connector independently; batching shifts output numerics.
            dtype = prompt_embeds.dtype
            pos_additive_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max
            neg_additive_mask = (neg_prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max

            with set_forward_context(current_timestep=None, attn_metadata=None):
                pos_embeds, pos_audio_embeds, pos_mask = self.connectors(
                    prompt_embeds, pos_additive_mask, additive_mask=True
                )
                neg_embeds, neg_audio_embeds, neg_mask = self.connectors(
                    neg_prompt_embeds, neg_additive_mask, additive_mask=True
                )

            batch.prompt_embeds = [pos_embeds]
            batch.audio_prompt_embeds = [pos_audio_embeds]
            batch.prompt_attention_mask = pos_mask
            batch.negative_prompt_embeds = [neg_embeds]
            batch.negative_audio_prompt_embeds = [neg_audio_embeds]
            batch.negative_attention_mask = neg_mask
            cache_value = (
                pos_embeds,
                pos_audio_embeds,
                pos_mask,
                neg_embeds,
                neg_audio_embeds,
                neg_mask,
            )
        else:
            # Prepare additive mask for connectors (as per diffusers implementation)
            dtype = prompt_embeds.dtype
            additive_attention_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max

            with set_forward_context(current_timestep=None, attn_metadata=None):
                (
                    connector_prompt_embeds,
                    connector_audio_prompt_embeds,
                    connector_mask,
                ) = self.connectors(
                    prompt_embeds, additive_attention_mask, additive_mask=True
                )

            batch.prompt_embeds = [connector_prompt_embeds]
            batch.audio_prompt_embeds = [connector_audio_prompt_embeds]
            batch.prompt_attention_mask = connector_mask
            cache_value = (
                connector_prompt_embeds,
                connector_audio_prompt_embeds,
                connector_mask,
                None,
                None,
                None,
            )

        if use_cache:
            # The connector deterministically maps text encoder embeddings and
            # masks into the video/audio conditioning tensors used by denoising.
            # When a later real request has the same prompt inputs, reusing this
            # result skips connector work without changing those conditioning
            # tensors. Warmup requests stay outside the cache.
            if len(self._connector_cache) >= self._CONNECTOR_CACHE_MAX_ENTRIES:
                self._connector_cache.pop(next(iter(self._connector_cache)))
            self._connector_cache[cache_key] = cache_value

        return batch

    def _connector_cache_key(
        self,
        batch: Req,
        server_args: ServerArgs,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        neg_prompt_attention_mask: torch.Tensor | None,
    ):
        return (
            server_args.pipeline_class_name,
            self.freeze_for_dedup(batch.prompt),
            self.freeze_for_dedup(batch.negative_prompt),
            bool(batch.do_classifier_free_guidance),
            self.freeze_for_dedup(batch.prompt_template),
            batch.max_sequence_length,
            tuple(prompt_embeds.shape),
            str(prompt_embeds.dtype),
            tuple(prompt_attention_mask.shape),
            tuple(neg_prompt_embeds.shape) if neg_prompt_embeds is not None else None,
            str(neg_prompt_embeds.dtype) if neg_prompt_embeds is not None else None,
            (
                tuple(neg_prompt_attention_mask.shape)
                if neg_prompt_attention_mask is not None
                else None
            ),
        )
