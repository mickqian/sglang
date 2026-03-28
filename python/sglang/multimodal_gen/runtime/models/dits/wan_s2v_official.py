# SPDX-License-Identifier: Apache-2.0
"""
Official Wan2.2 S2V engine wrapper.

This wrapper keeps the initial S2V bring-up pragmatic: the SGLang pipeline
loads a single engine object that delegates generation to the official
Wan2.2 `wan.speech2video.WanS2V` implementation.

The overlay repo is expected to materialize the official Wan python sources
under `official_code/` inside the component directory.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torchvision import transforms

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanS2VOfficialEngine(torch.nn.Module, OffloadableDiTMixin):
    _aliases = ["WanS2VOfficialEngine"]
    _sdpa_warned_padding_mask = False
    layer_names = ["blocks"]

    def __init__(
        self,
        engine: Any,
        component_model_path: str,
        config: dict[str, Any],
        scheduler_cls: type | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.component_model_path = component_model_path
        self.config = config
        self.scheduler_cls = scheduler_cls
        self.noise_model = engine.noise_model
        self.blocks = getattr(self.noise_model, "blocks", None)
        self.supports_standard_denoising = bool(
            config.get("enable_standard_denoising", False)
        )
        self.uses_native_components = bool(config.get("use_native_components", True))

    @staticmethod
    def _resolve_existing_path(
        component_model_path: str, path_value: str | None
    ) -> str:
        if not path_value:
            raise ValueError("WanS2VOfficialEngine config is missing a required path")
        path_value = os.path.expanduser(path_value)
        if os.path.isabs(path_value):
            resolved = path_value
        else:
            resolved = os.path.join(component_model_path, path_value)
        if not os.path.exists(resolved):
            raise ValueError(f"Resolved path does not exist: {resolved}")
        return resolved

    @staticmethod
    def _sdpa_flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        del window_size, deterministic, version
        if q_scale is not None:
            q = q * q_scale
        if softmax_scale is not None:
            q = q * softmax_scale

        if q_lens is not None or k_lens is not None:
            if not WanS2VOfficialEngine._sdpa_warned_padding_mask:
                logger.warning(
                    "Wan S2V SDPA fallback ignores q_lens/k_lens padding masks; "
                    "this is intended only for compatibility validation."
                )
                WanS2VOfficialEngine._sdpa_warned_padding_mask = True

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
        )
        return out.transpose(1, 2).contiguous()

    @classmethod
    def _compatible_flash_attention(
        cls,
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        del dropout_p
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        assert q.device.type == "cuda" and q.size(-1) <= 256

        b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        if q_lens is None:
            q = half(q.flatten(0, 1))
            q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
        else:
            q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

        if k_lens is None:
            k = half(k.flatten(0, 1))
            v = half(v.flatten(0, 1))
            k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
        else:
            k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
            v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)
        if q_scale is not None:
            q = q * q_scale

        module_attention = importlib.import_module("wan.modules.attention")
        fa3_func = getattr(
            importlib.import_module("flash_attn_interface"),
            "flash_attn_varlen_func",
            None,
        )
        fa2_func = getattr(
            importlib.import_module("flash_attn"), "flash_attn_varlen_func", None
        )

        use_fa3 = (
            (version is None or version == 3)
            and getattr(module_attention, "FLASH_ATTN_3_AVAILABLE", False)
            and fa3_func is not None
        )
        if use_fa3:
            x = fa3_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
                window_size=window_size,
            )
            if isinstance(x, tuple):
                x = x[0]
            return x.unflatten(0, (b, lq)).type(out_dtype)

        if fa2_func is None:
            raise RuntimeError(
                "No compatible flash attention varlen kernel is available for Wan S2V"
            )
        x = fa2_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        return x.unflatten(0, (b, lq)).type(out_dtype)

    @classmethod
    def _patch_attention_backend(cls, config: dict[str, Any]):
        backend = str(config.get("attention_backend", "auto")).lower()
        if backend not in {"auto", "flash", "sdpa"}:
            raise ValueError(f"Unsupported Wan attention backend: {backend}")

        module_attention = importlib.import_module("wan.modules.attention")
        if backend == "flash":
            module_attention.flash_attention = cls._compatible_flash_attention
            module_model = importlib.import_module("wan.modules.model")
            module_model.flash_attention = cls._compatible_flash_attention
            for module_name in (
                "wan.modules.s2v.model_s2v",
                "wan.modules.s2v.motioner",
                "wan.distributed.ulysses",
            ):
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    continue
                setattr(module, "flash_attention", cls._compatible_flash_attention)
            logger.info("Using official Wan flash attention backend (compat)")
            return

        if backend == "auto":
            has_fa2 = getattr(
                importlib.import_module("flash_attn"), "flash_attn_varlen_func", None
            )
            has_fa3 = getattr(module_attention, "FLASH_ATTN_3_AVAILABLE", False)
            has_fa3_func = getattr(
                importlib.import_module("flash_attn_interface"),
                "flash_attn_varlen_func",
                None,
            )
            if has_fa2 or has_fa3_func or not has_fa3:
                module_attention.flash_attention = cls._compatible_flash_attention
                module_model = importlib.import_module("wan.modules.model")
                module_model.flash_attention = cls._compatible_flash_attention
                for module_name in (
                    "wan.modules.s2v.model_s2v",
                    "wan.modules.s2v.motioner",
                    "wan.distributed.ulysses",
                ):
                    try:
                        module = importlib.import_module(module_name)
                    except ModuleNotFoundError:
                        continue
                    setattr(module, "flash_attention", cls._compatible_flash_attention)
                logger.info(
                    "Using official Wan flash attention backend (auto compat)"
                )
                return
            backend = "sdpa"

        logger.info("Patching official Wan attention backend to SDPA fallback")
        module_attention.flash_attention = cls._sdpa_flash_attention
        module_attention.FLASH_ATTN_2_AVAILABLE = False
        module_attention.FLASH_ATTN_3_AVAILABLE = False

        module_model = importlib.import_module("wan.modules.model")
        module_model.flash_attention = cls._sdpa_flash_attention
        for module_name in (
            "wan.modules.s2v.model_s2v",
            "wan.modules.s2v.motioner",
            "wan.distributed.ulysses",
        ):
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            setattr(module, "flash_attention", cls._sdpa_flash_attention)

    @classmethod
    def _resolve_attention_backend(cls, server_args, config: dict[str, Any]) -> str:
        backend = getattr(server_args, "attention_backend", None)
        if backend is None:
            return str(config.get("attention_backend", "auto")).lower()
        backend = str(backend).lower()
        if backend in {"torch_sdpa", "sdpa"}:
            return "sdpa"
        if backend in {"fa", "flash", "flashattention", "flash_attention"}:
            return "flash"
        return backend

    @classmethod
    def from_component_path(
        cls, component_model_path: str, server_args, config: dict[str, Any]
    ):
        code_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_code_root", "official_code")
        )
        checkpoint_root = cls._resolve_existing_path(
            component_model_path, config.get("wan_checkpoint_root", "checkpoints")
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        task_name = str(config.get("wan_task_name", "s2v-14B"))
        module_configs = importlib.import_module("wan.configs")
        module_s2v = importlib.import_module("wan.speech2video")
        module_unipc = importlib.import_module("wan.utils.fm_solvers_unipc")
        config = dict(config)
        config["attention_backend"] = cls._resolve_attention_backend(
            server_args, config
        )
        cls._patch_attention_backend(config)

        wan_configs = getattr(module_configs, "WAN_CONFIGS", None)
        if not isinstance(wan_configs, dict) or task_name not in wan_configs:
            raise ValueError(
                f"Official Wan config {task_name!r} not found under {code_root}"
            )

        local_device = get_local_torch_device()
        device_id = 0 if local_device.index is None else int(local_device.index)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        use_sp = (
            max(
                getattr(server_args, "sp_degree", 1) or 1,
                getattr(server_args, "ulysses_degree", 1) or 1,
            )
            > 1
        )

        config_obj = wan_configs[task_name]
        engine = module_s2v.WanS2V(
            config=config_obj,
            checkpoint_dir=checkpoint_root,
            device_id=device_id,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=use_sp,
            t5_cpu=bool(config.get("t5_cpu", False)),
            init_on_cpu=bool(config.get("init_on_cpu", True)),
            convert_model_dtype=bool(config.get("convert_model_dtype", False)),
        )
        instance = cls(
            engine=engine,
            component_model_path=component_model_path,
            config=config,
            scheduler_cls=getattr(module_unipc, "FlowUniPCMultistepScheduler", None),
        )
        if instance.supports_standard_denoising and instance.uses_native_components:
            if hasattr(engine, "text_encoder"):
                engine.text_encoder = None
            if hasattr(engine, "audio_encoder"):
                engine.audio_encoder = None
            if hasattr(engine, "vae"):
                engine.vae = None
        return instance

    @property
    def device(self):
        return getattr(self.engine, "device", torch.device("cpu"))

    def create_standard_scheduler(self):
        if self.scheduler_cls is None:
            return None
        return self.scheduler_cls(
            num_train_timesteps=self.engine.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )

    def get_default_negative_prompt(self) -> str:
        return self.engine.sample_neg_prompt

    def get_generation_size(self, *, image_path: str) -> tuple[int, int]:
        return self.engine.get_gen_size(
            size=None,
            max_area=int(self.config.get("max_area", 720 * 1280)),
            ref_image_path=image_path,
            pre_video_path=None,
        )

    def _normalize_infer_frames(self, num_frames: int) -> int:
        infer_frames = max(int(num_frames) - 1, 4)
        if infer_frames % 4 != 0:
            infer_frames = max((infer_frames // 4) * 4, 4)
        return infer_frames

    def _get_negative_prompt(self, negative_prompt: str | None) -> str:
        if negative_prompt is None or negative_prompt == "":
            return self.engine.sample_neg_prompt
        return negative_prompt

    def _encode_prompts(
        self,
        prompt: str,
        negative_prompt: str | None,
        *,
        offload_model: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        negative_prompt = self._get_negative_prompt(negative_prompt)
        if not self.engine.t5_cpu:
            self.engine.text_encoder.model.to(self.device)
            context = self.engine.text_encoder([prompt], self.device)
            context_null = self.engine.text_encoder([negative_prompt], self.device)
            if offload_model:
                self.engine.text_encoder.model.cpu()
        else:
            context = self.engine.text_encoder([prompt], torch.device("cpu"))
            context_null = self.engine.text_encoder(
                [negative_prompt], torch.device("cpu")
            )
            context = [tensor.to(self.device) for tensor in context]
            context_null = [tensor.to(self.device) for tensor in context_null]
        return context[0:1], context_null[0:1]

    def _load_audio_for_mux(
        self, audio_path: str
    ) -> tuple[torch.Tensor | None, int | None]:
        if sf is None:
            return None, None
        try:
            audio_np, sample_rate = sf.read(
                audio_path, dtype="float32", always_2d=False
            )
        except Exception as exc:  # pragma: no cover - best effort only
            logger.warning("Failed to load source audio for output muxing: %s", exc)
            return None, None
        if audio_np is None:
            return None, None
        return torch.from_numpy(audio_np), int(sample_rate)

    def prepare_standard_s2v_inputs(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        image_path: str,
        audio_path: str,
        pose_video_path: str | None,
        num_clip: int | None,
        num_frames: int,
        seed: int,
    ) -> dict[str, Any]:
        del seed
        if num_clip not in (None, 1):
            raise NotImplementedError(
                "Native Wan S2V denoising currently supports only a single clip"
            )

        infer_frames = self._normalize_infer_frames(num_frames)
        height, width = self.engine.get_gen_size(
            size=None,
            max_area=int(self.config.get("max_area", 720 * 1280)),
            ref_image_path=image_path,
            pre_video_path=None,
        )
        resize = transforms.Resize(min(height, width))
        crop = transforms.CenterCrop((height, width))
        to_tensor = transforms.ToTensor()

        model_pic = crop(resize(Image.open(image_path).convert("RGB")))
        ref_pixel_values = to_tensor(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel_values = ref_pixel_values.to(
            dtype=self.engine.vae.dtype, device=self.engine.vae.device
        )
        ref_latents = torch.stack(self.engine.vae.encode(ref_pixel_values))

        motion_frames = self.engine.motion_frames
        motion_pixels = torch.zeros(
            [1, 3, motion_frames, height, width],
            dtype=self.engine.param_dtype,
            device=self.device,
        )
        motion_latents = torch.stack(self.engine.vae.encode(motion_pixels))
        lat_motion_frames = (motion_frames + 3) // 4

        audio_emb, max_num_repeat = self.engine.encode_audio(
            audio_path, infer_frames=infer_frames
        )
        if max_num_repeat < 1:
            raise ValueError(f"Audio path produced no valid clips: {audio_path}")

        cond_list = self.engine.load_pose_cond(
            pose_video=pose_video_path,
            num_repeat=1,
            infer_frames=infer_frames,
            size=(height, width),
        )
        cond_latents = cond_list[0].to(
            dtype=self.engine.param_dtype, device=self.device
        )
        if pose_video_path is None:
            cond_latents = cond_latents * 0

        context, context_null = self._encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            offload_model=bool(self.config.get("offload_model", True)),
        )

        lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames
        latent_shape = (1, 16, lat_target_frames, height // 8, width // 8)
        audio_tensor, audio_sample_rate = self._load_audio_for_mux(audio_path)

        return {
            "height": int(height),
            "width": int(width),
            "infer_frames": int(infer_frames),
            "latent_shape": latent_shape,
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
            "ref_latents": ref_latents,
            "motion_latents": motion_latents,
            "cond_states": cond_latents,
            "audio_input": audio_emb[..., :infer_frames],
            "motion_frames": [motion_frames, lat_motion_frames],
            "drop_motion_frames": bool(self.engine.drop_first_motion),
            "audio": audio_tensor,
            "audio_sample_rate": audio_sample_rate,
        }

    def prepare_standard_s2v_latents(
        self,
        *,
        latent_shape: tuple[int, ...],
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        return randn_tensor(
            latent_shape,
            generator=generator,
            device=self.device,
            dtype=self.engine.param_dtype,
        )

    def decode_standard_s2v_output(
        self,
        *,
        latents: torch.Tensor,
        ref_latents: torch.Tensor,
        motion_latents: torch.Tensor,
        infer_frames: int,
        drop_motion_frames: bool,
    ) -> torch.Tensor:
        if drop_motion_frames:
            decode_latents = torch.cat([ref_latents, latents], dim=2)
        else:
            decode_latents = torch.cat([motion_latents, latents], dim=2)
        image = torch.stack(self.engine.vae.decode(decode_latents))
        image = image[:, :, -infer_frames:]
        if drop_motion_frames:
            image = image[:, :, 3:]
        return image

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        ref_latents: torch.Tensor | None = None,
        motion_latents: torch.Tensor | None = None,
        cond_states: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        motion_frames: list[int] | tuple[int, int] | None = None,
        drop_motion_frames: bool = False,
        add_last_motion: bool | int | None = None,
        seq_len: int | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        del guidance
        if timestep is None and "t" in kwargs:
            timestep = kwargs.pop("t")
        if timestep is None:
            raise ValueError("Wan S2V forward requires timestep")
        if encoder_hidden_states is None:
            raise ValueError("Wan S2V forward requires encoder_hidden_states")
        if isinstance(encoder_hidden_states, list):
            if len(encoder_hidden_states) == 0:
                raise ValueError("encoder_hidden_states list cannot be empty")
            context = encoder_hidden_states
        elif isinstance(encoder_hidden_states, torch.Tensor):
            if encoder_hidden_states.ndim == 3:
                context = [
                    encoder_hidden_states[i]
                    for i in range(encoder_hidden_states.shape[0])
                ]
            else:
                context = [encoder_hidden_states]
        else:
            raise TypeError(
                "Wan S2V encoder_hidden_states must be a tensor or list of tensors"
            )
        if seq_len is None:
            seq_len = int(
                hidden_states.shape[2]
                * hidden_states.shape[3]
                * hidden_states.shape[4]
                // 4
            )
        if ref_latents is None or motion_latents is None or cond_states is None:
            raise ValueError(
                "Wan S2V forward requires ref_latents, motion_latents, and cond_states"
            )
        if motion_frames is None:
            motion_frames = [
                self.engine.motion_frames,
                (self.engine.motion_frames + 3) // 4,
            ]
        if add_last_motion is None:
            add_last_motion = 2
        output = self.noise_model(
            hidden_states,
            t=timestep,
            context=context,
            seq_len=seq_len,
            ref_latents=ref_latents,
            motion_latents=motion_latents,
            cond_states=cond_states,
            audio_input=audio_input,
            motion_frames=motion_frames,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
            **kwargs,
        )
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise ValueError(
                    f"Wan S2V noise model returned unexpected output length: {len(output)}"
                )
            return output[0]
        return output

    def generate(
        self,
        *,
        prompt: str,
        image_path: str,
        audio_path: str,
        pose_video_path: str | None,
        num_clip: int | None,
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ):
        infer_frames = self._normalize_infer_frames(num_frames)
        logger.info(
            "Calling official WanS2V engine with infer_frames=%d, num_clip=%s",
            infer_frames,
            num_clip,
        )
        return self.engine.generate(
            input_prompt=prompt,
            ref_image_path=image_path,
            audio_path=audio_path,
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=num_clip,
            pose_video=pose_video_path,
            max_area=int(self.config.get("max_area", 720 * 1280)),
            infer_frames=infer_frames,
            shift=float(self.config.get("sample_shift", 3.0)),
            sample_solver=str(self.config.get("sample_solver", "unipc")),
            sampling_steps=num_inference_steps,
            guide_scale=guidance_scale,
            n_prompt="",
            seed=seed,
            offload_model=bool(self.config.get("offload_model", True)),
            init_first_frame=False,
        )


EntryClass = WanS2VOfficialEngine
