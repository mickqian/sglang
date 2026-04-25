# 2026-04-24 LTX-2.3 HQ 精度对齐

## 14:31 起点

- base: `origin/main` = `cd1fa7506`
- 新分支: `ltx23-hq-precision-align`
- 目标: 轻量参数对齐 case 先到 `>25 dB`，长期 `>35 dB`
- 当前先修一个明确 official 差异：HQ `res2s` 遇到末尾 `sigma=0` 时，官方实际跑 `... -> 0.0011` 的 RK step，再做一次 final denoise；native stage2 只扩了 sigmas 没扩 timesteps，stage1 也没追加 final `0.0011`。
- 修改范围: 只 gate 到 `LTX2TwoStageHQPipeline`，不影响 LTX2.3 one-stage / standard two-stage / TI2V 默认路径。
- 对齐百分比: 未复跑，沿用历史 canonical baseline `20.71 dB` 作为修复前参考。

## 14:50 lightweight 反证

- branch commit: `2b244f72e`
- case: `384x256`, `49 frames`, `4 steps`, `fps=24`, seed `10`, prompt `SpongeBob talking with Patrick`
- official HQ: `/tmp/ltx23_hq_align_light_2b244f/official_light.mp4`, wall `56.29s`
- `origin/main` SGLang: `15.91 dB`, pixel `60.44s`
- `2b244f72e` SGLang: `15.49 dB`, pixel `63.45s`
- 结论: final-denoise/timestep 补丁是回退，已 revert 为 `9ab34d5ca`。下一步改为 activation 边界定位，不再继续沿这个方向猜。
- 对齐百分比: 约 `15.91 / 35 = 45.5%`（lightweight case, 35 dB 目标）。

## 15:46 activation 边界

- multi-call dump: official `/tmp/ltx23_hq_actdump8_official/dump`，native `/tmp/ltx23_hq_actdump8_sglang/dump`。
- call0-2 是 step0 的 cond/neg/modality 当前点；video/audio latent 和 timestep bit-exact，patchify bit-exact。
- branch forward output 只有 `53-60 dB`；经过 CFG/modality guidance 后 guided x0 只有 video `36.48 dB`、audio `39.87 dB`。
- call3 是 step0 midpoint，不是下一步；midpoint actual video `57.86 dB`，主要由 guided x0 误差传播，不是 stage2 才开始。
- 排除项: output LayerNorm autocast 语义无改善；LoRA `(B*scale)@A` vs `(B@A)*scale` 无变化。
- 下一步: 继续查 branch model output 内部的低 50dB 来源，优先 output projection / LoRA fused weight / cache-dit wrapper 对 final projection 的影响；当前对齐百分比仍按最终视频约 `15.9 / 35 = 45.5%`。

## 16:00 output projection / attention 排除

- `proj_out/audio_proj_out` fused weight/bias official vs native bit-exact。
- `adaln_single` / `prompt_adaln_single` output bit-exact；`norm_out` input/output 仍高（video `83.0/82.8 dB`）。
- `scale_shift + proj_out` 把 video 从 `82.8 -> 67.4 -> 53.3 dB`；这是 tiny block activation diff 被 output modulation/projection 放大，不是权重或 AdaLN 参数错。
- 强制 `--attention-backend torch_sdpa`，以及临时打开 cuDNN SDP，都没有改善；video guided x0 约 `35.9 dB`，比 baseline `36.5 dB` 更差。
- 下一步: 不再沿 attention backend 试；改为 block 内子层二分（attention residual / FF residual / cross-modal residual），找首个从 `>90 dB` 掉到 `~83 dB` 的子路径。

## 16:46 block0 / LoRA RowParallel bias

- git: `9ab34d5ca` + 本地未提交 `RowParallelLinearWithLoRA` tp=1 fused-bias 修复。
- block0 子层 dump: `attn1/audio_attn1` 输入、Q/K/V、Q/K RMSNorm、权重全部 bit-exact。
- 修复前: `audio_attn1.to_out.0` 输入/weight/bias bit-exact，但 native 输出 `77.29 dB`；CPU `F.linear(input, weight, bias)` 匹配 official，说明差异来自 `RowParallelLinearWithLoRA` 先 GEMM 后单独 bf16 加 bias。
- 修复: tp=1 且非 `skip_bias_add` 时把 bias 传入 `quant_method.apply`，避免 RowParallel LoRA 路径和 official `nn.Linear` 的 fused-bias 语义分叉；TP>1 仍保留原来的 all-reduce 后单次加 bias。
- 结果: lightweight final video `15.91 -> 17.09 dB`；`audio_attn1.to_out.0` 恢复 bit-exact，`forward.output` video `53.31 -> 55.18 dB`。
- 反证: `.contiguous()` 无改善（`15.55 dB`）；`--attention-backend torch_sdpa` / inline official-shaped SDPA 都是 `16.30 dB`，低于默认 FA + row-bias。
- 剩余首个差异: video self-attn raw output，Q/K/V exact 后 attention result `77.03 dB`，默认 FA 仍比 torch SDPA 的最终画面更接近。
- 对齐百分比: `17.09 / 35 = 48.8%`（lightweight case）。

## 17:00 SDPA 分支定位

- `--attention-backend torch_sdpa` + row-bias 修复：最终视频 `16.30 dB`，低于默认 FA + row-bias 的 `17.09 dB`。
- 但 block0 self-attn 在 SDPA 下是 bit-exact，block0 output video `100.8 dB` / audio `84.4 dB`；默认 FA 的 block0 video 只有 `85.3 dB`。
- milestone blocks（SDPA）: video `block1 95.3`, `block8 87.3`, `block16 85.5`, `block32 89.6`, `block47 83.2 dB`；audio `block1 82.7`, `block8 78.5`, `block16 79.5`, `block32 73.0`, `block47 74.1 dB`。
- 结论: SDPA 可以对齐早期 self-attn，但不能解决最终画面；误差更像多处分支的小量漂移被 output projection/guidance 放大。默认 FA 当前反而是最终视频最优。
- 当前最佳: default attention + row-bias，`17.09 dB`；baseline/current origin-main video 对同一 official 为 `15.55 dB`。

## 17:12 判断准则

- 后续改动分两类记录：`semantic fix` 表示更接近官方实现，即使当前最终视频 PSNR 不涨也保留；`metric fix` 表示直接提升当前对拍 case。
- 不再用“单个 case PSNR 没涨”作为回滚正确语义修复的充分理由；回滚只针对确认偏离官方或影响其他 variant 语义的改动。

## 17:28 block0 SDPA 首因

- SDPA + row-bias 下，forward 输入 video/audio latent 和 timestep bit-exact；block0 self-attn/video audio self-attn 都 bit-exact。
- 首个非 exact 边界是 block0 prompt cross-attn output：video `86.38 dB`，audio `87.43 dB`；随后 V2A output `75.38 dB`，audio FF output `77.0 dB`。
- 结论: 继续查 `prompt_cross_attn` 的 KV/context 侧，比查 stage2/refinement 更有价值；这个方向属于 semantic alignment，即使最终 PSNR 短期不涨也应保留正确修复。
- 当前最终最佳仍是 default attention + row-bias：`17.09 dB`，对齐百分比 `48.8%`。

## 17:36 prompt KV 侧确认

- block0 prompt cross-attn Q 侧 bit-exact；KV 侧输入已经漂移：video `attn2.to_k/to_v.input 75.76 dB`，audio `audio_attn2.to_k/to_v.input 71.26 dB`，权重 bit-exact。
- 这说明首个漂移来自 text encoder / connector 输出，而不是 DiT prompt attention 的 q/k/v 权重或 SDPA。
- 试过 TP=1 Gemma3 separate-QKV（模仿 HF q/k/v 三个独立 linear），prompt KV、proj_out、最终视频完全不变；已撤掉，避免保留纯 kernel-shape 实验。

## 18:28 HQ connector sequential 语义

- text dump: official `/tmp/ltx23_hq_text_official2calls/dump`，native SDPA `/tmp/ltx23_hq_text_sglang_sdpa/dump`。
- official `EmbeddingsProcessor` 是 sequential text connector：call0/call1 分别对应 native 合批 connector 的 batch1/batch0；native 原逻辑在 CFG 时把 neg/pos 合批过 connector。
- Gemma/aggregate 不是主因：对齐正确 batch 后，`video_aggregate_embed.output` 约 `90-102 dB`，`audio_aggregate_embed.output` 约 `84-98 dB`。
- connector transformer 单次调用仍有残差：sequential 后 `video_connector.output` 约 `76 dB`，`audio_connector.output` 约 `69-73 dB`；这解释了 block0 prompt KV 侧的 `~71-76 dB`。
- semantic fix: 只对 `LTX2TwoStageHQPipeline` 的 CFG text connector 改为 official 顺序，先 positive 后 negative，其他 LTX/LTX2.3 pipeline 仍保持原合批行为。
- metric: lightweight case `384x256/49f/4steps/seed10`，row-bias + default attention `17.09 -> 17.26 dB`；row-bias + `torch_sdpa` `16.30 -> 17.75 dB`。
- 当前最佳仍未到 25 dB，下一步查单次 connector 内部差异（官方 `Embeddings1DConnector` vs native `LTX2ConnectorTransformer1d` 的 attention/norm/MLP 子层）。对齐百分比按默认 attention `17.26 / 35 = 49.3%`，按 SDPA debug path `17.75 / 35 = 50.7%`。

## 21:10 HQ res2s SDE dtype

- git: `9fcd1c542` + 本地修复。
- injected-text + SDPA activation 对拍显示：DiT call0-2 raw output bit-exact 后，native call3 midpoint 仍只有 video `67.58 dB` / audio `66.71 dB`。
- 复算 official res2s substep 发现：官方 `_get_new_noise` 生成 `float64` noise 并直接参与 double SDE；native HQ 路径把 noise cast 到 latent dtype/bf16 后再 `.float()`，导致 first midpoint drift。
- 修复: 只在 `ctx.use_native_hq_res2s_sde_noise` 路径保留 `float64` noise；非 HQ fallback 仍使用原 batch generator `.float()` 语义。
- 结果: injected-text run 的 call3、call4、call5 输入恢复 bit-exact；证明 stage1 first midpoint construction 已对齐。

## 21:40 official reference hygiene

- 发现 `/tmp/ltx23_hq_actdump8_official` 这类 activation hook official 不能作为最终视频 reference：clean official `/tmp/ltx23_hq_clean_official_current/official.mp4` 与 actdump official `/tmp/ltx23_hq_actdump8_official_current/official.mp4` 只有 `16.74 dB`。
- clean official 和 step-dump official bit-exact，说明 `Res2sDiffusionStep.step` hook 不扰动输出；扰动来自 LTXModel activation dump hook。
- 后续视频指标只用无 hook official；activation dump 只用于局部边界定位。
- 另一个 semantic fix: official substep SDE 用 double `sigma/sub_sigma`，但 full-step SDE 直接传完整 `sigmas`，保持 `float32`；native HQ final SDE 已改为只在 HQ 路径使用 float32 step sigma，避免影响其他 LTX variants。
- 最新无注入 native SDPA: `/tmp/ltx23_hq_current_native_sdpa/native.mp4` vs clean official `17.10 dB`，对齐百分比 `17.10 / 35 = 48.9%`。

## 22:15 non-perturbing step dump

- git: `12160462c`。
- official safe step dump: `/tmp/ltx23_hq_stepdump_official/step_dump`，native safe step dump: `/tmp/ltx23_hq_native_stepdump_sdpa/step_dump`。
- 验证: native stepdump 视频与 no-hook native bit-exact；official stepdump 视频与 clean official bit-exact。这个 hook 可作为 coarse boundary reference。
- step0 substep video/audio: `sample` exact、`noise` exact；`denoised_sample` 已经只有 video `61.66 dB`、audio `64.44 dB`，输出相同量级。
- 结论: 当前主误差在 sampler 前的 first guided x0/denoiser 输出，不在 initial latent/noise/res2s state update。下一步用 non-perturbing guider hook 对齐 cond/neg/modality x0，而不是再用 LTXModel activation hook 当最终视频 reference。

## 22:55 text injection / block output 复核

- git: `12160462c`。
- valid combined official-text injection + safe step dump: `/tmp/ltx23_hq_inject_text_stepdump_sdpa_combined`。结果没有改善：guided x0 normal vs official 为 video `39.27 dB` / audio `37.24 dB`，注入 official text 后为 video `38.76 dB` / audio `36.36 dB`；最终视频从 clean official 对比为 `16.83 dB`，低于当前 no-hook native SDPA `17.10 dB`。
- 结论修正: text/connector 有语义差异，sequential connector 仍是正确 semantic fix；但它不是当前 clean-reference 主瓶颈，不能继续把主误差归因到 text。
- RoPE coords dtype 实验: 临时强制 native LTX2.3 video RoPE coords 转 bf16，最终视频仍为 `17.1006 dB`，与当前 SDPA no-hook 完全一致；排除这个方向。
- block dump 复核: call0 block0/1/8/16/32/47 output 仍很高（video 多数 `82-88 dB`，audio `73-78 dB`），但 final `forward.output` 只有 video `54.44 dB` / audio `55.63 dB`，再经 CFG/modality guider 放大到 `~39 dB`。
- 当前判断: 主误差更像 DiT 全链路 bf16 小漂移在 output projection / x0 conversion / guider 处被放大，而不是某个早期 block 或 text 输入突然错位。对齐百分比仍按 clean official `17.10 / 35 = 48.9%`。

## 23:28 HQ final res2s denoise

- git: `a04d2eb11`。
- 发现 official HQ guider dump 有 `18` 个保存点，native 只有 `14` 个；原因是 official `res2s_audio_video_denoising_loop` 会把末尾 `0` 改成 `0.0011, 0`，最后一个主步仍走 RK2，再额外做 final denoise。native stage1 之前直接在 `sigma_next == 0` 处退化成一次 denoise；stage2 refinement 也 append 了 `0.0011, 0` 但没有更新 `num_steps/timesteps`，实际没跑 final denoise。
- 修复范围: 只对 `LTX2TwoStageHQPipeline`，stage1 sigma list 追加 `0.0011`；stage2 refinement 更新 `num_steps` 和 `timesteps` 为扩展后的 schedule。其他 LTX2 / LTX2.3 one-stage / 普通 two-stage 不改。
- metric: synced remote `a04d2eb11`，lightweight SDPA case `/tmp/ltx23_hq_current_a04d2eb_sdpa/native.mp4` vs clean official `/tmp/ltx23_hq_clean_official_current/official.mp4`，`17.1006 -> 17.4913 dB`，`+0.3908 dB`。
- 对齐百分比: `17.49 / 35 = 50.0%`。这是正向 semantic fix，但剩余主误差仍在 first guided x0 / DiT branch output。

## 23:46 guided-x0 注入定位

- git: `a04d2eb11`。
- reference hygiene: `/tmp/ltx23_hq_clean_official_current/official.mp4` 与旧 guider dump 所属 `/tmp/ltx23_hq_guider_official/official.mp4` 只有 `16.7411 dB`，不能混用。
- 同一轮 guider official 对拍: 普通 native `/tmp/ltx23_hq_current_a04d2eb_sdpa/native.mp4` 为 `17.9244 dB`；注入 18 个 official guided-x0 后 `/tmp/ltx23_hq_current_a04d2eb_inject_guided_x0/native.mp4` 为 `26.1998 dB`。
- 结论: stage1 guided-x0 之后的 upsample/stage2/decode 链路已超过短期 25 dB；主误差仍在 stage1 guided output 之前。早期 video guided calls 2/4/6 约 `22 dB`，优先查这些 call 的 DiT branch 输入/输出。
- 对齐百分比: diagnostic injected `26.20 / 35 = 74.9%`；真实 native 仍按 clean official `17.49 / 35 = 50.0%`。

## 00:18 stage1 video branch 定位

- git: `a04d2eb11`。
- 选择性注入同一轮 guider official: video-only `25.1774 dB`，audio-only `17.5818 dB`，只注入 video calls `2,4,6` 为 `22.7036 dB`；单独 call2/call4/call6 分别 `18.7677 / 18.7434 / 19.3550 dB`。
- 结论: 画面主误差来自 stage1 video guided outputs，且是连续 substep drift，不是单个 call bug；call6 单点贡献最大，但三者叠加才明显。
- official 源码复核: `GuidedDenoiser` 先构造 3-way passes，但默认 `max_batch_size=1` 的 `BatchSplitAdapter` 会拆成 sequential forward；当前 native HQ sequential pass 和官方默认路径一致。
- output-path dump: call0 `norm_out.pre` 很低，但 `norm_out.out ~48 dB`、`proj_out.pre ~53 dB`、`proj_out.out ~43 dB`；final projection 是误差放大器而不是公式错。临时 fp32 output proj 最终指标完全不变。
- 当前真实 native 对齐百分比仍为 `17.49 / 35 = 50.0%`；stage1-guided 注入上限为 `26.20 / 35 = 74.9%`。

## 00:58 final head / x0 path 复核

- git: `a04d2eb11`。
- 重新跑 block dump: official/native `block0/1/8/16/32/47` 前 3 个 stage1 passes 输出全部很高；最差是 audio late block 约 `74 dB`，video block47 约 `84 dB`，排除“某个 block 内语义明显错位”。
- x0 path dump: 首个 video pass `latent` bit-exact，`velocity 54.60 dB`（SDPA 下约 `56.42 dB`），`x0 43.74 dB`；x0 的 MAE 与 velocity 基本相同，PSNR 下降主要是 x0 动态范围较小，不是 denoise-mask/sigma/x0 conversion 公式错。
- output-path dump（SDPA native vs official）: call0 video `norm_out.pre 84.00 dB`，`norm_out.out 83.38 dB`，modulation 后 `proj_out.pre 68.92 dB`，最终 `proj_out.out 56.42 dB`；后续 video passes 快速降到 `41/30/26 dB` 并被 guider/sampler 放大。
- final AdaLN dump: `adaln_single/audio_adaln_single` 的 `temb` 和 `embedded_timestep` bit-exact；`scale_shift_table` 约 `85.7 dB`，`audio_scale_shift_table` 约 `81.0 dB`，不是主语义差异。
- output projection weight/bias（含 LoRA merge 后）此前已验证 bit-exact；因此当前主误差更像 bf16 DiT 小漂移经 final modulation/projection 与 CFG/modality guidance 放大，而不是 text、sampler、x0 conversion、final AdaLN 或 output weight layout 的离散 bug。
- clean official vs current SDPA native 仍为 `17.4913 dB`，真实对齐百分比 `17.49 / 35 = 50.0%`；guided-x0 注入上限仍为 `26.20 / 35 = 74.9%`。

## 01:10 projection kernel 排除

- git: `a04d2eb11`。
- official venv 中 `xformers/flash_attn` 都不可用，official default attention 实际走 PyTorch SDPA；native 加 `--attention-backend torch_sdpa` 后确认当前 clean-reference 指标仍是 `17.4913 dB`。
- 用 dump 的 official `proj_out.pre.000` + bit-exact official/native output projection weight/bias，在同一 GPU 上单独跑 `F.linear`，可以 bit-exact 复现 official `proj_out.out.000`（PSNR `999`）。因此 final projection 的 kernel dispatch 不是独立误差源。
- 更准确的结论: final head 是误差放大器；主残差来自进入 final head 前已存在的 DiT hidden-state 累计漂移（SDPA 下 call0 `norm_out.pre ~84 dB`，后续 stage1 passes 降到 `~73/62/57 dB`），再经 modulation/projection/guidance/sampler 放大。
- 下一步如果继续追 25 dB，优先做 stage1 `proj_out.pre` 注入验证或按 block interval 做 hidden-state 注入，定位需要提升的是哪一段 block 的累计漂移；不要再查 x0 conversion、AdaLN timestep、output weight layout。

## 01:28 block interval / recurrence 结论

- git: `a04d2eb11`。
- block dump 扩到 stage1 前 `12` 个 passes（`block0/1/8/16/32/47`，native 强制 SDPA）。文件数 official/native 都是 `144`。
- pass0-2（first sigma 的 cond/neg/mod）仍很高: video `block47` 约 `83-84 dB`；说明首轮 DiT 本身没有明显结构性错位。
- pass6-8 开始，进入 `block0` 前的 video hidden 已只有约 `76 dB`，到 `block47` 约 `60-62 dB`；pass9-11 更明显，`block0` 约 `68 dB`，`block1` 约 `58 dB`，`block47` 约 `56-57 dB`。
- 结论: later-pass 主误差不是某个固定 block 突然坏，而是 stage1 recurrence 中上一轮 guided update/latent state 已把误差带进下一轮 block0；继续查单个 block 内 weight/layout 的收益下降。若要继续突破，最高价值是验证“inject official stage1 guided-x0/proj_out.pre 后 recurrence 是否恢复”，然后只在能产生真实 semantic fix 时改代码。
- 真实 clean-reference 仍是 `17.4913 dB`，对齐百分比 `50.0%`。

## 01:39 native prompt cross-attn 顺序复核

- git: `5af97e9f6`。
- 按 official `transformer.py` 的 block order，把 LTX2.3 `cross_attention_adaln` 路径的 video prompt cross-attn 移到 video self-attn 之后、audio self-attn 之前；普通 non-cross 路径不改，避免影响其他 LTX2 variant。
- native CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_current_5af97e_prompt_ca_order_sdpa/native.mp4` vs `/tmp/ltx23_hq_clean_official_current/official.mp4`，`mean_psnr=17.4913 dB`，和 `a04d2eb11` 持平。
- 结论: 这是更贴近 official 源码的 semantic cleanup，但不是当前 PSNR 瓶颈；后续停止 injected run，改为只做 native source alignment + plain `sglang generate` 验证。
- 对齐百分比: `17.49 / 35 = 50.0%`。

## 01:54 connector RoPE dtype 回退

- git: `44896dc19` 尝试把 native connector 1D RoPE 输出 dtype 改成 `hidden_states.dtype`，表面更接近 official `Embeddings1DConnector`。
- plain native CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_current_44896dc_connector_rope_dtype_sdpa_retry/native.mp4` vs clean official，`mean_psnr=16.5527 dB`，相对 `17.4913 dB` 倒退 `0.9387 dB`。
- 处理: 已 revert 为 `62cd9b1f4`。该点不能作为 HQ 对齐修复保留；继续只接受真实 CLI PSNR 不倒退的 native 改动。
- 对齐百分比回到 `17.49 / 35 = 50.0%`。

## 17:22 clean guider 注入复核

- git: `a04d2eb11`。
- 先修正 reference hygiene: 旧 `/tmp/ltx23_hq_guider_official/official.mp4` 与 clean official 只有 `16.7411 dB`，不能作为 clean-reference 注入证据。重新用同一个 official 脚本、同参数、只 hook `MultiModalGuider.calculate` 导出 `/tmp/ltx23_hq_clean_guider_official/guider_dump`；新 official 视频与 `/tmp/ltx23_hq_clean_official_current/official.mp4` bit-exact。
- clean dump 注入结果（native CLI, SDPA, 384x256/49f/4steps/seed10）:
  - baseline native: `17.4913 dB`。
  - all guided calls `0-17`: `24.7167 dB`。
  - stage1 early calls `0-7`: `21.6685 dB`。
  - stage1 video-only `0,2,4,6`: `22.8052 dB`。
  - stage1 audio-only `1,3,5,7`: `17.8376 dB`。
  - all video calls `0,2,4,6,8,10,12,14,16`: `25.8521 dB`。
  - late calls `8-17`: `24.7167 dB`（与 all `0-17` 完全同分）。
  - late video calls `8,10,12,14,16`: `23.9348 dB`。
- 结论更新: 初期 25 dB 已由 diagnostic `video_all_even` 打穿。主误差确实在 stage1 guided output/recurrence 之前，但不是 audio 分支；最强信号是 video guided outputs，尤其后半 calls `8-16` 与早期 video calls 组合后达到 `25.85 dB`。audio dump 直接注入会破坏 native 当前 video/audio 耦合，不能作为修复方向。
- 下一步最高价值: 比较 video branch guided-x0 的三个输入分支 `cond/uncond_text/uncond_modality` 在 calls `8,10,12,14,16` 的误差和公式贡献，确认是 DiT video branch 输出本身漂移，还是 guider branch 选择/scale/rescale 语义仍有差异。真实 native 对齐仍为 `17.49 / 35 = 50.0%`；diagnostic 上限更新为 `25.85 / 35 = 73.9%`。

## 02:05 停止 injection，回到 native source alignment

- git: `85d2a3ce3`。
- 用户明确要求停止 injection 实验；后续只接受 native 源码对齐 + plain `sglang generate` 端到端验证。
- 复核 decoder generator/noise: official HQ 确实把 `generator` 传给 `video_decoder`，但当前 materialized `vae/config.json` 中 `video_decoder_config.timestep_conditioning=false`，该路径不会实际加噪；不改。
- 尝试 `d4b667956` 将 LTX2.3 decoder `res_x_y` shortcut norm 改为 official `GroupNorm(1)`，plain CLI 指标仍 `17.491346 dB`；随后发现当前 `video_decoder_config.decoder_blocks` 无 `res_x_y`，该改动是死分支，已 revert。
- 当前真实 native 指标仍为 `17.49 / 35 = 50.0%`。下一步只查会被当前 config 命中的 native 结构差异，优先 transformer/video guided path，而不是 decoder 死路径。

## 02:32 HQ autocast 语义复核

- git: `d6fec42b8`。
- 源码差异: sglang base denoising loop 默认包 `torch.autocast(bfloat16)`；official HQ DiT forward 没有全局 autocast，只是模块/输入为 bf16。已在 `LTX2TwoStageHQPipeline` 的 denoising context 内 HQ-only 关闭外层 autocast，不影响其他 LTX variant。
- plain CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_current_d6fec42_autocast_off/native.mp4` vs `/tmp/ltx23_hq_clean_official_current/official.mp4`，`mean_psnr=17.491346 dB`，与 `85d2a3ce3` 持平。
- 结论: 这是 source-level semantic alignment，但当前数值不受外层 autocast 影响；主瓶颈仍在 stage1 video guided recurrence 的 DiT 累计漂移。
- 对齐百分比: `17.49 / 35 = 50.0%`。

## 02:50 HQ tokenwise x0 复核

- git: `2be299ff6`。
- 源码差异: official `X0Model` 使用 tokenwise `Modality.timesteps` 做 velocity->x0；native HQ 在 no-mask 情况曾使用 scalar sigma。已改为仅 `LTX2TwoStageHQPipeline` 的 stage1 guided path 使用 `model_inputs_local.timestep_video/audio`。
- plain CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_current_2be299_token_x0/native.mp4` vs clean official `/tmp/ltx23_hq_clean_official_current/official.mp4`，`mean_psnr=17.491346 dB`，与 `d6fec42b8` 持平。
- 结论: 这是 source-level semantic alignment，但不是当前主误差来源；后续继续 native 源码对齐，不再做 guided/activation injection 实验。
- 对齐百分比: `17.49 / 35 = 50.0%`。

## 03:02 HQ res2s coefficient 复核

- git: `1536f09b4`。
- 源码差异: official `get_res2s_coefficients` 使用 Python `math` float 计算 `h/a21/b1/b2`；native 原来用 CUDA tensor 版本。已改为仅 HQ res2s 路径使用 official float 公式，普通 LTX 路径保持原 tensor 公式。
- plain CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_current_1536f09_coeff/native.mp4` vs clean official `/tmp/ltx23_hq_clean_official_current/official.mp4`，`mean_psnr=17.491346 dB`，完全持平。
- 结论: res2s coefficient 不是当前瓶颈；当前剩余误差仍表现为 stage1 video guided recurrence 中的 DiT bf16 累计漂移。
- 对齐百分比: `17.49 / 35 = 50.0%`。

## 19:03 HQ stage2 token timestep 复核

- git: `f55d69619`。
- 用户要求停止 injection；后续只做 native source alignment + plain `sglang generate` 端到端验证。
- 源码差异: official stage2 `SimpleDenoiser` 仍通过 `X0Model` 用 `Modality.timesteps` 做 velocity->x0；native stage2 res2s helper 之前用 scalar sigma。已改为仅 HQ timestep 语义路径使用 token timestep。
- plain CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_current_f55d696/native.mp4` vs clean official `/tmp/ltx23_hq_clean_official_current/official.mp4`，`mean_psnr=17.503858 dB`，相对 `1536f09b4` 提升 `+0.0125 dB`。
- 结论: 这是有效但极小的语义补齐，不是主误差；下一步继续查当前配置实际命中的 native transformer/video guided path 差异。对齐百分比按 35 dB 目标仍约 `50.0%`。

## 19:48 native-only 10s 历史点复核

- git: `023fb34d3`；用户再次强调不要 injection，后续只走 native source alignment + `sglang generate`。
- clean official reference: `/tmp/ltx23_official_10s_v2/video.mp4`；prompt 为 PR 23366 SpongeBob 10s case，`768x512/241f/fps24/15steps/seed10`。
- b479 default native rerun: `/tmp/ltx23_b479fc09f_pr23366_10s_cli_default/native_b479fc09f_pr23366_10s_default.mp4`，`global_psnr=14.5070 dB`。
- current HEAD default native: `/tmp/ltx23_023fb34_pr23366_10s_cli_default/native_023fb34_pr23366_10s_default.mp4`，`global_psnr=14.3712 dB`，相对 b479 只低 `0.1358 dB`。
- b479/current 的 `torch_sdpa` native 都约 `14.0 dB`；因此旧 notes 里的 `20.71 dB` 历史点在当前 clean official reference + 同一比较脚本下不可复现，不能作为继续 bisect 目标。
- 下一步: 不再追 injection 或 hook 视频；只对照 official HQ 源码查尚未实现的 native 语义差异，并用 plain CLI 复核。当前 10s 对齐百分比按 `14.37 / 35 = 41.1%`，lightweight case 仍约 `50.0%`。

## 04:20 HQ text length native 对齐

- git: `184d532e9`。
- 用户再次要求不要任何 injection；本轮只做 native 源码配置对齐和 plain `sglang generate` 验证。
- 源码差异: official `LTXVGemmaTokenizer` 默认 `max_length=256`，native `Gemma3ArchConfig.text_len=1024`。已新增 `LTX23HQPipelineConfig`，仅 HQ pipeline 使用 256，避免影响现有 LTX2.3 one-stage/two-stage CI baseline。
- 10s SpongeBob CLI 复核: `/tmp/ltx23_hq_text256_pr23366_10s_cli_default/native_hq_text256_pr23366_10s_default.mp4` vs `/tmp/ltx23_official_10s_v2/video.mp4`，`global_psnr=14.371213949670121`，与 `023fb34d3/e5ad4e9c6` 完全持平。
- 结论: 该 prompt 在 256 token 内或此前实际有效文本等价；改动作为 source-level semantic alignment 保留，但不是当前 PSNR 瓶颈。下一步继续查 native transformer/video guided path 的实际源码差异，不再使用 guided/activation injection。
- 当前 10s 对齐百分比: `14.37 / 35 = 41.1%`。

## 04:45 LTX distilled LoRA fusion native 复核

- git: `2559666fc` 尝试仅对 LTX distilled LoRA merge 使用 official loader 的 fusion 顺序（`B * strength @ A`，delta cast 后加 base weight），不引入 injection。
- 10s SpongeBob CLI 复核: `/tmp/ltx23_hq_official_lora_fuse_pr23366_10s_cli_default/native_hq_official_lora_fuse_pr23366_10s_default.mp4` vs `/tmp/ltx23_official_10s_v2/video.mp4`，`global_psnr=14.371213949670121`，和 `184d532e9` 完全持平；pixel generate time `87.43s`。
- 结论: LoRA fusion 顺序不是当前主误差来源。该改动增加额外 LTX path 分支但没有可观测收益，已按“keep native、少分支”原则 revert 为 `8967406a1`。
- 当前 10s 对齐百分比仍为 `14.37 / 35 = 41.1%`。

## 05:18 native-only 历史高点复跑结论

- git: `2ad0bdb26`。
- 源码差异: official HQ 用 `torch.Generator(device=self.device)`；native 新增 HQ-only `generator_device=None` 默认配置，但 CLI 仍由 sampling params 继承 cuda generator 语义，因此输出未变化。
- current 10s SpongeBob CLI: `/tmp/ltx23_hq_device_generator_pr23366_10s_cli_default/native_hq_device_generator_pr23366_10s_default.mp4` vs `/tmp/ltx23_official_10s_v2/video.mp4`，`global_psnr=14.371213949670121`，pixel generate time `157.77s`。
- 复跑所谓历史高点 `b479fc09f`: `/tmp/ltx23_b479_recheck_pr23366_10s_cli_default/native_b479_recheck_pr23366_10s_default.mp4`，同一 official reference 下 `global_psnr=14.50700982030883`，pixel generate time `99.83s`。
- 结论: 旧 collab log 中 `20.71 dB` 在当前 clean official reference、同一 prompt/settings、同一比较脚本下不可复现；后续不再围绕 injection 或旧高点追溯，只做 native source alignment + plain `sglang generate` 端到端复核。
- 当前 10s 对齐百分比: `14.37 / 35 = 41.1%`；b479 复跑也只有 `14.51 / 35 = 41.4%`。

## 05:32 10s 参数对齐和 text length 修正

- git: `2ad0bdb26`。
- 发现参数不一致: 当前 official reference 脚本 `/tmp/run_official_10s_v2.py` 固定 `NUM_INFERENCE_STEPS=30`，而此前 native CLI 未显式传步数，HQ sampling default 走 15。用同一 current commit 加 `--num-inference-steps 30` 复跑，`global_psnr=16.931129448461213`，相对 15-step `14.3712` 提升 `+2.56 dB`；输出 `/tmp/ltx23_current_30steps_pr23366_10s_cli/native_current_30steps_pr23366_10s.mp4`，pixel time `165.53s`。
- 发现真实语义 bug: official `/tmp/LTX-2-official` loader 使用 `LTXVGemmaTokenizer(tokenizer_root, 1024)`；SpongeBob prompt token 数为 `467`，native HQ 的 `text_len=256` 会截断正向 prompt。已准备把 HQ `Gemma3ArchConfig(text_len=1024)` 恢复为 official 语义。
- 当前 30-step 对齐百分比: `16.93 / 35 = 48.4%`。下一步复跑 1024 text length 后再更新指标。

## 09:34 真实 PYTHONPATH 复跑和 connector RoPE 负例

- git: `4fcfa964e`。
- 发现远端 `sglang generate` 如果不显式设置 `PYTHONPATH=$PWD/python:$PYTHONPATH`，会 import `/sgl-workspace/sglang` 的旧安装包，而不是 `/tmp/ltx23-hq-precision-align-run`。后续远端 CLI 复核必须带该环境变量。
- 用真实当前分支复跑 `737318096`（HQ text_len=1024、30 steps）: `/tmp/ltx23_text1024_30steps_localpy_pr23366_10s_cli/native_text1024_30steps_localpy_pr23366_10s.mp4` vs `/tmp/ltx23_official_10s_v2/video.mp4`，`global_psnr=16.931129448461213`，pixel time `160.03s`，与 stale import 结果相同。结论: 1024 text length 是正确语义，但不是该 case 的主 PSNR 瓶颈。
- 检查 materialized connector config: `rope_type=split`、`rope_double_precision=true`，和 LTX-2.3 checkpoint/donor config 一致；不是 interleaved/float32 配置错误。
- 尝试 `397b7ea40` 将 connector RoPE/apply 改为 hidden dtype 计算以贴近表面 official 源码，plain CLI 降到 `global_psnr=15.4571684842985`，方向错误；已 revert 为 `4fcfa964e`，不保留该负向改动。
- 当前 10s 对齐百分比仍为 `16.93 / 35 = 48.4%`。下一步继续查当前配置实际命中的 DiT/prompt cross-attn/native guided path 差异。

## 09:58 native torch_sdpa 后端复核

- git: `a032a7f22`。
- 用户要求不要 injection；本轮只用 plain `sglang generate`，在 10s SpongeBob case 上把 native `--attention-backend` 改为 `torch_sdpa`。
- 结果: `/tmp/ltx23_hq_sdpa_30steps_pr23366_10s_cli/native_hq_sdpa_30steps_pr23366_10s.mp4` vs `/tmp/ltx23_official_10s_v2/video.mp4`，`global_psnr=16.198498824294433`，pixel time `107.82s`。
- 对比默认 backend 的 `16.931129448461213`，`torch_sdpa` 低 `0.73 dB`；attention backend/kernel dispatch 可能贡献漂移，但不是“切 SDPA 即收敛”的主修复路径。
- 当前 `torch_sdpa` 10s 对齐百分比: `16.20 / 35 = 46.3%`；默认 backend 仍是 `48.4%`。

## 10:37 HQ res2s SDE NaN fallback 复核

- git: `8294790ff`。
- 源码差异: official `Res2sDiffusionStep.get_sde_coeff` 对 NaN 的 fallback 是 `sigma_up -> 0`、`sigma_down -> sigma_next`、`alpha_ratio -> 1`；native 原来统一 `nan_to_num(..., nan=0)`。已按 official 语义对齐，改动局限在 LTX res2s helper。
- plain CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_sde_nan_8294790_30steps_pr23366_10s_cli/native_hq_sde_nan_8294790_30steps_pr23366_10s.mp4` vs `/tmp/ltx23_official_10s_v2/video.mp4`，`global_psnr=16.931129448461213`，`mean_psnr=17.086487352484422`，pixel time `145.56s`。
- 结论: 这是 source-level semantic alignment，但当前 SpongeBob 10s case 完全持平；SDE NaN fallback 不是主误差。下一步继续查 HQ stage1 guided pass 与 official `GuidedDenoiser + BatchSplitAdapter` 的 native 语义差异。
- 当前 10s 对齐百分比: `16.93 / 35 = 48.4%`。

## 10:50 HQ video RoPE coords dtype 复核

- git: `65ec2e2c3`。
- 源码差异: official `VideoLatentTools.create_initial_state` 会把 video positions cast 到 latent dtype（HQ 为 bf16）后进入 RoPE；native 默认保持 video coords fp32。已仅在 `LTX2TwoStageHQPipeline` forward context 中临时启用 `quantize_video_rope_coords_to_hidden_dtype`，不影响其它 LTX pipeline。
- plain CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_rope_coord_dtype_65ec2e2_30steps_pr23366_10s_cli/native_hq_rope_coord_dtype_65ec2e2_30steps_pr23366_10s.mp4` vs `/tmp/ltx23_official_10s_v2/video.mp4`，`global_psnr=16.931129448461213`，`mean_psnr=17.086487352484422`，pixel time `120.60s`。
- 结论: 这是 source-level semantic alignment，但当前 case 完全持平；video RoPE coords dtype 不是主误差。下一步继续查 transformer 内部 kernel/layout 差异，尤其 native parallel linear/attention wrapper 与 official `torch.nn.Linear`/official attention path 的差异。
- 当前 10s 对齐百分比: `16.93 / 35 = 48.4%`。

## 11:05 HQ DiT RoPE frequency precision 复核

- git: `36050355e`。
- 源码差异: official `LTXModelConfigurator` 会把 `frequencies_precision == "float64"` 解释为 `double_precision_rope=True`；native 之前只读 `rope_double_precision`。已让 native DiT 在缺少 `rope_double_precision` 时 honor `frequencies_precision`，这是更贴近 official config 的语义。
- plain CLI 复核（无 hook/无 injection）: `/tmp/ltx23_hq_rope_float64_3605035_30steps_pr23366_10s_cli/native_hq_rope_float64_3605035_30steps_pr23366_10s.mp4` vs `/tmp/ltx23_official_10s_v2/video.mp4`，`global_psnr=16.931129448461213`，`mean_psnr=17.086487352484422`，pixel time `146.13s`。
- 结论: 当前 materialized config/arch 路径上该修复没有改变 SpongeBob 10s 输出；保留作为 official 语义修复，但不是主误差。下一步继续查 current config 实际命中的 DiT forward 分支，重点是 prompt cross-attn / AdaLN / attention mask 与 official `transformer.py` 的细微差异。
- 当前 10s 对齐百分比: `16.93 / 35 = 48.4%`。

## 11:42 official reference / attention path 复核

- git: `77924ea69`，当前分支 `ltx23-hq-precision-align`，worktree clean。
- reference hygiene: `/tmp/ltx23_official_10s_v2/video.mp4` 来自 dirty `/tmp/LTX-2-official`；clean 59ca official `/tmp/ltx23_official_10s_clean59ca/video.mp4` 与它只有 `16.8585 dB`。后续区分 old reference 和 clean reference，不能把两者混用做 bisect。
- current native 30-step SpongeBob vs old reference: `global_psnr=16.9311 dB`；vs clean 59ca reference: `global_psnr=16.2406 dB`。旧 notes 的 `20.71 dB` 历史点在当前 prompt/settings/reference 下不可复现，不再作为恢复目标。
- materialized HQ config 已确认: connector `rope_type=split`、`rope_double_precision=true`、`connector_rope_base_seq_len=4096`；transformer `attention_type=default`、`force_sdpa_v2a_cross_attention=true`、`cross_attention_adaln=true`。connector RoPE base 和 type 与 checkpoint/official config 等价，不是当前主误差。
- official `AttentionFunction.DEFAULT` 源码实际是 `xformers else PyTorch SDPA`，不是自动 FA3；安装了 `flash_attn_interface` 不代表 default 使用 FA3。native default 仍会按平台优先 FA，`--attention-backend torch_sdpa` 是更接近 official 的 debug path，但此前 10s old-reference 指标低 `0.73 dB`，不能直接作为 metric fix。
- source audit: HQ guided stage1 在 official 通过 `BatchSplitAdapter(max_batch_size=1)` sequential 跑 cond/neg/modality；native HQ 已有 per-pass sequential 分支。prompt cross-attn AdaLN、AV CA gate timestep factor、res2s `0.0011 -> 0` tail、initial packed noise shape 当前源码都与 official 语义一致。
- 下一步: 不再做 guided-x0 injection 扩展；优先用 plain CLI 或低扰动 activation 对拍验证 `force_sdpa_v2a_cross_attention`/native attention wrapper 是否是剩余可改的 source-level 差异。当前 10s old-reference 对齐百分比: `16.93 / 35 = 48.4%`。

## 12:21 attention backend clean-reference 排序

- git: `fbe7638c3`。
- plain CLI 实验: `--attention-backend fa` 会让 native 默认路径里原本被 `force_sdpa_v2a_cross_attention=true` 限制的 V2A 也走 FA；生成 `/tmp/ltx23_hq_fbe7638_fa_30steps_pr23366_10s_cli/native_fbe7638_fa_30steps_pr23366_10s.mp4`，pixel time `166.83s`。
- 10s SpongeBob old reference 指标: default `16.9311 dB`，FA-only `16.4231 dB`，此前 torch_sdpa `16.1985 dB`。old reference 下 default 最优，但该 reference 来自 dirty official，不能作为唯一依据。
- 10s SpongeBob clean 59ca reference 指标: torch_sdpa `16.8763 dB`，default `16.2406 dB`，FA-only `16.1768 dB`。clean official 下 torch_sdpa 最接近 official；FA-only 没有改善，`force_sdpa_v2a_cross_attention` 不是负向主因。
- 结论: precision 对拍应固定 clean official reference，并用 `--attention-backend torch_sdpa` 作为 official-like debug path；默认/FA 路径保留给性能，不能因为单 case PSNR 较低就强制改默认 backend。
- 当前 clean-reference debug 对齐百分比: `16.88 / 35 = 48.2%`。

## 12:40 history output 复查和 stage2 步数判断

- git: `8e3993f23`。
- 复查历史视频 `/tmp/ltx23_3e8g6_pr23366_10s_cli/native_3e8g6_pr23366_10s.mp4`: vs old reference `17.1952 dB`，vs clean 59ca reference `17.3760 dB`，vs current default output `15.9490 dB`。collab log 里的 `20.71 dB` 仍不可复现。
- 轻量 recheck: `b479fc09f` detached worktree `/tmp/ltx23-b479-run`，`384x256/49f/4steps/SpongeBob talking with Patrick/torch_sdpa` 输出 `/tmp/ltx23_b479_light_sdpa_recheck/native_b479_light_sdpa.mp4`，pixel time `85.67s`。
- 同一参数 current HEAD 输出 `/tmp/ltx23_head_light_sdpa_recheck/native_head_light_sdpa.mp4`，pixel time `97.04s`；两者互比 `15.9609 dB`，说明 b479 到 current 的代码确实改变了输出。但原 lightweight official reference 已从 `/tmp` 清理，暂时不能判断哪一个更接近 clean official。
- 观察到 b479 stage2 tqdm 为 `3`，current 为 `4`。源码复核后暂不改: official `res2s_audio_video_denoising_loop` 的 final denoise 不在 tqdm 内；native 当前把 `0.0011 -> 0` final denoise 作为最后一个 loop step 执行，输出语义应等价。不能仅凭 tqdm 计数回滚 stage2 final denoise。
- 下一步若继续追 history delta，应重新生成 clean official lightweight reference，再按 commit 做 native CLI bisect；不要只对齐 b479 输出本身。当前可靠 clean 10s debug 指标仍是 `16.88 / 35 = 48.2%`。
