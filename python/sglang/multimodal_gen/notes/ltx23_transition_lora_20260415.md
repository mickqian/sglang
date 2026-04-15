## 2026-04-15

- 基线 git hash: `d2f479e54`
- 进展: 已把 LTX2/LTX-2.3 TI2V 的 condition image 从“单前缀注入”改成支持 `1` 图或 `2` 图 `[first_frame, last_frame]`；双图在 SP 下按本 rank 的本地首/末帧 span 覆盖，不再错误地把双图当成一段统一 shard。
- 进展: `LTX2RefinementStage` 在 stage2 预噪声前会按 full-res 重新准备 condition latent，避免沿用 stage1 half-res latent。
- 进展: OpenAI `/v1/videos` 已开始保留多 reference，multipart/JSON 都会落到 `image_path` 列表；对 LTX2 TI2V 增加了 `>2` references 的明确报错。
- 当前精度对齐判断: `70%`。语义路径已对齐到计划，但还没跑 UT/接口测试，也还没做真实 case smoke。

## 2026-04-15（第二次更新）

- 基线 git hash: `d2f479e54`
- 进展: OpenAI `/v1/videos` 现在支持 `input_reference/reference_url` 的单值或列表；multipart 会保留表单里的 reference 顺序，JSON 会把 `input_reference + reference_url` 统一归一化到 `SamplingParams.image_path`。
- 进展: `LTX2RefinementStage` 的 TI2V 预噪声路径会先按 stage2 full-res 重新准备 condition latent，再做边界帧覆盖；避免复用 stage1 half-res condition latent。
- 验证: `py_compile` 已通过；`git diff --check` 已通过。
- 验证失败点: 尝试补纯 UT，但本机 `pytest` 在 import `sglang` 时被 optional quantization / triton 依赖链拖住，和本次功能改动无关。UT 文件已撤掉，避免把环境 hack 带入分支。
- 当前精度对齐判断: `78%`。代码路径、语法、静态 diff 已稳定；还缺真实生成 smoke 或 API e2e 才能接近 `90%+`。

## 2026-04-15（第三次更新）

- 基线 git hash: `917d14c57496d5509319b241234ad948464370b4`
- 进展: 已把 `LoRA adapter ... does not contain the weights for layer ...` 的逐层 warning 改成按 adapter 聚合摘要；正常 partial coverage 现在只打一条 summary，包含 `missing/applied/total` 和少量 example layers。
- 进展: 仅当某个 adapter `applied_count == 0` 时保留 `warning` 级别；其余 partial coverage 降成 `info`，避免 `LTX-2.3` 社区 LoRA 把日志刷爆。
- 验证: `py_compile` 已通过；`git diff --check` 已通过。
- 当前精度对齐判断: `82%`。日志行为已经和之前 smoke 分支的经验对齐；还没在这条分支上复跑真实生成确认新摘要日志输出。

## 2026-04-15（第四次更新）

- 基线 git hash: `ec902788ebe49b664ed3e4e1fd2ca8b147106a88`
- 进展: 已在远端 `sglang_mick` 容器上用官方 native one-stage pipeline 跑通 `valiantcat/LTX-2.3-Transition-LORA` 的 model card `output2` prompt，输出 `/tmp/ltx23_native_runs/official_one_stage_33f.mp4`；同配置的 `SGLang` 输出为 `/tmp/ltx23_native_runs/sglang_one_stage_33f.mp4`。
- 验证: 两边视频元信息完全一致，都是 `768x512 / 33 frames / 24 fps / 1.375s`。
- 验证: 逐帧比较结果为 `mean_mae=3.3099`、`mean_psnr=33.1304 dB`；sample frames 的 MAE 落在 `2.48 ~ 3.98`，说明 `SGLang` one-stage 与官方 native one-stage 已经比较接近。
- 验证: 新的 LoRA 聚合日志已在远端生成路径确认生效，`Transition-LORA` 会输出一条 `missing 1084/1660, applied 576` 的摘要，而不是逐层 warning。
- 当前精度对齐判断: `92%`。one-stage 的 T2V 路径和官方 native 已经对齐到可接受范围；还需要补 two-stage 对拍，确认 stage2 路径也没有偏移。

## 2026-04-15（第五次更新）

- 基线 git hash: `ec902788ebe49b664ed3e4e1fd2ca8b147106a88`
- 进展: 已在同一台 H200 机器上补完 `LTX2TwoStagePipeline` 对拍；官方 native 输出为 `/tmp/ltx23_native_runs/official_two_stage_33f.mp4`，`SGLang` 输出为 `/tmp/ltx23_native_runs/sglang_two_stage_33f.mp4`。
- 验证: 两边视频元信息同样完全一致，都是 `768x512 / 33 frames / 24 fps / 1.375s`。
- 验证: two-stage 逐帧比较结果为 `mean_mae=5.5512`、`mean_psnr=28.3202 dB`；比 one-stage 松一些，但仍然明显处于“同一 pipeline 语义、内容接近”的区间，不像是 stage2 路径错接或 LoRA 没有生效。
- 验证: `SGLang` two-stage 日志确认走到了 `LTX2UpsampleStage -> LTX2LoRASwitchStage -> LTX2RefinementStage -> LTX2AVDecodingStage`，并且 stage2 切换后仍然只输出一条 `Transition-LORA` 的 missing-layer 摘要。
- 当前精度对齐判断: `95%`。`valiantcat/LTX-2.3-Transition-LORA` 在 native one-stage / two-stage 上都已跑通，`SGLang` 与官方 native 的 T2V 结果基本对齐；若还要继续追，就该下钻到 `prompt embeds / one-step noise_pred / stage2 refine inputs` 的中间张量级对拍。

## 2026-04-15（第六次更新）

- 基线 git hash: `ea149b2105feb0ab8c1db18a967ad36470c3c21d`
- 进展: 已确认 `spongebob` case 就是 `valiantcat/LTX-2.3-Transition-LORA` model card 的 `output3`；按同一 prompt 跑通了 `10.041667s / 241 frames / 24 fps` 的官方 native one-stage 与 `SGLang` one-stage，对应输出分别为 `/tmp/ltx23_native_runs/official_one_stage_241f_spongebob.mp4` 和 `/tmp/ltx23_native_runs/sglang_one_stage_241f_spongebob.mp4`。
- 验证: 两边长视频元信息仍然完全一致，都是 `768x512 / 241 frames / 24 fps / 10.041667s`。
- 验证: 逐帧比较结果为 `mean_mae=11.2966`、`mean_psnr=22.1348 dB`；明显比 `33` 帧短视频 case 漂移更大，说明长时长下 `SGLang` 和官方 native 的采样轨迹会进一步分叉，但仍保持同一 prompt / LoRA / 输出规格。
- 观察: `SGLang` one-stage 这条长视频的 denoise 平均步长约 `3.78s/step`，总像素生成时间约 `127.56s`；官方 native 单步约 `4.24s/step`，生成时间略慢。
- 当前精度对齐判断: `90%`。短视频 one-stage / two-stage 已较稳；长视频 `241f` one-stage 仍可用，但和官方 native 的像素级一致性明显下降，后续若要继续追，需要看 `prompt embeds / sigma schedule / one-step noise_pred` 是否已经在长序列开始处出现偏移。

## 2026-04-15（第七次更新）

- 基线 git hash: `895fde5ae`
- 进展: 已针对 `spongebob 241f` 长视频 case 下钻中间张量，对拍官方 native one-stage 与 `SGLang` one-stage 的 `prompt_embeds / latent_0 / latent_1 / full trajectory`。
- 发现: prompt embeds 本身已经很接近，`video prompt` 的 `MAE` 约 `0.00124`；修正前 `latent_0 / latent_1` 也只在 `1e-3` 量级偏差，但 `SGLang` trajectory 从 step 10 之后开始持续放大，到 step 29 时 `trajectory_video` 的单步 `MAE` 已到 `0.2855`。
- 根因: `SGLang` native LTX-2.3 one-stage 之前把 latent 固定成 `fp32`，而官方 native one-stage 实际沿用 pipeline 的 `bf16` latent / noise dtype。长视频下这类微小数值差异会沿 denoise 轨迹逐步累积。
- 修复: `LTX2AVLatentPreparationStage._get_latent_dtype()` 已改为对 native LTX-2.3 也走 `pipeline_config.get_latent_dtype(prompt_dtype)`，不再强制 one-stage `fp32`。
- 验证: 修正后 `audio_latent_0 / video_latent_0` 已与官方完全一致；`video_latent_1` 的 `MAE` 降到 `0.000108`。`trajectory_video` 整体 `MAE` 从 `0.05866` 降到 `0.03810`，step 29 的单步 `MAE` 从 `0.2855` 降到 `0.1958`。
- 验证: 最终成片对拍也同步改善。`official_one_stage_241f_spongebob.mp4` 对 `sglang_one_stage_241f_spongebob_after_bf16.mp4` 的逐帧比较结果为 `mean_mae=7.5984`、`mean_psnr=25.5048 dB`；相比修正前的 `11.2966 / 22.1348 dB` 有明显提升。
- 当前精度对齐判断: `93%`。`LTX-2.3 Transition LoRA` 的 native one-stage 长视频精度已经实质收敛；剩余漂移更像是 scheduler / denoise 内部数值路径上的细部差异，而不是 LoRA、prompt、或 condition 语义没对上。

## 2026-04-15（第八次更新）

- 基线 git hash: `83c2561cc`
- 进展: 已把 `spongebob 241f` 的 first-step 四个 guider pass (`cond / uncond / ptb / mod`) 都抓出来，分别对拍官方 native 与 `SGLang` native。
- 发现: 不带 LoRA 时，`SGLang` 的 first-step `video_x0_*` 仍然和官方有稳定差距，`MAE` 大约在 `0.00396 ~ 0.00471`；`trajectory_video` 整体 `MAE=0.04216`。这说明剩余误差不是 LoRA 专属问题，base model forward 本身就还有一层偏差。
- 发现: 带 `Transition-LORA` 时，first-step `video_x0_cond_0` 的 `MAE` 会从 no-LoRA 的 `0.00443` 增加到 `0.00603`，`video_x0_mod_0` 也会从 `0.00471` 增加到 `0.00732`。LoRA 会额外放大一部分误差，但不是主要来源。
- 发现: 将官方导出的 `prompt_embeds / negative_prompt_embeds / audio_prompt_embeds` 直接覆盖到 `SGLang` probe 后，no-LoRA 的 `trajectory_video` 会从 `0.04216` 降到 `0.03755`，同时 audio 侧 first-step `x0` 误差也会进一步下降。这说明 text path 的小偏差会被长视频 denoise 轨迹持续放大。
- 试验失败点: 我尝试把 `LTX2TextConnectorStage` 的 additive mask 改成官方同款的 `4D + -finfo(dtype).max`，但数值结果完全不变；这条改动已验证无效，并已回滚，不保留在当前分支。
- 当前精度对齐判断: `93.5%`。目前已能确定剩余误差主要分成两块：`(1) text path` 的小偏差；`(2) native DiT/base model forward` 的固有偏差。下一步更值得下钻的是官方 `Gemma -> feature extractor / connector input` 和 `SGLang` 对应阶段，而不是继续猜 scheduler 或 attention mask。

## 2026-04-15（第九次更新）

- 基线 git hash: `47d685dfa`
- 进展: 我补了 `text-only probe`，把官方 native 和 `SGLang` 的 text path 按相同中间层对拍到 `input_ids / attention_mask / packed_features / aggregate features / final prompt embeds`。
- 发现: `input_ids` 和原始 `attention_mask` 是完全一致的；`prompt_packed_features` 的 `MAE` 只有 `0.000321`，`negative_prompt_packed_features` 也只有 `0.000331`。这说明 tokenizer 和 `FeatureExtractorV2` 的 RMSNorm+flatten 语义基本没跑偏。
- 发现: 误差在 `aggregate_embed` 后会被放大，`prompt_video_features MAE=0.003878`、`prompt_audio_features MAE=0.002863`；到 final connector 输出时又回落到 `prompt_embeds_video MAE=0.001241`、`prompt_embeds_audio MAE=0.001387`。
- 发现: 把官方导出的 `packed_features` 直接覆盖给 `SGLang` connector 后，final prompt embed 还能进一步缩小到 `prompt_embeds_video MAE=0.000662`、`prompt_embeds_audio MAE=0.001065`。这说明 text path 里的偏差大约一半来自 `Gemma hidden_states -> packed_features` 之前，剩下一半来自 `aggregate_embed / connector` 路径的细小数值差异。
- 发现: 如果直接看整条 `1024` token 序列，`hidden_state_1` 会显得非常离谱（`MAE` 约 `1.5`），但这是被 left padding token 放大的假象。按 `attention_mask` 只看有效 token 后，`prompt_hidden_state_1 MAE` 只有 `0.00116`、`negative_prompt_hidden_state_1 MAE` 只有 `0.00125`；到第 `8` 层和 final norm (`48`) 才逐层积累到 `0.0247 ~ 0.0280`。
- 结论: 这轮可以明确排除 tokenizer 问题；`hidden_state_0` 完全一致，说明 embedding 层没问题。真正的 text path 漂移是自研 `Gemma3` attention/decoder 路径里对有效 token 的小偏差逐层累积；而整序列上看见的巨大 hidden-state diff，大部分只是 padding query 的噪音，不是主语义 token 已经炸掉。
- 当前精度对齐判断: `93.8%`。text path 的主要矛盾已经从“可能哪里都不对”收敛到“自研 `Gemma3` 有效 token attention 数值路径上的小偏差”；下一步更值得直接核对 `Gemma3Attention` 的 valid-token mask / sliding layer 语义，而不是继续在 tokenizer 或 connector 上兜圈子。

## 2026-04-15（第十次更新）

- 基线 git hash: `3f92e2b26`
- 试验: 我尝试把自研 `Gemma3Attention` 从 `torch.scaled_dot_product_attention(..., enable_gqa=True)` 改成更接近 HF `eager_attention_forward` 的实现（显式 `repeat_kv + fp32 softmax + matmul`），想验证是不是 attention kernel 语义本身导致了 text path 漂移。
- 验证: `text-only probe` 结果没有变好，反而略差。`prompt_embeds_video MAE` 从 `0.001241` 变成 `0.001204`（几乎不变），但 `prompt_embeds_audio MAE` 从 `0.001387` 变成 `0.001378`（也基本不变）；更关键的是 `prompt_packed_features / prompt_video_features / prompt_audio_features` 全部没有改善，`video_features MAE` 还从 `0.003878` 升到 `0.004494`。
- 结论: “只把 text encoder attention kernel 从 SDPA 改成 eager-like 实现” 不是主要矛盾，至少不是一个能直接带来可见收益的最小 patch。这条改动已回滚，不保留在当前分支。
- 当前精度对齐判断: `93.8%`。目前最可信的判断仍然是：有效 token 的小偏差主要来自自研 `Gemma3` attention 路径里更细的 mask / RoPE / layer pattern 语义，而不是单纯 `SDPA vs eager` 的 softmax 路径。

## 2026-04-15（第十一次更新）

- 基线 git hash: `9af0931fc`
- 修复: `Gemma3` text encoder 的 sliding causal window 语义已纠正：不再把 `sliding_window` 错减 `1`，也不再错误地屏蔽“太远的未来 token”；现在会按 HF 语义屏蔽过旧的 past tokens。
- 修复: `Gemma3` text encoder 的 SDPA mask 已从手工 `4D additive float mask` 改成与 HF 一致的 `4D bool allow-mask`。这让整条 `1024` token 序列上的 `hidden_state` 噪音显著下降，例如 `prompt_hidden_state_1/8/48 MAE` 从 `1.543 / 6.010 / 3.855` 降到 `0.000269 / 0.005728 / 0.006488`。
- 结论: 这两条修复都是真 bug，但对当前 `spongebob` case 的主语义路径帮助有限。原因很明确：当前 prompt 的有效长度只有 `237`（negative 是 `250`），远低于 `sliding_window=1024`；而 `pack_text_embeds_v2` 会在 flatten 后把 padding token 全部清零，所以 padding-query 噪音不会继续泄漏到 `packed_features`。
- 验证: 修复后 `prompt_packed_features / prompt_embeds_video / prompt_embeds_audio` 仍然分别停在 `0.000321 / 0.001241 / 0.001387`，和修复前几乎完全一致；说明当前 text path 剩余误差依旧集中在有效 token 本身，而不是 padding 或 sliding。
- 验证: 我顺手核了现成的 backend A/B。`sglang_spongebob_241f_nolora.pt`（`fa`）对官方的 `trajectory_video MAE=0.04216`，而 `sglang_spongebob_241f_nolora_sdpa.pt`（`torch_sdpa`）是 `0.04280`，略差；仍然是“覆盖官方 prompt embeds”那条最好，`trajectory_video MAE=0.03755`。
- 结论: 目前可以排除“把 DiT attention backend 从 `fa` 切到 `torch_sdpa` 就会更准”这条思路；下一步更值得追的是 `(1) Gemma 有效 token path` 的细粒度语义，或 `(2) base DiT forward` 本身的数值差异。
- 当前精度对齐判断: `93.8%`。这轮把两个真实语义 bug 修掉了，也明确排除了 `torch_sdpa` backend 这条分支；但 `LTX-2.3 Transition LoRA` 的主误差来源仍未转移。
