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
