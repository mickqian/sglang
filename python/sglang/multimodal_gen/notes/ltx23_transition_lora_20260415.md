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
