## 2026-03-31 LTX2 Two-Stage Alignment

### Trusted Provenance

- 当前只认这一份代码：
  - repo: `/Users/mick/repos/sglang`
  - branch: `codex/ltx2-from-862f-clean-main`
  - head: `0b0bd41385c0b190d05686ccc8b8c17a85698ae1`
- 官方基线：Hugging Face `Lightricks/LTX-2` model card two-stage sunset example
- 旧 worktree `/Users/mick/repos/sglang-ltx2-twostage-20707` 的经验只保留“源码级修复方向”和“workflow 教训”；凡是和当前 repo/worktree 混用后得出的数值结论，一律作废

### Included Branch History

- `a210a3613`: LTX2 two-stage pipeline 对齐官方 example
- `9cf4c214f`: `width/height` 语义对齐为最终分辨率
- `ba360cb38`: stage2 distilled LoRA 改为 unmerged path
- `3fc15a470`: 修 Gemma3 native loader 和 LoRA stage-switch offload
- `a2099b1b0`: 修 `--model-id LTX-2` registry 命中
- `37580c37e`: 默认禁用 custom stage1 guider
- `1268c5536`: 修 stage1 sigma schedule
- `1989058c1` / `ec7d2fa8e` / `d27839abc`: 对齐 initial AV latents 的 dump、dtype、layout
- `9eb8eb436` / `3274dbc55` / `0b0bd4138`: 对齐 CFG denoise path、scheduler step 语义、refinement scheduler step state

### Current Trusted State

- 当前可信进度需要回到这份 repo/provenance 重新计数；旧 worktree 上报过的 `78%` 到 `93%` 进度不再保留
- 当前最可信的状态是：
  - native LTX2 two-stage 主链路已经搭起来
  - 一批实现层问题已经进入 branch history
  - `LTX2AVDenoisingStage` 的 step/block divergence 需要在这份 repo 上重新 dump、重新定位
- 之后所有精度结论都必须绑定：
  - `repo path`
  - `branch`
  - `git rev-parse HEAD`
  - 关键源码文件 `sha1sum`

### Workflow Lessons

- 本地存在多个 repo/worktree 时，先打印并记录 `repo path + branch + HEAD`，再开始跑；中途切目录后必须重新核对，不能沿用上一份上下文
- 任何 dump、视频、日志，都必须绑定到具体 provenance；不同 worktree 的结果不能混着解释
- 任何 `scp` / `docker cp` / 容器内 `cp` 完成后，必须先验目标文件 `sha1sum`，确认远端实际跑到的新代码，再启动任务
- 如果远端 run 完才发现 hash 不对，这轮结果直接作废，不做数值分析
