# LTX-2 Precision Alignment Log

Updated: 2026-03-30 Asia/Shanghai

## Goal

对齐 `Lightricks/LTX-2` 官方 `diffusers` two-stage `sunset` 成品，不追 `bit-exact`，只追最终视频/音频尽量接近官方输出。

目标 case:

- prompt: `A beautiful sunset over the ocean`
- negative prompt: `shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static.`
- width: `1536`
- height: `1024`
- frames: `121`
- fps: `24`
- stage1 steps: `40`
- stage2 distilled steps: `3`
- guidance scale: `4.0`
- seed: `10`

## Trusted Baseline

- official numeric baseline:
  - `diffusers==0.37.0`
  - model snapshot:
    - `47da56e2ad66ce4125a9922b4a8826bf407f9d0a`
- official reference video:
  - `tmp/ltx23_sunset_official_h100.mp4`
- source-compare baseline:
  - latest local clone at `/tmp/diffusers_latest`
- current local SGLang precision branch:
  - `e2c2d84bd13a811b3dc8b412d8acca9b75a883a6`

## Current Product State

当前最佳 clean revision 成品:

- revision:
  - `e2c2d84bd13a811b3dc8b412d8acca9b75a883a6`
- rerun machine:
  - `h200_2`
- current output:
  - `tmp/ltx23_sunset_current_e2c2d84_h200.mp4`

对官方成品逐帧 compare:

- `avg_psnr ~= 11.86548`
- `avg_ssim ~= 0.324327`
- `min_psnr ~= 11.80022`
- `min_ssim ~= 0.311076`

结论:

- 成品仍然明显没有对齐官方 `diffusers`
- 当前进度判断: `95%`
- 现在的工作重点必须继续放在 `stage1 transformer / denoising` 主链，而不是运行性问题

## Precision Fixes That Should Stay

### 1. Stage2 distilled sigma schedule

- revision:
  - `fe4a972413f21ab3b7d8540def3eb85690a678cf`
- change:
  - `runtime/pipelines/ltx_2_pipeline.py`
  - 把 stage2 distilled sigma 从错误的 `4` 个值改回官方 `3` 个值
- verdict:
  - 这是正确的源码对齐
  - 但对最终成品几乎没有可见收益

### 2. CFG moved to x0-space

- revision:
  - `bec689a08df11a31474d664add9ae38be433c8e7`
- change:
  - `runtime/pipelines_core/stages/denoising_av.py`
  - 按官方语义在 `x0-space` 做 CFG，再转回 velocity
- product gain vs previous best:
  - `avg_psnr`: `11.85836 -> 11.86281`
  - `avg_ssim`: `0.322659 -> 0.323474`
- verdict:
  - 小幅净收益，应该保留

### 3. Transformer sigma/audio_sigma modulation alignment

- revision:
  - `e2c2d84bd13a811b3dc8b412d8acca9b75a883a6`
- changes:
  - `runtime/models/dits/ltx_2.py`
  - `runtime/pipelines_core/stages/denoising_av.py`
- aligned behavior:
  - `prompt_adaln` 使用 `sigma`
  - `audio_prompt_adaln` 使用 `audio_sigma`
  - AV cross-attn modulation timestep selection 对齐官方 API
- product gain vs `bec689a`:
  - `avg_psnr`: `11.86281 -> 11.86548`
  - `avg_ssim`: `0.323474 -> 0.324327`
- verdict:
  - 小幅净收益，应该保留

## Highest-Value Remaining Source Difference

当前最值得优先继续追的源码差异已经不是 attention backend，而是 `stage1` scheduler 语义。

对照官方 `diffusers==0.37.0` 可确认：

- 官方 `pipeline_ltx2.py`
  - stage1 默认 `sigmas = np.linspace(1.0, 1 / 40, 40)`
  - 然后在 scheduler 里继续做
    - `use_dynamic_shifting=True`
    - exponential `time_shift(mu=2.05)`
    - `shift_terminal=0.1`
- 本地旧实现
  - `runtime/pipelines/ltx_2_pipeline.py`
  - 直接按 token 数构造一套自定义 sigma schedule
  - 实际数值和官方 baseline 明显不同

典型量级：

- 官方 stage1 first5:
  - `[1.0, 0.996449, 0.992737, 0.988851, 0.984780]`
- 本地旧 stage1 first5:
  - `[1.0, 0.997985, 0.995870, 0.993649, 0.991312]`
- 官方 stage1 last5:
  - `[0.488430, 0.420719, 0.337809, 0.233936, 0.1]`
- 本地旧 stage1 last5:
  - `[0.615389, 0.544788, 0.449136, 0.312220, 0.1]`

这是一条直接作用在整个 stage1 denoising loop 上的主干差异，比之前一直盯的 attention backend 更像当前成品偏差的主因。

## Branches Already Ruled Out

### Text / connector math is not the current main blocker

在 clean revision `a409386e4d894ade65b925bed1b95abc2a2bea22` 上，已经确认：

- standalone `TextEncodingStage` 对官方 `encode_prompt(...)` 是 exact
- standalone `LTX2TextConnectorStage` 对官方 connectors 是 exact
- loader-loaded components 对官方 component outputs 是 exact

因此，不应再继续把主精力放在：

- Gemma loader kwargs
- connector 权重加载
- connector 数学实现
- `caption_projection` 实现本体

### Current residual is inside transformer / denoising

同一个 clean revision 下，尽管 text / connector standalone 路径 exact，full pipeline 仍然有：

- `stage1_video_latent mean ~= 0.00310461`
- `stage1_audio_latent mean ~= 0.01979691`

这说明当前剩余主问题已经不在 text side，而是在:

- `stage1 transformer`
- `denoising execution semantics`

## Next Step

只继续做精度相关工作：

1. 先验证 stage1 scheduler 对齐这刀是否带来明显成品收益
2. 如果收益有限，再回到 attention backend policy
3. 每次改动后只看:
   - 官方 `sunset` 成品 compare
   - `avg_psnr`
   - `avg_ssim`
4. 不再把 OOM、decode 内存修复、远端同步细节写进这份 log
