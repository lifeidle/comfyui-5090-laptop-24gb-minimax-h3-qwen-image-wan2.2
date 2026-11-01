# Round 6 · ComfyUI 0.43、MOSS-Image、FLUX.3 Klein 与 H3 LightVAE（2026-11-01/02）

> 测量环境：ComfyUI **0.43.0** / PyTorch 2.14.1+cu130（优化内核保持激活，新增 mesh_bvh 等几何 capability）/
> NVIDIA 驱动 617.42 / RTX 5090 Laptop 24 GB。计时口径同前（服务端权威计时）。
> 重测注意：同 seed 同工作流命中全节点缓存会返回 0.0 s，必须换 seed。

## 1. 环境升级（0.39.0 → 0.43.0，跨 v0.40/v0.41/v0.42/v0.43 四版）

配套包：comfy-kitchen 0.2.40 / comfy-aimdo 0.6.0 / frontend 1.57.4 / workflow-templates 0.11.97 / embedded-docs 0.5.15。
升级注意：`comfyui-workflow-templates-media-assets-02==0.1.24` 清华镜像漏同步，需官方 PyPI 补装（wheel 体积大，官方源约 45 分钟）。

H3（MiniMax H3）随升级获得的改善：LightVAE 支持（编解码分离加载）、单图作为 clip 首帧解码、
PDD head banks 单头 LoRA 兼容（v0.40-0.43 累计，详见 release notes）。

## 2. MOSS-Image（OpenMOSS，2026-10-16 发布，Apache 2.0）—— 新覆盖组 29

**架构**：5B DiT（SFT）+ Qwen3.5-9B 文本编码器 + Flux2 VAE（latent 与 FLUX.2/3 同族）。
**卖点**：双语文字渲染 + JSON 布局精确控制（BBox 指定文字/物体位置）——直击中文物料场景。

权重（Comfy-Org/MOSS-Image，ModelScope 可达）：`moss_image_sft_int8_convrot` DiT 5.41 GB +
`qwen3.5_9b_int8_convrot` 编码器 9.75 GB + `flux2-vae` 0.34 GB（共 15.5 GB）。

| 测试 | 规格 | 服务端净计时 | 中文渲染 |
|---|---|---|---|
| T2I 同题（水乡 + 「水乡晨曲」牌匾） | 1024² / 50 步 / cfg 3.0 / AuraFlow shift 5 | 冷 49.0 s → **热 26.4 s** | **4/4 全对**，构图全要素到位 |
| **布局控制海报**（4 个 BBox：品牌横幅/菜单/营业时间/咖啡杯） | 1024² / 50 步 / cfg 3.0 | **29.9 s** | **4 块文字 100% 正确**（含「每日 8:00 - 22:00」数字格式），位置精确落位 |

布局控制用法（API）：`MOSSImageLayoutPrompt` 的输出是 JSON 字符串（summary.caption + items[label/caption/text/bbox] +
`bbox_coord_space: abs_centralized, bbox_format: yxyx`），直接作为 CLIPTextEncode 的 text——
**可以跳过画布节点（CreateBoundingBoxes 的 editor_state 是强制 socket）手工内联 layout JSON**，
bbox 用 0-1000 分数坐标 [y0,x0,y1,x1] 减去 width/2、height/2 的像素偏移。

对比 Qwen-Image 2512（12.2 s @1328²/4 步 Lightning，中文 30/30）：MOSS 是无蒸馏基线（26.4 s @1024²/50 步），
中文渲染同级（单测试样本均为满分）；速度上 Qwen-Image 的 Lightning 加速仍占优。
**手册「中文物料」推荐位建议并列：要快选 Qwen-Image 2512 + Lightning，要布局控制选 MOSS-Image。**

## 3. FLUX.3 Klein distilled（BFL，2026-10 开源）—— 新覆盖组 30

BFL 官方 `black-forest-labs/flux-3-image-klein-comfy`（HuggingFace）按 ComfyUI 布局打包：
distilled DiT nvfp4/int8-convrot 5.26-5.83 GB、qwen3vl_4b 编码器 int8 4.86 GB、flux2-vae 0.34 GB（与 MOSS 共用）。
FLUX.3 Dev 全量 backbone 仍未开源；大杯 Image/Max 仍 API-only。

实测（distilled int8-convrot + qwen3vl int8，CLIPLoader type=`flux3`，Flux2Scheduler 15 步，
BasicGuider 免 CFG，SamplerCustomAdvanced 结构）：
**同题水乡 1024²：6.8 s（含权重冷加载）**。牌匾「水乡晨曲」写成「水多晨曲」——"乡"错成"多"，**中文 3/4**。

**结论**：Klein distilled 是速度型选手（15 步 6.8 s），中文文字渲染不及 MOSS-Image（3/4 vs 4/4）。
FLUX.2 Klein 4B/9B（现图像编辑位）是否升级到 FLUX.3 Klein，建议再做编辑能力专项对比后决定。

## 4. H3 LightVAE 显存实测（lightx2v/LightVAE）

`lightvae-lite-h3-decoder-int8-convrot.safetensors`（**834 MB**）vs 标准 `minimax_h3_video_vae_fp16`（5.21 GB）。
T2VA 4 步工作流（1344×768，同 seed 777001，nvidia-smi 0.5 s 采样峰值）：

| VAE | 服务端净计时 | VRAM 峰值 |
|---|---|---|
| 标准 video_vae_fp16 | 154.7 s | **22142 MiB** |
| LightVAE int8 decoder | **135.7 s** | **22078 MiB** |

**诚实结论（负结果）**：24 GB 卡 + 1344×768 档位下，显存峰值收益 ≈ 0（64 MiB 在噪声内）——
**峰值由 H3 DiT（int8 15 GB）+ Qwen3.5-VL-32B 编码器（nvfp4 15.7 GB）主导，VAE 不在峰值窗口**。
LightVAE 的真实价值在 ① 8-12 GB 显存的卡（此时 VAE 占比大）② 超长帧数 decode（decode 峰值随帧数线性增长）。
速度上 LightVAE int8 decoder 反而快 19 s（-12%）。
画质未做逐帧对比，建议使用前抽看。

## 5. 模型生态扫描（2026-11-01）

- 视频线无变局：Wan 2.2 14B / MiniMax H3 仍是本地开源双旗舰；可灵 4.0 / Krea 3 / HeyGen / PixVerse V6 Max
  全部为 ComfyUI 云端 Partner 节点；Seedance 仍 API-only；Step 5 Preview 是 600B LLM 基座（非视频）。
- Wan-Dancer-14B（音乐驱动舞蹈，双模型 69 GB）仍列为暂缓。
- 图像线新选择：MOSS-Image（本文已覆盖）、Fibo 2.0（Bria 开源，轻量备选，未测）、
  FLUX.3 Klein（已覆盖）。Qwen-Image 3.0 / Wan 2.5-3.0 / FLUX.3 Max 仍 API-only。

## 6. 覆盖度终态

**30 组实测** = 图像 13 + 视频 6 + 角色·动作迁移 1 + **MOSS-Image 文生图/布局控制 1（新增）** +
**FLUX.3 Klein 文生图 1（新增）** + 音乐 5 + 3D 1 + 放大修复 2 + 量化对比实验等。
ComfyUI 引擎 0.43.0、PyTorch 2.14.1+cu130、驱动 617.42。
