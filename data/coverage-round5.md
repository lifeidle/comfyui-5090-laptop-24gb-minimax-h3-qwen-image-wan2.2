# Round 5 · 环境升级、NVFP4 复测与 Animate-2（2026-10-11）

> 测量环境：ComfyUI 0.39.0 / PyTorch 2.14.1+cu130 / NVIDIA 驱动 617.42 / RTX 5090 Laptop 24 GB / Python 3.11.9。
> 所有计时均为**服务端权威计时**（execution_start → execution_success 时间戳差）。
> 复测注意事项：同 seed 同工作流会命中 ComfyUI 全节点缓存并返回 0.0 s 假计时，重测必须换 seed。

## 1. 环境升级

| 组件 | 升级前 | 升级后 |
|---|---|---|
| ComfyUI | 0.38.0 | **0.39.0**（codeload tarball 覆盖装，exclude models/custom_nodes/user/input/output） |
| PyTorch | 2.11.0+cu128 | **2.14.1+cu130** |
| NVIDIA 驱动 | 616.56 | 617.42 |
| pip | 24.0 | 26.2.1 |

配套包随 requirements.txt 一次锁定：comfy-kitchen 0.2.37 / comfyui-frontend-package 1.53.10 /
comfyui-workflow-templates 0.11.76 / comfyui-embedded-docs 0.5.13。

升级后启动日志：`comfy_kitchen backend cuda: {'available': True, 'disabled': False}` ——
**搁置已久的 cu130 优化内核（NVFP4 / INT8-ConvRot / FP8 全套）正式激活**。
升级前该行是 `'disabled': True`（原因：`You need pytorch with cu130 or higher`）。

## 2. NVFP4 复测（Z-Image-Turbo，1024² / 8 步，同参数同源）

| 配置 | 服务端净计时 | 说明 |
|---|---|---|
| Round 3/4 基线（cu128，eager 回退） | 13.8 s（两轮 17.7 / 13.8） | §5 时代数字 |
| cu130 首跑（含 8.33 GB 权重冷加载） | 16.9 s | 参考值，不进对比 |
| **cu130 热跑 seed 777001** | **4.0 s** | |
| **cu130 热跑 seed 888002** | **4.0 s** | 复测一致 |
| int8 对照（cu130，seed 888002） | 10.5 s | convrot 反量化路径 |

**结论**：

1. §5 的「NVFP4 快 22% 是下界」预言兑现，真实差距 **3.45×**（13.8 → 4.0）。
   旧数字是 nvfp4 与 bf16 **都在 eager 回退下**测的相对值。
2. 内核解锁后 NVFP4 对 INT8 的优势扩大到 **2.6×**（旧 22%~25% → 新 2.6×）。
   INT8 走 convrot 反量化，从 eager 到内核收益很小（11.3→10.5）；NVFP4 独占
   Blackwell 原生 fp4 加速单元（`scaled_mm_nvfp4`），内核解锁后差距才真正拉开。
3. 量化选型结论不变且更强：Blackwell（sm_120/121）上 4-bit 一律 NVFP4。
4. 画质结论（§5 的噪声地板实验）不受影响——本轮只复测速度。

## 3. 新增覆盖组 28：Wan 2.2 Animate-2 14B（角色·动作迁移）

模型血统：Wan-AI 官方 2026-07 发布（Apache 2.0），ComfyUI 0.39 原生支持
（`WanAnimate2ToVideo` Move / `WanAnimateToVideo` Mix 两个节点并存）。

权重选型：**int8_convrot 单文件 18.41 GB**（Comfy-Org repackage；cu130 内核原生加速格式，
比 bf16 单文件 32.79 GB 省一半显存）。文本编码器 umt5_fp8 / VAE wan_2.1 / 4 步 LoRA
（lightx2v I2V rank64）全部复用 Round 1-4 资产，零重复下载。

| 模式 | 拓扑 | 分辨率/帧 | 步数 | 服务端净计时 | 产物 |
|---|---|---|---|---|---|
| **Move**（参考图角色执行动作视频） | LoadImage + LoadVideo/GetVideoComponents → WanAnimate2ToVideo（pose_video 直吃视频帧，姿态编码内置，无需 DWPose）→ KSampler → TrimVideoLatent | 832×480 / 77 帧 / 16 fps | 4 步 cfg 1.0（lightx2v） | **105 s**（含 18.41 GB 权重首次加载） | wan_animate_move_test_00001_.mp4 |
| **Mix**（参考图角色换入视频场景，背景保留） | 背景帧 ImageScale 512² → PointsEditor(256,256) → Sam2Segmentation(video) → GrowMask(10) → BlockifyMask(32) → WanAnimateToVideo(background_video + character_mask) | 640×640 / 77 帧 / 16 fps | 6 步 cfg 1.0 | **155 s**（含权重加载 + SAM2 逐帧分割） | wan_animate_mix_test_00001_.mp4 |

Mix 附加资产：ComfyUI-segment-anything-2 节点包（kijai，零额外依赖）+
`sam2_hiera_base_plus.safetensors` 323 MB（官方模板同款 2.0 base+ 档，models/sam2/）。

未接：`face_video`（表情迁移链）——身体动作 + 背景替换已完整；表情迁移为可选加强。

## 4. 模型生态结论（2026-10-11 扫描，多源交叉核实）

- **Wan 2.5 / 2.6 / 2.7 / 3.0**（3.0 于 2026-08-24 发布，30 s 单段原生音频）：**全部无开源权重**，
  API-only（Wan-AI HF org 2026-09-24 逐仓核实仍只有 2.1/2.2 时代资产）。
  **Wan 2.2 14B MoE 仍是本地开源视频天花板**。2026-07 新开源的 Wan 特化模型：
  Wan2.2-Animate-2-14B（本文已覆盖）、Wan-Dancer-14B（音乐驱动舞蹈，global+local 双模型
  共 69 GB，暂缓）。
- **Qwen-Image 3.0**（2026-07-21 发布）：API-only 无权重；2.1（2026-09-20）仍是最新本地权重
  （注意其许可证为 Qwen Research License 非商用，商用位仍以 2512 Apache 2.0 为准）。
- **FLUX 3 Image**（2026-10-01 发布）：API-only，BFL 官方称开源 Dev 版数周内发布——
  权重落地后按 FLUX.2 经验（32B → Q4 GGUF 19-20 GB）24 GB 卡是 Q4 档舒适入场线，值得盯。
- **ACE-Step 1.5**（含 XL 4B 系列）仍是最新开源音乐模型，无续作。

## 5. 覆盖度终态

**28 组实测** = 图像 13 + 视频 6 + **角色·动作迁移 1（新增）** + 音乐 5 + 3D 1 + 放大修复 2。
全部出片/出曲/出模。
