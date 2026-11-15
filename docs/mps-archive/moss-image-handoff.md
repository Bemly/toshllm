# MOSS-Image (sd.cpp) 交接 — Metal 全 GPU 跑通；长 prompt 卡死根因已闭环

最后更新：2026-11-12 晚。给接手的人/agent，读完这一份就能继续，不要重复已排除的实验。

## 0. 一句话状态

**全 GPU（TE + DiT + VAE 全在 MTL0）已跑通并提速**（ToshGEMM，q8_0 13.24 TFLOPS）。
**长 prompt 卡死的根因已代码级闭环 + 实测坐实**：长 prompt → 激活超 VRAM 预算 →
segmented compute → **权重逐出级联（降级到磁盘 residency）→ 每步从盘串行重载**
（prefetch 在同后端配置下被结构性禁用）→ CPU 600%、GPU 空闲、RSS 4-6GB、一层 5 分钟。
**实测解法**：DiT 开 FA（`--flash-attn off,diffusion=on`）单步 270s → 45s；
或 prompt ≤1500 字符 / 1024×1024。

## 1. 目标

`engines/vision/stable-diffusion.cpp`（fork，分支 `fa-rdna2`）跑通 MOSS-Image：
MOSS-Image-Base DiT + Qwen3.5-9B 文本编码器 + FLUX.2 VAE，在 RX 6800 上 GPU 出图。

## 2. 当前性能（全 GPU，13.4GB VRAM）

| 场景 | 时间 |
|---|---|
| 512×512，8 步（cfg 3.0） | 采样 **31.6s**，整图 55s |
| 1024×1456，12 步，短 prompt | ~300s |
| 1024×1456，**2464 字符长 prompt，FA 默认开** | **130.5s / 12 步 = 36.2s/步**（基线 635.4s / 270s/步，4.9× / 7.5×） |
| MUL_MAT q8_0 m=4096,n=512,k=14336 | **13.24 TFLOPS**（mul_mv 回退 2.00） |
| MUL_MAT f16 同上 | **13.76 TFLOPS** |

FA 之后的瓶颈：130s 里 **VAE 解码占 55s（42%）**——下一轮优化目标。
加速默认值（`008861b`）：**`flash_attn_mode="on"`** + **`conv_direct_mode="on"`**（全模块强制，conv3d 也已 igemm，
0.3→1.6–5.2 TFLOPS，CONV_3D 276/276 过）。sage_attn（CUDA 专用）、enable_mmap（pread 快 7 倍）故意未开。
Metal opt-in 开关不强开：`GGML_METAL_CONCURRENCY_ENABLE`（MOSS DiT 数据竞争、结果不确定）、
`GGML_METAL_MM_AMD_ENABLE`（白图）、`TOSH_MM_ACC_HALF`（见下）。

**FA prefill 换成 ToshLLM `pf_amd`（2026-11-13，sd `4779eb8`，已同步上游 `8f3460b`，reduce 修复 `9d5404d`）**：`fa_amd.metal` 新增
`kernel_flash_attn_ext_amd_pf_dk{64,128,256}` 与 `pf_nomask_dk{64,128}`（来源 ToshLLM GPL-3.0，文件内已注明），
prefill（nq≥32）自动走，`GGML_METAL_FA_PF_DISABLE=1` 回退旧 tile。原版 half2 跨 128 维累加使出图偏离
（vs 无 FA PSNR 32.3dB），改为每 16 项刷 float 后 37.2dB（旧 tile 39.1dB）。FLASH_ATTN_EXT：sd 263/263、llama 921/921。
MOSS 形状 1.99→8.4 TFLOPS，1024×1536 每步 34.9→22.5s。llama Doujinshi-14B Q4_K_M：pp4096 77.6–80.7→86.0，
pp512 持平（llama prefill 被矩阵乘卡在 ~2.4 TFLOPS：spec 树无 ToshGEMM，只有 Q4_K/IQ3_S 的 mmq_amd）。

**1024×1536 每步 ~35s 拆账（pf 之前）**（cond+uncond 两次前向 × 42 层，L≈6170，test-backend-ops 实测形状）：
FA `fa_amd` tile 224.5ms/层 **2.08 TFLOPS → ~18.9s（54%）**；Q8_0 ToshGEMM 四个 matmul 153ms/层 ~9 TFLOPS
→ ~12.9s（37%）；其余 ~3s。**下一个大头是 FA prefill kernel**（逐 token 点积、PV 每行重读整块 V、
每 64 token 5 个 barrier），改成 ToshGEMM 式寄存器分块两段 GEMM 预计可到 5–7 TFLOPS。
`TOSH_MM_ACC_HALF=1`：matmul 9→13.7 TFLOPS，512²/8 步采样 32.6→27.2s，无 NaN，但与 float 出图 PSNR 37dB
（4.2% 像素差 >8 级），非逐位一致，未设默认。

**VAE 直接卷积看门狗已根治（2026-11-13，未提交）**：原 Metal `kernel_conv_2d` 是上游朴素版
（一线程一输出、标量三重循环、无 threadgroup 分块，~0.4 TFLOPS），flux2 VAE 在 1024×1536 的
上采样 conv（256→256 ch，3×3，~1.85 TFLOP）单个 dispatch 要好几秒 → 看门狗 status 5 → error state。
`e9f11c7` 的 `auto,vae=off` 只是绕开。现在 `conv.metal` 新增 `kernel_conv_2d_igemm`（ToshGEMM manual
骨架：64×64 tile、K 步 32、乒乓 16KB shmem、float 累加，直接从图像 gather，不落 im2col），
无 simdgroup matrix 的设备（RX 6800）自动走它；`GGML_METAL_CONV_NAIVE=1` 回退旧 kernel 对照。
sd.cpp 的 conv_direct auto 对非 Apple Metal 判定 on（`conv2d direct auto: MTL0 -> on (implicit GEMM)`）。

| 1024×1536 VAE 解码 | 超时 | decode | 备注 |
|---|---|---|---|
| 旧 direct conv | 3 次 | 32.6s（假成功） | backend 进 error state |
| im2col（`auto,vae=off`） | 0 | 57.9–61.5s | im2col 临时 buffer 把 TE 5.5GB 挤成 Disk 降级 |
| **igemm direct conv（现默认）** | **0** | **9.8–10.0s** | 无 Disk 降级 |

验证：`test-backend-ops -o CONV_2D -b MTL0` 2030/2030 过；perf 5–11×（1536²×32→64：230ms→19.8ms，
0.38→4.4 TFLOPS）；与 im2col 同 seed 出图 PSNR 66dB（1024×1536/1 步）/ 72.6dB（512²/8 步），最大差 1 级。

## 3. 长 prompt 卡死：根因闭环（2026-11-12）

### 症状

CPU 550–600%、GPU 个位数、RSS 从几百 MB 涨到 4–6GB、进度卡 0 步、
日志每 0.2s 一条 `loading tensors completed`（`model_loader.cpp:1490`）。

### 机制（四处代码 dive + 实测日志双重确认）

```
长 prompt → 文本 token 拼进 DiT 序列 → attention 激活暴涨（含 seq² score 物化）
  → ggml_runner.cpp:876-896 的 fits() 失败（预算 = max_vram_assignment.bytes_for_backend，diffusion_engine.cpp:312）
  → segmented compute（默认开，stable-diffusion.cpp:392；42 层每层一段，切分点 mark_graph_cut 在 moss_image.hpp:187）
  → 每段 ensure_compute_backend_capacity（model_manager.cpp:1879）逐出级联：
      抢 workspace → LRU → 最后把权重降级 Disk residency（model_manager.cpp:2017）
  → 下一段从盘重载（model_manager.cpp:468 load_tensors → 1490 刷屏）
  → 且 prefetch 被结构性禁用（model_manager.cpp:1732-1747：params==compute backend 时
     prefetch_candidate=false → 管线 disable）→ 重载完全串行
  → Q8_0 反量化 + copy_to_backend 全在 CPU → CPU 600%、GPU 空闲、RSS 4-6GB
```

### 实测对照（带可见性日志，`src/model_manager.cpp` +45/−9）

| 场景 | 单步 | 采样期逐出 | 模块换载 |
|---|---|---|---|
| 短 prompt 88 字符 | ~17s | 0 | 0 |
| 长 prompt 2464 字符（无 FA） | **270s**（每步 thrash） | 8 次释放 + 2 次 Disk 降级 | — |
| 长 prompt 2464 字符（FA on） | **36s** | **0** | 一次性：released ×7 + 1 次 5.27GB Disk 降级 |

**关键区分（别读混）**：FA 后剩余的 7 次释放/1 次降级发生在 **TE→DiT 和 DiT→VAE 模块切换**
（VAE 进显存、DiT 权重让位，一次性换载开销，设计内）；**采样期零逐出**——FA 把"每步 thrash"
彻底消掉了，不是"还差一点"。FA 单步 36.2s vs 基线 270s = **7.5 倍**；端到端 130.5s vs 635.4s = 4.9 倍。

### FA 反转（长 prompt 的最优解）

标准 attention 路径每层物化 `[heads, L_q, L_k]` 的 F32 score 矩阵——1024×1456 + 2464 字符下
这是每层**数百 MB**的临时激活，正是打爆 `fits()` 的元凶。FA（`fa_amd.metal`）不物化 score。

- 短 prompt：FA 慢 15%（多 4 个 cast/permute kernel，`ggml_extend.cpp:694-721`；RDNA2 无矩阵单元，FA 与 GEMM 全走 VALU）
- **长 prompt：FA 快 7.5 倍**（score 矩阵消失 → 激活回到预算内 → 零逐出）
- auto 探针只测 (512,64)/(1024,128) 两个形状，对 L≈6000+ 是盲的——**auto=off 在长 prompt 场景是误判**
  （`core/probe/flash_attn.cpp:86`：建议加 (2048,128)/(4096,128) case，或 auto 按 L 长度选择）
- 默认值：**`flash_attn_mode="on"`**；conv_direct auto（VAE 走 igemm 直接卷积，见 §2）
  sage_attn（CUDA 专用）和 enable_mmap（pread 快 7 倍）故意未开
- 若 TE 预填/VAE 解码回退异常：`--flash-attn off,diffusion=on` 回到实测最优（仅 DiT 开）
- 限制：dk==dv∈{64,128,256,512}、F16 KV、无 bias/softcap、mask ne[3]==1（`ggml_extend.cpp:772`）

### 已排除（都实测过，别再查）

- 算子回退：SD_STRICT_GPU 在卡死负载只抓到 11/258 RESHAPE（无害），无大规模回退
- bf16 promote：加载期转换，图里无 bf16 节点（MOSS 权重 Q8_0/f16；日志 RAM 0.00MB 佐证）
- batched CFG：MOSS 是 DiT，`sd_version_is_unet` 直接排除；#2149 是 UNet 的 GROUP_NORM 白图正确性 bug
- 桌面应用抢内存（dsh）：时间线不符（症状早于 dsh 启动 24 分钟）
- 局域网 vs 本机：无关
- LoRA 文件损坏：已重下逐字节验证

### 模式 B：GPU Timeout → error state

`command buffer 0 failed with status 5` → `GPU Timeout Error` → backend 进 error state
（日志原话 "recreate the backend to recover"），之后**所有请求秒失败于 TE 编码**（曾一次刷 542 条）。
backend 实例级问题，**重启进程即恢复**（不是设备级损坏）。触发源（什么让 command buffer 跑超时）
仍未定位——这是下一个立项方向。

## 4. 已交付（都在 `bemly/fa-rdna2`，已 push）

| commit | 内容 |
|---|---|
| `370612d` | Metal barrier 消除默认关闭 + NoCopy 非页对齐回退 |
| `41f25f5`+`5277bc4`+`04890d7` | `mul_mm_amd` 手写 GEMM（已被 ToshGEMM 取代，默认关） |
| `62f996f` | **ToshGEMM 移植**：`kernel_mul_mm` 手工寄存器分块路径，AMD 默认启用 |
| 后续 | pread 加载（29.5s→3.9s）、float GEMM 累加、tile band、bf16 promote、ncb 开关 |
| `807c2bd` | 逐出/Disk 降级可见性（INFO + 限频 WARN）+ FA 默认 on |

## 5. ToshGEMM 移植要点

来源：`github.com/engeldlgado/toshllm`，`patches/llama/metal/0006-toshgemm.patch`（新布局）。

- `kernels/mul_mm.metal`：新增 4 个 function constant（`FC_MUL_MM + 8..11`，我们树 +6/+7 已占用 amax/map0_ne02）
- 手工路径：每 lane 4×4 累加器、`(k,k+1)` 打包 rank-1 更新、每 NK=32 冲 float
- **坑**：8 subgroup（256 线程）时 A tile 只能由前 128 线程装载——`loadA` 守卫三处不能少；
  smem 固定 8192（B tile 要 4KB）
- `GGML_METAL_MM_MANUAL_DISABLE=1` 可回退 mul_mv

## 6. 可复现命令

### 6.1 全 GPU 出图（短 prompt，稳）

```sh
cd /Users/bemly/Projects/mps
engines/vision/stable-diffusion.cpp/build-upstream/bin/sd-cli \
  --diffusion-model models/vision/diffusion_models/MOSS-Image-Base-Q8_0.gguf \
  --llm models/vision/text_encoders/Qwen3.5-9B-Q8_0.gguf \
  --vae models/vision/vae/flux2-vae.safetensors \
  --tokenizer models/vision/tokenizers/tokenizer.json \
  --auto-fit off --cfg-scale 3.0 --rng cpu \
  -H 512 -W 512 --steps 8 -s 42 -p "a photo of a cat" -o out.png
```

### 6.2 长 prompt（带 FA）

服务启动加 `--flash-attn off,diffusion=on`，prompt 可到 ~2400 字符（单步 45s）。
**不加 FA 时长 prompt 单步 270s（segmented 逐出），别用。**

### 6.3 回归与自检

```sh
./build-upstream/bin/test-backend-ops test -b MTL0 -o MUL_MAT   # 2452/2452
ls -1 /Library/Logs/DiagnosticReports/*.gpuRestart | wc -l      # 应恒为 40（本轮零新增）
grep -ac "command buffer 0 failed with status 5" /tmp/sd-server-moss.log  # >0 = 后端已坏，重启
grep -ac "released params backend buffers" /tmp/sd-server-moss.log       # >0 = 逐出发生
grep -ac "demoted" /tmp/sd-server-moss.log                                # >0 = 踩到 Disk 降级
```

## 7. 环境与坑

- 构建：`cmake -S . -B build-upstream -DCMAKE_BUILD_TYPE=Release -DSD_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON -DSD_BUILD_EXAMPLES=ON -DLLAMA_OPENSSL=OFF`（WebUI 前端用 pnpm 构建，见 `examples/server/frontend/dist/`）
  - **注意 `SD_SERVER_BUILD_FRONTEND`**：cmake 默认 OFF 时二进制只内嵌占位页，用户看到的是
    "Stable Diffusion Server is running" 一行字。要内嵌真实 WebUI：cmake 加
    `-DSD_SERVER_BUILD_FRONTEND=ON`（自动用 `frontend/dist/gen_index_html.h`，无需 pnpm）；
    或启动时带 `--serve-html-path examples/server/frontend/dist/index.html`。
- 黑苹果（iMac20,1 + RX 6800，OpenCore，`agdpmod=pikera`）：GPU 挂掉无 panic，只有
  `gpuRestart` 报告 + WindowServer 被 watchdog 杀（`displayState: OFF`）。
- **不要建议重启机器**（用户明确说重启上百次没用，见 `AGENTS.md §0`）。
- 16GB 内存机器：sd-server 跑起来 free 只剩 150MB，**python 大对象会 MemoryError**（校验用 shell 工具）。
- `git push`：`git -c http.proxy= -c https.proxy= push bemly fa-rdna2`

## 8. 关键代码位置

| 文件 | 说明 |
|---|---|
| `ggml/src/ggml-metal/kernels/mul_mm.metal` | `kernel_mul_mm`，含 ToshGEMM 手工路径 |
| `ggml/src/ggml-metal/kernels/fa_amd.metal` | AMD flash attention（VALU 实现，无矩阵单元） |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | `get_pipeline_mul_mm` 手工路径参数（manual/narrow/double_buffer/acc_f32） |
| `src/core/compute_workspace.cpp` | `make_scheduler`（CPU 兜底）+ `SD_STRICT_GPU` 守卫 |
| `src/core/ggml_runner.cpp:876-896,922,988-1003` | `fits()` / segmented 决策 / `using N segments`（只在段数变化时打印）/ per-segment ensure_capacity |
| `src/model_manager.cpp:1732-1747,1879,2017,468` | prefetch 禁用判定 / 逐出级联 / Disk 降级 / 重载入口 |
| `src/model_loader.cpp:1490` | "loading tensors completed" 日志点（刷屏来源） |
| `src/pipeline/diffusion_engine.cpp:2489,2816` | batched CFG workaround（仅 UNet 走，MOSS 不适用） |
| `src/pipeline/image.cpp:190` | img2img strength → t_enc |
| `src/model/integrations/moss_image.cpp` | MOSS 模型定义 |
| `/tmp/toshllm` | 上游参考仓库（clone），`patches/` 下还有 FA、融合、MoE 等补丁 |

## 9. 修复优先级（未做项）

| 优先级 | 事项 |
|---|---|
| 1 | 长图前释放 VRAM（关 dsh/浏览器）+ FA 开 diffusion——已验证 5.4 倍 |
| 2 | Mode B 触发源定位：什么让 command buffer 跑超时（下一个立项） |
| 3 | 同 backend 逐出后允许 prefetch（`model_manager.cpp:1732-1747` 放宽 + `prepare_prefetch_params`）——线程风险，需实跑验证 |
| 4 | 逐出级联避免 Disk 降级（fail-fast） |
