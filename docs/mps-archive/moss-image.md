# 跑 MOSS-Image（sd.cpp / RX 6800 / Metal）

最后更新：2026-11-12。命令都是在这台机器上实测过的。

## 1. 模型文件

四个文件放在 `models/vision/` 下，按 sd-server 网页端要求的目录分类
（`diffusion_models/`、`text_encoders/`、`tokenizers/`、`vae/`）：

| 文件 | 目录 | 大小 | 作用 |
|---|---|---|---|
| `MOSS-Image-Base-Q8_0.gguf` | `diffusion_models/` | 5.5 G | DiT（42 层，dim 3072） |
| `Qwen3.5-9B-Q8_0.gguf` | `text_encoders/` | 9.5 G | 文本编码器 |
| `flux2-vae.safetensors` | `vae/` | 160 M | VAE |
| `tokenizer.json` | `tokenizers/` | — | Qwen 分词器 |

全 GPU 时显存合计约 **13.4 GB**（TE 8.0 + DiT 5.2 + VAE 0.16），16 GB 卡给
WindowServer 留 1 GB，别再往上加。

## 2. 构建

```sh
cd /Users/bemly/Projects/mps/engines/vision/stable-diffusion.cpp
cmake -S . -B build-upstream -DCMAKE_BUILD_TYPE=Release \
  -DSD_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DSD_BUILD_EXAMPLES=ON -DLLAMA_OPENSSL=OFF
cmake --build build-upstream -j8
```

`-DSD_METAL=ON` 不能少，否则 sd.cpp 侧的 Metal 扩展不编译。
`-DLLAMA_OPENSSL=OFF` 是因为系统 OpenSSL 3.5.9 低于上游要求的 3.6。

## 3. 官方推荐参数（模型卡 / 用户指南）

MOSS-Image-Base **没有做步数蒸馏**，参数对质量影响很大：

| 参数 | 官方推荐 | 对应我们的 CLI |
|---|---|---|
| 采样器 | Flow-matching Euler | 内部固定（日志：`sampling using Euler method`） |
| 步数 | **≥ 50**（25–35 可接受，低于 20 画面松散） | `--steps 50` |
| CFG | **3.0**（范围 3.0–6.0） | `--cfg-scale 3.0` |
| Shift | 5.0 | 由离散调度器内部对应，无单独开关 |
| 跳过层 | `uncond_skip_layers = [22]`（24、26 也有帮助） | 已硬编码在 `diffusion_engine.cpp:2853` |
| 分辨率 | 0.25–4 MP，推荐 1 MP（1024²）、1.5 MP（1456×1024）、2 MP（1456²） | `-H 1024 -W 1024` |
| 宽高约束 | **必须是 16 的整数倍** | 自己算好，不对会直接报错 |
| 语言 | 英文训练量最大；中文也支持 | `-p "..."` |
| 布局控制 | Prompt 里放 JSON（bbox 顺序 `yxyx`，坐标 `abs_centralized`） | `-p '{"summary":...,"items":[...]}'` |

**步数不是硬性 50**，边际收益掉得很快（512² 实测，同一 seed）：

| 步数 | 采样 | 效果 |
|---|---|---|
| 8 | 28 s | 只能看构图，细节糊 |
| 20 | 65 s | **已经可用**，二次元/插画风格看不出明显缺陷 |
| 30 | 96 s | 接近 50，明暗过渡更顺 |
| 50 | 155 s | 官方推荐值，头发/布料/晶体细节最细 |

结论：日常出图 20–30 步够用；给客户或要抠细节、画面里有密集文字/复杂布局时再上 50。
低步数吃亏的主要是写实质感和文字排版，不是构型。

## 4. 基本用法（全 GPU，推荐）

```sh
cd /Users/bemly/Projects/mps
engines/vision/stable-diffusion.cpp/build-upstream/bin/sd-cli \
  --diffusion-model models/vision/diffusion_models/MOSS-Image-Base-Q8_0.gguf \
  --llm            models/vision/text_encoders/Qwen3.5-9B-Q8_0.gguf \
  --vae            models/vision/vae/flux2-vae.safetensors \
  --tokenizer      models/vision/tokenizers/tokenizer.json \
  --auto-fit off --cfg-scale 3.0 --rng cpu \
  -H 512 -W 512 --steps 50 -s 42 \
  -p "a photo of a cat" -o out.png
```

出图在 `out.png`。512×512 **50 步约 157 秒**（加载 4s + 采样 148s + VAE 1.7s）。
想先看构图可以 `--steps 8`，28 秒出一张草图，但细节和 50 步差距很大。
256×256 八步约 **19 秒**。

## 5. 常用参数

| 参数 | 说明 |
|---|---|
| `-H/-W` | 分辨率。256 整图约 19s，512 约 37s，1024 也支持但显存和步时都涨 |
| `--steps` | 采样步数。官方要求 ≥ 50；8 步只能看构图 |
| `-s` | 随机种子，`-1` 随机、固定值可复现 |
| `-p` | 提示词，英文效果最好 |
| `--negative-prompt` | 负面提示 |
| `-b` | 批量张数，每多一张显存和耗时线性涨 |
| `--cfg-scale` | 无分类器引导，3.0 是默认推荐；1.0 = 不引导（快一倍，质量下降） |
| `--rng cpu` | 必须留着。`cuda` 那个选项在 Metal 上会走 std_default，结果不可复现 |
| `--auto-fit off` | 关掉自动放置，避免它把模块挪来挪去 |

### 只让 DiT 上 GPU（显存约 6 GB）

显存紧张时把文本编码器放 CPU，代价是每次启动多花约 20 秒：

```sh
... --backend te=cpu ...
```

### 全 CPU（最慢，但绝对不会崩）

```sh
... --backend te=cpu,diffusion=cpu --vae-tiling ...
```

8 步 256 约 245 秒。

## 6. 性能参考（实测，全 GPU）

| 场景 | 加载 | 采样 | 整图 |
|---|---|---|---|
| 256×256，8 步 | 4 s | 11.5 s | **19 s** |
| 512×512，8 步（草图） | 4 s | 28 s | **37 s** |
| 512×512，50 步（正式） | 4 s | 148 s | **157 s** |
| 1024×1024，50 步（正式） | 4 s | 771 s | **808 s（13.5 分钟）** |

矩阵乘现在跑 ToshGEMM（无 `simdgroup_matrix` 设备的寄存器分块路径），
q8_0 4096×512×14336 约 13 TFLOPS。

1024² 的 token 数是 512² 的 4 倍，单步 2.96 s → **15.4 s**（约 5.2×，attention 是超线性的）。
1024² 全 GPU 跑完 50 步显存没有溢出、`gpuRestart` 零新增，属于安全范围。
建议：先用 512² 扫 seed 和提示词，定稿再用 1024² 出。

**动漫/插画风格表现很好**（官方定位偏设计类，但二次元同样稳）；弱项是手部和长杆状道具
（法杖穿过手掌之类），需要靠 seed 多试或补负面提示。

## 7. 排错

**出白图（std=0、mean=255）**
先确认没有开那个会打挂 GPU 的旧 kernel：
```sh
env | grep GGML_METAL_MM_AMD_ENABLE     # 有就 unset
```
还白就把 `--backend te=cpu,diffusion=cpu` 跑一遍区分是 TE 还是 DiT 的问题。

**显示器黑屏 / 整机假死**
GPU 复位了。查：
```sh
ls -1 /Library/Logs/DiagnosticReports/*.gpuRestart | wc -l
```
跑一次涨一批，说明是当前 workload 打挂的，不是随机故障。当前版本正常应该**不涨**。
不要建议重启机器解决——重启不解决，先按上面的计数定位。

**变慢**
- 确认 `--auto-fit off`，并且没开 `GGML_METAL_MM_MANUAL_DISABLE`
- `GGML_METAL_MM_MANUAL_DISABLE=1` 会退回 `kernel_mul_mv`，慢 6 倍，只在排错时用
- 有别的进程占显存时，Metal 会颠簸，先 `ps aux | grep -E "llama|sd-cli"` 看看

**加载慢**
用 `mmap` 之外的方式才会慢；当前版本走 `pread`，15 GB 约 4 秒。如果慢到 30 秒，
说明二进制是旧的（`ifstream` 版本）。

## 8. 快速自检

```sh
cd engines/vision/stable-diffusion.cpp
# 1) GEMM 正确性：应 2452/2452
./build-upstream/bin/test-backend-ops test -b MTL0 -o MUL_MAT | tail -1
# 2) GEMM 吞吐：应约 13 TFLOPS
./build-upstream/bin/test-backend-ops perf -b MTL0 -o MUL_MAT \
  -p 'type_a=q8_0,type_b=f32,m=4096,n=512,k=14336' | tail -1
# 3) GPU 复位计数：跑图前后应不变
ls -1 /Library/Logs/DiagnosticReports/*.gpuRestart | wc -l
```

## 9. 相关文档

- `docs/moss-image-handoff.md`：实现细节、移植踩坑、剩余优化项
- 上游参考实现：`/tmp/moss_pipeline.py`（scheduler 取整、chat template、VAE 反归一化）
