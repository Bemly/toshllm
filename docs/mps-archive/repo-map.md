# 仓库 / 工作树关系图（2026-11-15 刷新）

> 旧 worktree 目录（`llama.cpp-kvmem` / `metal-exact` / `ninfer-plan` / `rx6800*`）已在
> 2026-11-10 删掉，分支 ref 仍在 `Bemly/llama.cpp` 可重建。根目录曾经出现的空壳
> `audiocpp-src/`、断掉的 `llama.cpp` 软链已于 2026-11-15 清掉——真源码只在 `engines/` 下。
> **`models/` 只在本地、永不提交、清理时禁止动。**

## 一句话

三个自有 fork，各自一个工作树：

| 引擎 | 本地路径 | GitHub | 默认分支 / HEAD |
|---|---|---|---|
| llama.cpp | `engines/text/llama.cpp-spec`（本地仍是干净 FA 工作区） | https://github.com/Bemly/llama.cpp | **`main`** @ `cade09cae`（ToshLLM 整树 = FA/ToshGEMM + server/UI）；干净 FA tip 留在分支 **`spec`** @ `5efc52b61` |
| stable-diffusion.cpp | `engines/vision/stable-diffusion.cpp` | https://github.com/Bemly/stable-diffusion.cpp | **`main`** @ `9561c75`（FA-RDNA2 + image 0050+） |
| audio.cpp | `engines/audio/audiocpp-src` | https://github.com/Bemly/audio.cpp | **`main`** @ `27471850`（ToshGEMM + FA + metallib 预加载） |

> 2026-11-15：三叉默认分支均为 `main`。ToshLLM App 用 **git submodule** 钉上述 tip（`Bemly/toshllm` @ `497f794`），已删除 `patches/{llama,image,audio}` 叠层，避免再盖坏 Metal 快路径。

上游对照 remote（只拉不推）：llama → `ggml-org/llama.cpp`；sd → `leejet/stable-diffusion.cpp`；audio → `0xShug0/audio.cpp`。

RDNA2 加速默认在：ToshGEMM（`GGML_METAL_MM_MANUAL_DISABLE=1` 关）、FA prefill pf_amd（`GGML_METAL_FA_PF_DISABLE=1` 关）。来源 ToshLLM 补丁为 GPL-3.0。

## 目录表（现在磁盘上有什么）

| 路径 | 是什么 | 备注 |
|---|---|---|
| `engines/text/llama.cpp-spec` | **唯一** llama worktree / 主 git 库 | 含 FA-AMD + PQ2/Bonsai + kvmem + ninfer P0–P6 + 上游同步；26 模型矩阵见 `docs/spec-mtp-plan.md §5` |
| `engines/vision/stable-diffusion.cpp` | MOSS / 图像引擎 | 构建优先 `build-upstream`；MOSS 用法见 `docs/moss-image.md` |
| `engines/audio/audiocpp-src` | ASR / TTS 引擎 | 目录名仍叫 `audiocpp-src`，分支是 `rx6800-fa`（旧文档里的 `audiocpp-rx6800-fa` 已并到这里） |
| `models/{text,vision,audio}/` | 权重（gitignore） | **511G+ 本地资产，任何清理都不要碰** |
| `docs/` / `scripts/` / `patches/` | 文档、入口脚本、历史补丁 | |
| `envs/` / `logs/` / `bench-results/` / `out/` / `outputs/` | 运行环境与实验产物 | 可按需再清，与引擎无关 |

已删除、不要再写进新文档的路径：`engines/text/llama.cpp-kvmem` 等六个旧 worktree、`engines/audio/audiocpp-rx6800-fa`、根目录 `audiocpp-src/`、根目录 `llama.cpp` 软链、`h3check` / `indextts-src` / `Strata` 等旧对照 clone（分支/补丁可能还在 git 或 `patches/`）。

## 构建目录

| 在哪编 | 目标 | 说明 |
|---|---|---|
| `llama.cpp-spec/build-spec` | `llama-server` / `llama-cli` / `llama-bench` / kvmem 变体 | 主力：`-DLLAMA_KVMEM=ON` |
| `stable-diffusion.cpp/build-upstream` | `sd-cli` / `sd-server` | 对照与日常；`build/macos-metal-release` 若 CMake 路径 stale 就别用，改 `build-upstream` |
| `audiocpp-src/build-upstream` | `audiocpp_cli` / `audiocpp_server` | 同上；旧 `build/macos-metal-release` 可能 stale |

## 模型（本地 only）

全在 `models/`，gitignore。text 全矩阵见 `docs/spec-mtp-plan.md §5`；vision（MOSS）见 `docs/moss-image.md`；audio 包见 `CONFIG.md` 历史段与 `scripts/dl-audiocpp-packages.sh`。

## 新会话 sanity check

```sh
# 三棵树都在默认分支、干净
git -C engines/text/llama.cpp-spec status -sb && git -C engines/text/llama.cpp-spec log --oneline -1
git -C engines/vision/stable-diffusion.cpp status -sb && git -C engines/vision/stable-diffusion.cpp log --oneline -1
git -C engines/audio/audiocpp-src status -sb && git -C engines/audio/audiocpp-src log --oneline -1

# llama 轻量 bench（先按 AGENTS.md §0 查内存/GPU）
cmake --build engines/text/llama.cpp-spec/build-spec --target llama-bench -j 4
GGML_METAL_N_CB=16 ./engines/text/llama.cpp-spec/build-spec/bin/llama-bench \
  -m /Users/bemly/Projects/mps/models/text/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf \
  -ngl 99 -p 512 -n 64 -r 1 -ctk q8_0 -ctv q8_0 -ub 128
```

## 历史分支图（Bemly/llama.cpp，仅追溯）

```
upstream/master ──...──┬── rx6800-fa / dk128 / q8-pq2（旧）
                       └── kvmem-eval → metal-exact-alloc → spec（现默认）
```

旧交接文（`fa-rdna2-handoff.md`、`metal-exact-alloc-plan.md`、`mmq-decode-handoff.md` 等）保留实验结论，路径以本文件为准。
