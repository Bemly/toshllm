# Patches

Patches against the pinned upstream commits, one directory per engine:

```
llama/          llama.cpp, whole-tree merge split into 7 path-disjoint patches (0001-0007)
llama-tests/    llama-server integration suites, kept outside the patch series
image/          stable-diffusion.cpp (Bemly fork), applied by scripts/engines/image.sh
audio/          audio.cpp (speech-to-text), applied by scripts/engines/audio.sh
diag/           diagnostic-only patches for bench/profiling builds, not applied by default
```

`scripts/build-engines.sh` discovers them per engine and applies them in **numeric order of the
file name**, recursively, so the number is the apply order: never reuse one, and keep it when
moving a file.

## `llama/`

Since the toshllm 0.87.34 merge this is **not** a per-feature series. The whole merge was
committed once in the scratch port tree and the resulting tree diff against the spec base is
split by path into seven patches. They are mutually disjoint and together cover every changed
file, so the numeric prefix is just the apply order — the result is the same in any order.

Base: `d469424b236c3a01554a23b52925f00c886b42df` (Bemly/llama.cpp `spec`). This is also
`LLAMA_COMMIT` in `scripts/build-engines.sh`, so `git apply` of the series lands directly on the
clone the build script makes.

| patch | paths |
|---|---|
| `0001-ggml-metal.patch` | `ggml/src/ggml-metal/` (host + kernels, including the new `kernels/fa_tosh.metal`) |
| `0002-ggml-core.patch` | the rest of `ggml/` |
| `0003-llama-src.patch` | `src/`, `include/` |
| `0004-common.patch` | `common/` |
| `0005-server.patch` | `tools/server/` |
| `0006-web-ui.patch` | `tools/ui/` |
| `0007-other.patch` | everything else (`CMakeLists.txt`, `cmake/`, `examples/`, `models/`, `tests/`, `tools/llama-bench/`, `tools/mtmd/`, `tosh-moe/`) |

New files are included by the two-tree diff and `--binary` is used so binary assets are never
silently dropped. The server test suites that used to live at `llama/server/tests/` now live in
`patches/llama-tests/`, outside the series (see `scripts/test-engine.sh`).

### Re-syncing with toshllm upstream

The per-feature patches are gone; do not add an `0008`. In the scratch repo
(`~/Projects/toshllm-port/llama`) keep a `tosh-upstream` branch at the new upstream tree, merge
it into `port`, resolve conflicts by the rules in `MERGE.md`, commit, then re-export all seven
patches with `git diff -U8 --binary --no-renames <base> port -- <path group>` and re-run the
round-trip check below.

## `image/`

The image engine is the Bemly/stable-diffusion.cpp fork (`fa-rdna2`, pinned at full hash
`9d5404dbfe8a26dba98dbdafa97898852cac39d3` in `scripts/engines/image.sh`). Its ggml vendors
the split `kernels/` layout, so the seven patches here apply to the fork root with plain
`git apply`, in filename order, with no `--reject`/`-C2` handling and no submodule init.

**Kept (src-level feature ports only):**

| patch | purpose |
|---|---|
| `0050-image-vae-tile-yield.patch` | VAE tile yield |
| `0053-image-lora-f16-weight-cast.patch` | LoRA f16 weight cast |
| `0057-image-qwen21-fused-swiglu.patch` | Qwen-Image 2.1 fused SwiGLU |
| `0059-image-qwen21-rope-table.patch` | Qwen-Image 2.1 RoPE table |
| `0060-image-qwen21-prefix-f16.patch` | Qwen-Image 2.1 prefix f16 |
| `0062-image-qwen21-f16-inputs.patch` | Qwen-Image 2.1 f16 inputs |
| `0063-image-qwen21-v-view.patch` | Qwen-Image 2.1 V view |

**Deleted (do not restore):** `0001-1` … `0001-6`. The fork already has ToshGEMM + FA
(`amd_tile` / `amd_pf` / `amd_vec`). `0001-6` replaced `fa_amd.metal` with toshllm names
(`tile_amd`) while host still looked up `amd_tile` → missing kernel / crash; restoring only
the metal file still GPU-timed-out, so the whole `0001-*` stack is untrusted (T6 / T7b,
2026-11-15).

Round trip (T7b): applying the seven in order to a clean `9d5404d` copy and
`git add -A -f && git write-tree` reproduces `port^{tree}` exactly (`6d857cd…`).

The fork already carries the rest of the old toshllm image work (pread loader, bf16 promote,
`TOSH_MM_ACC_HALF`, nc_b, the qwen-21 down-projection F16 path, FA nomask, mm band, im2col
rewrites and the wave64 dispatches), so the originals that targeted the retired leejet `ggml`
were deleted rather than ported — see the `## sd` / `## T7b` sections of `PORTING-LOG.md`.

## Changing one

Work in the scratch port tree, not the vendor tree: `git diff <base> port` is the single source
of truth for the whole series, and it may also carry experiments that are in no patch.

```sh
# in ~/Projects/toshllm-port/llama, after editing and committing the merge on `port`
git diff -U8 --binary --no-renames d469424b2 port -- tools/ui > \
    ~/Projects/toshllm/patches/llama/0006-web-ui.patch
# then re-run the round-trip check below
```

Three rules that have each cost real time:

- **`-U8` and `--binary` for `llama/`.** Anything narrower rewrites every hunk header in the file,
  and the `git diff` default of three lets a hunk land in a different kernel while the build
  still exits 0. A two-tree `git diff <base> port` includes new files automatically; `--binary`
  keeps binary assets from being dropped.
- **Every patch must still apply.** `scripts/check-patch-series.sh <ref>` runs the whole series in
  a throwaway worktree and reports each patch. The seven groups are path-disjoint, so order does
  not change the outcome, but the numeric prefix remains the apply order.
- **Verify the round trip (zero diff).** Apply `0001`–`0007` in order into a worktree at the
  pinned commit, stage everything with `git add -A -f`, then compare `git write-tree` with
  `port^{tree}`. The hashes must be identical; a file-by-file diff alone can miss file modes and
  ignored new files.

`scripts/build-engines.sh` resets the vendor tree and re-applies the patch files, so working
tree edits are discarded: regenerate before building, never after.
