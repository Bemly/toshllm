# Patches

Engine sources no longer live behind a patch series. They are **git submodules**
pinned to Bemly forks (see `.gitmodules`):

| path | remote | notes |
|---|---|---|
| `vendor/llama.cpp` | https://github.com/Bemly/llama.cpp | `main` = ToshLLM engine tree (FA-RDNA2 + ToshGEMM + server/UI). Clean FA-only tip kept on branch `spec`. |
| `vendor/stable-diffusion.cpp` | https://github.com/Bemly/stable-diffusion.cpp | `main` = FA-RDNA2 + image feature ports |
| `vendor/audio.cpp` | https://github.com/Bemly/audio.cpp | `main` = FA/ToshGEMM + metallib preload |

Build with `./scripts/build-engines.sh` (no `git apply`). Bump with the weekly
`bump-engines` workflow or by hand:

```sh
git submodule update --remote vendor/llama.cpp
git add vendor/llama.cpp && git commit -m "Bump llama.cpp submodule"
```

## Still under `patches/`

| dir | purpose |
|---|---|
| `llama-tests/` | llama-server integration suites (`scripts/test-engine.sh`) |
| `diag/` | optional diagnostic overlays for bench builds (not applied by default) |

Do **not** reintroduce a `patches/llama/` or `patches/image/` apply series — that
duplicates (and can regress) the Metal fast paths already on the forks.
