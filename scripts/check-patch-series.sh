#!/bin/zsh
# Obsolete: engines are Bemly git submodules (see patches/README.md / .gitmodules).
# Kept so old docs/CI links fail loudly with guidance.
echo "check-patch-series.sh retired: engines are submodules under vendor/{llama,audio,stable-diffusion}.cpp" >&2
echo "Bump with: git submodule update --remote <path>   or the bump-engines workflow" >&2
exit 1
