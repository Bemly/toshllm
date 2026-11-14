#!/bin/zsh
set -e
cd "$(dirname "$0")/.."
SKIP_AUDIO=1 SKIP_IMAGE=1 ./scripts/build-engines.sh
