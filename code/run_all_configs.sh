#!/bin/bash
# Launches every config in configs/ (except base.yaml) via launch_experiment.sh.
# Extra args (--no-slurm, --tmux) are forwarded to each launch.
set -euo pipefail
cd "$(dirname "$0")"

for cfg in configs/*.yaml; do
    [ "$(basename "$cfg")" = "base.yaml" ] && continue
    ./launch_experiment.sh "$cfg" "$@"
    sleep 2
done
