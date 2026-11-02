#!/usr/bin/env bash
# Logs and results are unique for every run. The policy source is unchanged.
set -euo pipefail
INSERT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec 9>"$INSERT_ROOT/runtime/session.lock"
if ! flock -n 9; then
    echo 'Another robot_insert session is running. Stop it before starting a new demo.' >&2
    exit 1
fi
INSERT_POLICY=${1:-WaveArm}
case "$INSERT_POLICY" in WaveArm|CheatCode) ;; *) echo 'Use WaveArm or CheatCode.' >&2; exit 2 ;; esac
INSERT_GUI=${2:-true}
case "$INSERT_GUI" in true|false) ;; *) echo 'GUI must be true or false.' >&2; exit 2 ;; esac
INSERT_RUN="$(date -u +%Y%m%dT%H%M%SZ)_${INSERT_POLICY}_$RANDOM"
mkdir -p "$INSERT_ROOT/results/$INSERT_RUN"
printf '%s\n' "$INSERT_RUN" > "$INSERT_ROOT/runtime/latest-run.txt"
echo "Running $INSERT_POLICY. Results: $INSERT_ROOT/results/$INSERT_RUN"
exec "$INSERT_ROOT/scripts/enter.sh" /bin/bash /workspace/scripts/session.sh \
    "$INSERT_RUN" "$INSERT_POLICY" "$INSERT_GUI"
