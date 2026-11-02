#!/usr/bin/env bash
set -eo pipefail
INSERT_RUN=$1
INSERT_POLICY=$2
INSERT_GUI=$3
INSERT_SETTINGS=${4:-}
INSERT_OUTPUT="/workspace/results/$INSERT_RUN"
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
export GZ_PARTITION="robot_insert_${INSERT_RUN}"
export AIC_RESULTS_DIR="$INSERT_OUTPUT/scoring"
export ROS_LOG_DIR="$INSERT_OUTPUT/ros_logs"
mkdir -p "$AIC_RESULTS_DIR" "$ROS_LOG_DIR"
cleanup() {
    trap - EXIT INT TERM
    # These process groups only contain children of this isolated session.
    for pid in "${INSERT_MONITOR_PID:-}" "${INSERT_MODEL_PID:-}" "${INSERT_EVAL_PID:-}"; do
        if [[ -n "$pid" ]]; then kill -INT -- "-$pid" 2>/dev/null || true; fi
    done
    sleep 2
    for pid in "${INSERT_MONITOR_PID:-}" "${INSERT_MODEL_PID:-}" "${INSERT_EVAL_PID:-}"; do
        if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    done
    sleep 2
    for pid in "${INSERT_MONITOR_PID:-}" "${INSERT_MODEL_PID:-}" "${INSERT_EVAL_PID:-}"; do
        if [[ -n "$pid" ]]; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
INSERT_TRUTH=false
if [[ "$INSERT_POLICY" == CheatCode ]]; then INSERT_TRUTH=true; fi
INSERT_EVAL_EXTRA=()
if [[ "$INSERT_POLICY" == FTACT ]]; then
    INSERT_EVAL_EXTRA=("aic_engine_config_file:=$INSERT_OUTPUT/engine_config.yaml")
fi
setsid /entrypoint.sh \
    "gazebo_gui:=$INSERT_GUI" launch_rviz:=false \
    "ground_truth:=$INSERT_TRUTH" start_aic_engine:=true \
    shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=120 \
    "${INSERT_EVAL_EXTRA[@]}" \
    > "$INSERT_OUTPUT/eval.log" 2>&1 &
INSERT_EVAL_PID=$!
# The router belongs to the evaluator and must start before the model.
sleep 3
if [[ "$INSERT_POLICY" == FTACT ]]; then
    INSERT_ML_PATH="/workspace/scripts:/workspace:/workspace/tools/lerobot_reader_python:/workspace/tools/dataset_python:/host_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
    setsid env PYTHONPATH="$INSERT_ML_PATH" PYTHONDONTWRITEBYTECODE=1 \
        FTACT_SETTINGS="$INSERT_SETTINGS" \
        ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=policies.FTACT \
        > "$INSERT_OUTPUT/model.log" 2>&1 &
else
    setsid ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
        "-p" "policy:=aic_example_policies.ros.$INSERT_POLICY" \
        > "$INSERT_OUTPUT/model.log" 2>&1 &
fi
INSERT_MODEL_PID=$!
setsid python3 /workspace/scripts/monitor.py "$INSERT_OUTPUT" \
    > "$INSERT_OUTPUT/monitor.log" 2>&1 &
INSERT_MONITOR_PID=$!
INSERT_STATUS=0
wait "$INSERT_EVAL_PID" || INSERT_STATUS=$?
printf '%s\n' "$INSERT_STATUS" > "$INSERT_OUTPUT/evaluator-exit-code.txt"
exit "$INSERT_STATUS"
