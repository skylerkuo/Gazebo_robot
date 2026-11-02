#!/usr/bin/env bash
# Four-view learned ACT, official evaluator, UI and crash-free isolated teardown.
set -eo pipefail
ulimit -c 0
INSERT_RUN=$1
INSERT_GUI=$2
INSERT_WALL_TIMEOUT=${3:-1200}
INSERT_POLICY_MODULE=${4:-policies.FTACT}
INSERT_GROUND_TRUTH=false
case "$INSERT_POLICY_MODULE" in
    policies.FTACT|policies.HumanFTACT) ;;
    policies.AutoExpertFTACT) INSERT_GROUND_TRUTH=true ;;
    *) exit 2;;
esac
INSERT_OUTPUT="/workspace/results/$INSERT_RUN"
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
export GZ_PARTITION="robot_insert_multiview_${INSERT_RUN}"
export AIC_RESULTS_DIR="$INSERT_OUTPUT/scoring"
export ROS_LOG_DIR="$INSERT_OUTPUT/ros_logs"
mkdir -p "$AIC_RESULTS_DIR" "$ROS_LOG_DIR"
cleanup() {
    trap - EXIT INT TERM
    if [[ -n ${INSERT_MONITOR_PID:-} ]]; then
        kill -INT -- "-$INSERT_MONITOR_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$INSERT_MONITOR_PID" 2>/dev/null || break
            sleep .1
        done
        kill -KILL -- "-$INSERT_MONITOR_PID" 2>/dev/null || true
    fi
    # Avoid this host's Gazebo graphics destructor/WSL core-dump bug.
    for INSERT_PID in "${INSERT_MODEL_PID:-}" "${INSERT_EVAL_PID:-}"; do
        if [[ -n "$INSERT_PID" ]]; then kill -KILL -- "-$INSERT_PID" 2>/dev/null || true; fi
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
setsid /entrypoint.sh "gazebo_gui:=$INSERT_GUI" launch_rviz:=false \
    "ground_truth:=$INSERT_GROUND_TRUTH" start_aic_engine:=true shutdown_on_aic_engine_exit:=false \
    model_discovery_timeout_seconds:=120 \
    "world_file:=$INSERT_OUTPUT/sideview_world.sdf" \
    "description_file:=$INSERT_OUTPUT/recording_robot.xacro" \
    "ros_gz_bridge_config_file:=$INSERT_OUTPUT/bridge.yaml" \
    "aic_engine_config_file:=$INSERT_OUTPUT/engine_config.yaml" \
    > "$INSERT_OUTPUT/eval.log" 2>&1 &
INSERT_EVAL_PID=$!
sleep 3
INSERT_ML_PATH="/workspace/scripts:/workspace:/workspace/tools/lerobot_reader_python:/workspace/tools/dataset_python:/host_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
setsid env PYTHONPATH="$INSERT_ML_PATH" PYTHONDONTWRITEBYTECODE=1 \
    FTACT_SETTINGS="$INSERT_OUTPUT/settings.json" \
    ros2 run aic_model aic_model --ros-args -p use_sim_time:=true "-p" "policy:=$INSERT_POLICY_MODULE" \
    > "$INSERT_OUTPUT/model.log" 2>&1 &
INSERT_MODEL_PID=$!
setsid python3 /workspace/scripts/monitor.py "$INSERT_OUTPUT" > "$INSERT_OUTPUT/monitor.log" 2>&1 &
INSERT_MONITOR_PID=$!
INSERT_STATUS=0
python3 /workspace/scripts/sideview_engine_done.py "$INSERT_OUTPUT" \
    --evaluator-pid "$INSERT_EVAL_PID" --model-pid "$INSERT_MODEL_PID" \
    --timeout-seconds "$INSERT_WALL_TIMEOUT" > "$INSERT_OUTPUT/engine_monitor.log" 2>&1 || INSERT_STATUS=$?
if [[ "$INSERT_STATUS" != 0 ]]; then cat "$INSERT_OUTPUT/engine_monitor.log" >&2 || true; fi
printf '%s\n' "$INSERT_STATUS" > "$INSERT_OUTPUT/evaluator-exit-code.txt"
exit "$INSERT_STATUS"
