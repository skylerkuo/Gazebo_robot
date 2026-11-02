#!/usr/bin/env python3
"""Check actual simulator evidence and official engine results after a demo."""
import argparse
import json
from pathlib import Path
import re
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", help="Run directory; defaults to the latest run")
    parser.add_argument("--require-insertion", action="store_true", help="Require successful insertion in every trial (automatic for CheatCode runs)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run) if args.run else root / "results" / (root / "runtime/latest-run.txt").read_text().strip()
    evidence = json.loads((run / "evidence.json").read_text())
    scoring = yaml.safe_load((run / "scoring/scoring.yaml").read_text())
    trials = {name: result for name, result in scoring.items() if name != "total"}
    require_insertion = args.require_insertion or "_CheatCode_" in run.name
    insertions = {
        name: result["tier_3"].get("message") == "Cable insertion successful." and result["tier_3"]["score"] > 0
        for name, result in trials.items()
    }
    checks = {
        "evaluator_exit_zero": (run / "evaluator-exit-code.txt").read_text().strip() == "0",
        "simulation_clock_received": evidence["clock_messages"] > 10,
        "joint_feedback_received": evidence["joint_messages"] > 10,
        "synchronized_observations_received": evidence["observations"] > 10,
        "pose_commands_received": evidence["pose_commands"] > 10,
        "arm_moved": max(evidence["joint_range_radians"].values(), default=0) > 0.1,
        "all_camera_previews_saved": all((run / (name + "_camera.png")).is_file() for name in ("left", "center", "right")),
        "three_trials_scored": len(trials) == 3,
        "all_trials_model_valid": bool(trials) and all(result["tier_1"]["score"] == 1 for result in trials.values()),
    }
    if require_insertion:
        checks["all_trials_inserted_correctly"] = bool(insertions) and all(insertions.values())
    launch_log = (run / "eval.log").read_text(errors="replace")
    shutdown_issues = {
        "simulation_container_aborted_during_shutdown": bool(re.search(r"component_container.*exit code -6", launch_log)),
        "gui_required_sigkill": bool(re.search(r"gz-\d+.*failed to terminate.*SIGTERM", launch_log)),
    }
    report = {
        "passed": all(checks.values()), "checks": checks,
        "scope": "Official cable insertion reference policy" if require_insertion else "Official simulator, observations, robot control and scoring pipeline",
        "insertion_success_by_trial": insertions,
        "correct_insertions": sum(insertions.values()),
        "shutdown_clean": not any(shutdown_issues.values()),
        "shutdown_issues": shutdown_issues,
        "trial_scores": trials, "official_total": scoring["total"],
        "evidence": evidence,
    }
    (run / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "evidence"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
