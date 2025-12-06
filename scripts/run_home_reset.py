#!/usr/bin/env python3
import argparse
import os
import sys
import time
import subprocess
from typing import Dict, List, Optional

try:
    import yaml
except Exception as e:
    print(f"[run_home_reset] Failed to import yaml (PyYAML required): {e}")
    sys.exit(2)


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_deg_list_from_config(cfg: Dict) -> Optional[List[float]]:
    system = (cfg.get("system") or {})
    pose: Dict[str, float] = (system.get("robot_init_pose") or {})
    keys = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    if not pose:
        return None
    vals: List[float] = []
    for k in keys:
        if k not in pose:
            print(f"[run_home_reset] Missing init pose key in config: {k}")
            return None
        try:
            vals.append(float(pose[k]))
        except Exception as e:
            print(f"[run_home_reset] Invalid numeric value for {k}: {pose[k]} ({e})")
            return None
    return vals


def main() -> int:
    p = argparse.ArgumentParser(description="Run homereset.py using values from config.yaml.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: ./config.yaml)")
    p.add_argument("--script-path", default=None, help="Path to homereset.py (default: from config system.home_reset_script or ./homereset.py)")
    p.add_argument("--port", default=None, help="Robot port override (default: from config system.robot_port)")
    p.add_argument("--type", dest="robot_type", default=None, help="Robot type override (default: from config system.robot_type or env ROBOT_TYPE)")
    p.add_argument("--deg", default=None, help="Comma-separated degrees list (6 joints). If omitted, use config system.robot_init_pose")
    p.add_argument("--duration", default=None, help="Motion duration seconds (default: from config system.home_reset_duration_s or 4)")
    p.add_argument("--sleep-before", type=float, default=None, help="Sleep seconds before executing homereset (default: system.homing_sleep_before_s)")
    p.add_argument("--sleep-after", type=float, default=None, help="Sleep seconds after executing homereset (default: system.homing_sleep_after_s)")
    args = p.parse_args()

    cfg = load_config(args.config)
    system = (cfg.get("system") or {})

    # Resolve script
    script = args.script_path or system.get("home_reset_script") or "homereset.py"
    script_path = script if os.path.isabs(script) else os.path.join(os.getcwd(), script)
    if not os.path.isfile(script_path):
        print(f"[run_home_reset] homereset.py not found: {script_path}")
        return 1

    # Resolve port/type
    port = args.port or system.get("robot_port", "/dev/ttyACM1")
    robot_type = args.robot_type or os.getenv("ROBOT_TYPE", system.get("robot_type", "so101_follower"))

    # Resolve degrees list
    deg_str = args.deg
    if deg_str is None:
        vals = build_deg_list_from_config(cfg)
        if vals is None:
            print("[run_home_reset] Could not build --deg from config; please pass --deg explicitly.")
            return 1
        deg_str = ",".join(str(v) for v in vals)

    # Resolve duration and sleeps
    duration = args.duration or str(system.get("home_reset_duration_s", 4))
    sleep_before = args.sleep_before if args.sleep_before is not None else float(system.get("homing_sleep_before_s", 0))
    sleep_after = args.sleep_after if args.sleep_after is not None else float(system.get("homing_sleep_after_s", 0))

    print(f"[run_home_reset] Using config: {args.config}")
    print(f"[run_home_reset] Script: {script_path}")
    print(f"[run_home_reset] Port: {port}  Type: {robot_type}")
    print(f"[run_home_reset] Deg: {deg_str}  Duration: {duration}s  Sleep(before/after): {sleep_before}/{sleep_after}")

    if sleep_before > 0:
        print(f"[run_home_reset] Sleeping {sleep_before}s before homereset...")
        time.sleep(sleep_before)

    cmd = [
        sys.executable,
        script_path,
        "--port", str(port),
        "--type", str(robot_type),
        "--deg", str(deg_str),
        "--duration", str(duration),
    ]
    print("[run_home_reset] Exec: " + " ".join(cmd))
    try:
        # Let homereset.py control its own exit code; we just relay it
        proc = subprocess.run(cmd)
        rc = int(proc.returncode or 0)
    except Exception as e:
        print(f"[run_home_reset] Failed to execute homereset.py: {e}")
        rc = 2

    if sleep_after > 0:
        print(f"[run_home_reset] Sleeping {sleep_after}s after homereset...")
        time.sleep(sleep_after)

    print(f"[run_home_reset] Done with exit code {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

