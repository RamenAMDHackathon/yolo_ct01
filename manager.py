import os
import glob
import shutil
import time
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import yaml
import cv2
import torch
from ultralytics import YOLO


# -------------------------
# Device selection
# -------------------------
def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_built() and torch.backends.mps.is_available()
    if mps_ok:
        return "mps"
    return "cpu"


@dataclass
class TargetTask:
    policy_path: str
    task_text: str
    duration: int


class TaskManager:
    def __init__(self, config_path: str = "config.yaml", weights: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.config_path = config_path
        self.weights = weights
        self.conf_threshold = conf_threshold

        self.cfg = self._load_config()
        system_cfg = self.cfg.get("system", {}) or {}
        yolo_cfg = self.cfg.get("yolo", {}) or {}
        self.camera_index: int = int(system_cfg.get("camera_index", 0))
        # Environment variable ROBOT_PORT can override config
        self.robot_port: str = os.getenv("ROBOT_PORT", system_cfg.get("robot_port", "/dev/ttyACM1"))
        # Mode selection: validate (dry-run) or production
        # Priority: ENV RUN_MODE -> config.system.mode
        self.run_mode: str = os.getenv("RUN_MODE", str(system_cfg.get("mode", "validate"))).lower()
        self.dry_run: bool = self.run_mode != "production"
        self.log_file: str = str(system_cfg.get("log_file", "task_runs.log"))

        # Command defaults (overridable by env)
        self.control_bin_default: str = str(system_cfg.get("control_bin", os.getenv("CONTROL_BIN", "lerobot-record")))
        self.robot_type_default: str = str(system_cfg.get("robot_type", os.getenv("ROBOT_TYPE", "so101_follower")))
        self.extra_args_default: str = str(system_cfg.get("extra_args", os.getenv("EXTRA_ARGS", "")))

        # YOLO settings (config > env override > defaults)
        self.weights = os.getenv("WEIGHTS", str(yolo_cfg.get("weights", self.weights)))
        self.conf_threshold = float(os.getenv("CONF", yolo_cfg.get("conf", self.conf_threshold)))
        self.imgsz = int(os.getenv("IMGSZ", yolo_cfg.get("imgsz", 640)))
        self.draw_all = str(os.getenv("DRAW_ALL", str(yolo_cfg.get("draw_all", self.run_mode == "validate"))).lower()) in ("1","true","yes")

        # Homing/initial pose config
        self.homing_sleep_before_s: int = int(system_cfg.get("homing_sleep_before_s", 2))
        self.homing_sleep_after_s: int = int(system_cfg.get("homing_sleep_after_s", 2))
        self.robot_init_pose_cfg: Dict[str, float] = dict(system_cfg.get("robot_init_pose", {}))

        self.targets: Dict[str, TargetTask] = {
            k: TargetTask(
                policy_path=str(v.get("policy_path")),
                task_text=str(v.get("task_text", "")),
                duration=int(v.get("duration", 15)),
            )
            for k, v in (self.cfg.get("targets", {}) or {}).items()
        }

        self.device = get_best_device()
        print(f"[manager] Loading YOLO weights: {self.weights}")
        self.model = YOLO(self.weights)
        try:
            self.model.to(self.device)
        except Exception:
            pass
        # Class name map from model
        names = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "names"):
            names = self.model.model.names
        elif hasattr(self.model, "names"):
            names = self.model.names
        if isinstance(names, dict):
            self.class_names = {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            self.class_names = {i: n for i, n in enumerate(names)}
        else:
            self.class_names = {}

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def cleanup_eval_outputs(self):
        """Remove previous evaluation outputs to avoid collisions before task generation.

        Deletes paths matching /home/amddemo/hackathon_ramen/outputs/eval_*
        (both files and directories). Errors are logged but ignored.
        """
        pattern = "/home/amddemo/hackathon_ramen/outputs/eval_*"
        print(f"[manager] [cleanup] Removing previous outputs matching: {pattern}")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print("[manager] [cleanup] No prior outputs to remove.")
            return
        for p in paths:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                    # shutil.rmtree with ignore_errors does not raise; still log action
                    print(f"[manager] [cleanup] Removed dir: {p}")
                else:
                    os.remove(p)
                    print(f"[manager] [cleanup] Removed file: {p}")
            except Exception as e:
                print(f"[manager] [cleanup] Failed to remove {p}: {e}")

    def get_follower_initial_norm_pose(self) -> Dict[str, float]:
        """Return initial normalized joint positions for SO-101 follower arm.

        Prefer values from config.system.robot_init_pose; fallback to defaults.
        """
        if self.robot_init_pose_cfg:
            return {k: float(v) for k, v in self.robot_init_pose_cfg.items()}
        # Default pose (NORM values)
        return {
            "shoulder_pan.pos": 14.42,
            "shoulder_lift.pos": -96.63,
            "elbow_flex.pos": 91.40,
            "wrist_flex.pos": 74.78,
            "wrist_roll.pos": 1.38,
            "gripper.pos": 2.26,
        }

    def apply_initial_pose_if_possible(self):
        """Best-effort: log initial pose and call optional script if present.

        This does not fail the pipeline if the helper is missing.
        """
        pose = self.get_follower_initial_norm_pose()
        print("[manager] [robot] Initial pose (norm): " + ", ".join(f"{k}={v}" for k, v in pose.items()))
        # Sleep before homing
        try:
            if self.homing_sleep_before_s > 0:
                print(f"[manager] [robot] Sleeping {self.homing_sleep_before_s}s before homing...")
                time.sleep(self.homing_sleep_before_s)
        except Exception:
            pass
        helper = os.path.join("scripts", "apply_initial_pose.sh")
        if os.path.isfile(helper) and os.access(helper, os.X_OK):
            try:
                env = os.environ.copy()
                env["ROBOT_PORT"] = self.robot_port
                env["ROBOT_TYPE"] = os.getenv("ROBOT_TYPE", self.robot_type_default)
                # Provide pose via env for the helper if it wants to use it
                env["ROBOT_INIT_POSE"] = ";".join(f"{k}={v}" for k, v in pose.items())
                print(f"[manager] [robot] Applying initial pose via {helper} ...")
                subprocess.run([helper, env.get("ROBOT_TYPE", "so101_follower"), self.robot_port], check=False, env=env)
            except Exception as e:
                print(f"[manager] [robot] Initial pose helper failed or not applicable: {e}")
        else:
            print("[manager] [robot] No helper script found; skipping actual application (logged only).")
        # Sleep after homing
        try:
            if self.homing_sleep_after_s > 0:
                print(f"[manager] [robot] Sleeping {self.homing_sleep_after_s}s after homing...")
                time.sleep(self.homing_sleep_after_s)
        except Exception:
            pass

    def open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            sys = platform.system()
            if sys == "Darwin":
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
            elif sys == "Linux":
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        return cap

    def find_target_in_frame(self, frame) -> Optional[str]:
        """Run YOLO and return the first matched target class name if present."""
        results = self.model.predict(source=frame, verbose=False, conf=self.conf_threshold, imgsz=self.imgsz, device=self.device)
        if not results:
            return None
        res = results[0]
        if res.boxes is None or res.boxes.cls is None or res.boxes.xyxy is None:
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else []
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            cls_name = self.class_names.get(cls_id, str(cls_id))
            if cls_name in self.targets:
                return cls_name
        return None

    def detect_targets(self, frame) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Return list of (class_name, confidence, (x1,y1,x2,y2)) for configured targets only."""
        results = self.model.predict(source=frame, verbose=False, conf=self.conf_threshold, imgsz=self.imgsz, device=self.device)
        detections: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        if not results:
            return detections
        res = results[0]
        if res.boxes is None or res.boxes.cls is None or res.boxes.xyxy is None:
            return detections
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else []
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            conf = float(confs[i]) if i < len(confs) else 0.0
            cls_name = self.class_names.get(cls_id, str(cls_id))
            if cls_name not in self.targets:
                continue
            x1, y1, x2, y2 = xyxy[i]
            detections.append((cls_name, conf, (int(x1), int(y1), int(x2), int(y2))))
        return detections

    def detect_all(self, frame) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Return list of all detections (class_name, conf, bbox)."""
        results = self.model.predict(source=frame, verbose=False, conf=self.conf_threshold, imgsz=self.imgsz, device=self.device)
        detections: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        if not results:
            return detections
        res = results[0]
        if res.boxes is None or res.boxes.cls is None or res.boxes.xyxy is None:
            return detections
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else []
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            conf = float(confs[i]) if i < len(confs) else 0.0
            cls_name = self.class_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = xyxy[i]
            detections.append((cls_name, conf, (int(x1), int(y1), int(x2), int(y2))))
        return detections

    def draw_bboxes(self, frame, detections: List[Tuple[str, float, Tuple[int, int, int, int]]], color=(0,255,0)):
        for cls_name, conf, (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(y1, th + bl + 6)
            cv2.rectangle(frame, (x1, y_text - th - bl - 6), (x1 + tw + 6, y_text), color, -1)
            cv2.putText(frame, label, (x1 + 3, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    def draw_frame_with_status(self, frame, status_text: str):
        # Draw semi-transparent bar at top
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), thickness=-1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, status_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self):
        print(f"[manager] device={self.device}, camera_index={self.camera_index}, robot_port={self.robot_port}, mode={self.run_mode}")
        interval_s = int(os.getenv("TASK_INTERVAL_S", "20"))
        allowed = {k for k in self.targets.keys() if k in ("cup", "bottle")}
        while True:
            cap = self.open_camera()
            if not cap.isOpened():
                print("[manager] Failed to open camera. Retrying in 2s...")
                time.sleep(2)
                continue

            print(f"[manager] Monitoring. 20s ticker; only cup/bottle logs. Press 'q' to quit.")
            recognized_order: List[str] = []
            recognized_set = set()
            start_time = time.time()
            next_tick = start_time  # align ticks to 0s, 20s, 40s from start

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[manager] Failed to read frame. Reopening camera...")
                    break

                remain = max(0, interval_s - int(time.time() - ((next_tick - interval_s) if next_tick > start_time else start_time)))
                self.draw_frame_with_status(frame, f"Monitoring... next tick in {remain}s (q: quit)")

                # Detect configured targets
                dets = self.detect_targets(frame)

                # In debug mode, also draw all detections (yellow) to verify YOLO works
                # Optional drawing of all detections; suppress console spam
                if self.draw_all:
                    all_dets = self.detect_all(frame)
                    if all_dets:
                        self.draw_bboxes(frame, all_dets, color=(0, 255, 255))  # yellow for all

                if dets:
                    # Draw and update queue for cup/bottle only
                    self.draw_bboxes(frame, dets, color=(0, 255, 0))
                    for cls_name, conf, (x1, y1, x2, y2) in dets:
                        if cls_name in allowed and cls_name not in recognized_set:
                            recognized_set.add(cls_name)
                            recognized_order.append(cls_name)

                # Show frame and handle quit
                cv2.imshow("Manager", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                # Flush tasks every interval
                now = time.time()
                if now >= next_tick:
                    # Process at aligned tick (0,20,40s from start). Handle time drift by advancing in steps.
                    while next_tick <= now:
                        next_tick += interval_s
                    if recognized_order:
                        # Pop exactly one command per tick
                        print(f"[manager] [tick] queue={recognized_order}")
                        cls_name = recognized_order.pop(0)
                        if cls_name in recognized_set:
                            recognized_set.remove(cls_name)
                        if cls_name in allowed:
                            robot_port = self.robot_port
                            control_bin = os.getenv("CONTROL_BIN", self.control_bin_default)
                            robot_type = os.getenv("ROBOT_TYPE", self.robot_type_default)
                            extra_args = os.getenv("EXTRA_ARGS", self.extra_args_default).strip()
                            task = self.targets.get(cls_name)
                            if task:
                                policy_path = task.policy_path
                                duration = task.duration
                                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                                # コマンド生成前に必ずカメラを解放（validate / production 共通）
                                print("[manager] [camera] Releasing camera before task generation...")
                                try:
                                    cap.release()
                                    cv2.destroyAllWindows()
                                except Exception:
                                    pass
                                time.sleep(0.3)
                                print("[manager] [camera] Camera released.")

                                # 事前クリーンアップ: 過去の評価出力を削除
                                self.cleanup_eval_outputs()
                                # 初期姿勢の適用（ベストエフォート）
                                self.apply_initial_pose_if_possible()

                                try:
                                    if self.dry_run:
                                        # 検証モードでもカメラ解放後にコマンドを生成（ログ出力）
                                        dry_msg = f"[DRY-RUN {ts}] {control_bin} --robot.type={robot_type} --robot.port={robot_port} --policy.path={policy_path} {extra_args} (for {duration}s)"
                                        print(dry_msg)
                                        try:
                                            with open(self.log_file, "a") as f:
                                                f.write(dry_msg + "\n")
                                        except Exception:
                                            pass
                                    else:
                                        print(f"[manager] Executing '{cls_name}': {policy_path} for {duration}s")
                                        env = os.environ.copy()
                                        env.pop("DRY_RUN", None)
                                        env.pop("LOG_FILE", None)
                                        env["CONTROL_BIN"] = control_bin
                                        env["ROBOT_TYPE"] = robot_type
                                        env["EXTRA_ARGS"] = extra_args
                                        subprocess.run(["bash", "run_task.sh", policy_path, robot_port, str(duration)], check=True, env=env)
                                except subprocess.CalledProcessError as e:
                                    print(f"[manager] Task failed with code {e.returncode}")
                                except FileNotFoundError:
                                    print("[manager] run_task.sh not found. Ensure it is present and executable.")
                                finally:
                                    # タスク生成/実行後はカメラ再オープン（共通）
                                    print("[manager] [camera] Reopening camera after task...")
                                    try:
                                        cap = self.open_camera()
                                        if cap is not None and cap.isOpened():
                                            print("[manager] [camera] Camera reopened successfully.")
                                        else:
                                            print("[manager] [camera] Failed to reopen camera; will retry.")
                                    except Exception:
                                        print("[manager] [camera] Exception while reopening camera; will retry.")

            # Close and reopen camera on next loop
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    TaskManager().run()
