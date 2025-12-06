import os
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
        self.robot_port: str = os.getenv("ROBOT_PORT", system_cfg.get("robot_port", "/dev/ttyACM0"))
        # Mode selection: validate (dry-run) or production
        # Priority: ENV RUN_MODE -> config.system.mode
        self.run_mode: str = os.getenv("RUN_MODE", str(system_cfg.get("mode", "validate"))).lower()
        self.dry_run: bool = self.run_mode != "production"
        self.log_file: str = str(system_cfg.get("log_file", "task_runs.log"))

        # YOLO settings (config > env override > defaults)
        self.weights = os.getenv("WEIGHTS", str(yolo_cfg.get("weights", self.weights)))
        self.conf_threshold = float(os.getenv("CONF", yolo_cfg.get("conf", self.conf_threshold)))
        self.imgsz = int(os.getenv("IMGSZ", yolo_cfg.get("imgsz", 640)))
        self.draw_all = str(os.getenv("DRAW_ALL", str(yolo_cfg.get("draw_all", self.run_mode == "validate"))).lower()) in ("1","true","yes")

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
        while True:
            cap = self.open_camera()
            if not cap.isOpened():
                print("[manager] Failed to open camera. Retrying in 2s...")
                time.sleep(2)
                continue

            print("[manager] Monitoring camera. Press 'q' to quit.")
            matched: Optional[str] = None
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[manager] Failed to read frame. Reopening camera...")
                    break

                self.draw_frame_with_status(frame, "Monitoring... (press q to quit)")

                # Detect configured targets
                dets = self.detect_targets(frame)

                # In debug mode, also draw all detections (yellow) to verify YOLO works
                if self.draw_all:
                    all_dets = self.detect_all(frame)
                    if all_dets:
                        print(f"[manager] YOLO detections: {len(all_dets)} -> {[d[0] for d in all_dets[:5]]}{'...' if len(all_dets)>5 else ''}")
                        self.draw_bboxes(frame, all_dets, color=(0, 255, 255))  # yellow for all

                if dets:
                    # Log target detections to stdout
                    for cls_name, conf, (x1, y1, x2, y2) in dets:
                        print(f"[manager] Detected target: {cls_name} conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})")
                    matched = dets[0][0]  # choose first target class

                    # Draw target boxes in green
                    self.draw_bboxes(frame, dets, color=(0, 255, 0))
                    cv2.putText(frame, f"Target detected: {matched}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Manager", frame)
                    cv2.waitKey(1)
                    time.sleep(1.0)
                    break

                if not dets and self.draw_all:
                    # If no targets but have general detections, still show the frame
                    cv2.imshow("Manager", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    continue

                cv2.imshow("Manager", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Close camera before launching robot task (to avoid device contention)
            cap.release()
            cv2.destroyAllWindows()

            if matched is None:
                # reopen camera loop
                continue

            # Run associated task
            task = self.targets[matched]
            policy_path = task.policy_path
            duration = task.duration
            robot_port = self.robot_port

            print(f"[manager] Executing task for '{matched}': policy={policy_path}, port={robot_port}, duration={duration}s")
            try:
                env = os.environ.copy()
                # In validate mode (mac mini), force dry-run and set log file
                if self.dry_run:
                    env["DRY_RUN"] = "1"
                    env["LOG_FILE"] = self.log_file
                # Use bash to ensure script is executed properly
                subprocess.run([
                    "bash", "run_task.sh", policy_path, robot_port, str(duration)
                ], check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"[manager] Task failed with code {e.returncode}")
            except FileNotFoundError:
                print("[manager] run_task.sh not found. Ensure it is present and executable.")

            # After completion, loop continues to reopen the camera


if __name__ == "__main__":
    TaskManager().run()
