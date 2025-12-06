# 出力のコマンドを修正する必要あり
import sys
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import platform
from math import isnan
import numpy as np
import torch
from ultralytics import YOLO


# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("garbage_sort")


class DeviceManager:
    """Selects the best available compute device across macOS M1 (mps),
    NVIDIA (cuda), or CPU. Returns a device string suitable for Ultralytics YOLO.
    """

    @staticmethod
    def get_best_device() -> str:
        # Prefer CUDA if available (Ubuntu + NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info(f"Selected device: {device}")
            return device

        # Then Apple Silicon via Metal Performance Shaders (macOS)
        mps_ok = getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_built() and torch.backends.mps.is_available()
        if mps_ok:
            device = "mps"
            logger.info(f"Selected device: {device}")
            return device

        # Fallback to CPU
        device = "cpu"
        logger.info(f"Selected device: {device}")
        return device


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    trash_label: str
    xyz_m: Optional[Tuple[float, float, float]] = None


class GarbageDetector:
    """YOLOv8-based detector with simple trash-type classification logic.

    - Loads yolov8n.pt
    - Filters detections by confidence and target classes
    - Maps detections to trash categories (RECYCLE/BURNABLE/COMPOST)
    """

    # COCO class IDs of interest
    TARGET_CLASS_TO_TRASH: Dict[int, str] = {
        39: "RECYCLE",   # bottle
        41: "BURNABLE",  # cup
        46: "COMPOST",   # banana
        47: "COMPOST",   # apple
    }

    # Visual colors (BGR) for OpenCV drawing by trash label
    TRASH_COLORS: Dict[str, Tuple[int, int, int]] = {
        "RECYCLE": (255, 0, 0),   # Blue
        "BURNABLE": (0, 0, 255),  # Red
        "COMPOST": (0, 200, 0),   # Green
    }

    def __init__(
        self,
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
        weights: str = "yolov8n.pt",
    ) -> None:
        self.device = device or DeviceManager.get_best_device()
        self.conf_threshold = conf_threshold
        self.weights = weights

        logger.info("Loading YOLO model... this may download weights on first use")
        self.model = YOLO(self.weights)
        try:
            # Move model to target device if supported
            self.model.to(self.device)
        except Exception as e:
            # Some backends rely on specifying device at predict time; continue gracefully
            logger.debug(f"model.to({self.device}) not applied: {e}")

        # Build reverse COCO id->name map from model metadata when available
        # Ultralytics models include names in model.names
        self.class_names: Dict[int, str] = {}
        try:
            names = self.model.model.names if hasattr(self.model, "model") else getattr(self.model, "names", None)
            if isinstance(names, dict):
                self.class_names = {int(k): v for k, v in names.items()}
            elif isinstance(names, list):
                self.class_names = {i: n for i, n in enumerate(names)}
        except Exception as e:
            logger.debug(f"Failed to read class names from model: {e}")

        # Cache target set for quick lookup
        self.target_class_ids = set(self.TARGET_CLASS_TO_TRASH.keys())

    def detect(self, frame_bgr) -> List[Detection]:
        """Run inference and return filtered detections.

        Returns a list of Detection with class name, confidence, bbox, and trash label.
        """
        if frame_bgr is None:
            return []

        results = self.model.predict(
            source=frame_bgr,
            verbose=False,
            conf=self.conf_threshold,
            device=self.device,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None:
            return detections

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

        for i in range(len(xyxy)):
            cls_id = int(clss[i]) if i < len(clss) else -1
            conf = float(confs[i]) if i < len(confs) else 0.0
            if cls_id not in self.target_class_ids:
                continue

            x1, y1, x2, y2 = xyxy[i]
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            class_name = self.class_names.get(cls_id, str(cls_id))
            trash_label = self.TARGET_CLASS_TO_TRASH.get(cls_id, "UNKNOWN")

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=(x1i, y1i, x2i, y2i),
                    trash_label=trash_label,
                )
            )

        return detections

    def draw(self, frame_bgr, detections: List[Detection]):
        """Draw bounding boxes and labels on the frame in-place."""
        for det in detections:
            color = self.TRASH_COLORS.get(det.trash_label, (128, 128, 128))
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f} | {det.trash_label}"
            if det.xyz_m is not None:
                x, y, z = det.xyz_m
                label += f" | X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"

            # Draw filled label background for readability
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            th_total = th + baseline + 4
            y_text = max(y1, th_total)
            cv2.rectangle(frame_bgr, (x1, y_text - th_total), (x1 + tw + 6, y_text), color, thickness=-1)
            cv2.putText(
                frame_bgr,
                label,
                (x1 + 3, y_text - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Garbage Sort - Real-time detection with simple trash mapping.")
    parser.add_argument("--camera-id", "-c", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--conf", "-t", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--weights", "-w", type=str, default="yolov8n.pt", help="Path to YOLO weights (default: yolov8n.pt)")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Device override: auto|cpu|mps|cuda|cuda:0 ... (default: auto)")
    parser.add_argument("--window-name", type=str, default=None, help="OpenCV window name (default shows device and conf)")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices and exit")
    parser.add_argument("--max-devices", type=int, default=10, help="Max camera indices to probe when listing (default: 10)")
    # OAK-D options
    parser.add_argument("--use-oakd", action="store_true", help="Use OAK-D Lite (DepthAI) pipeline for camera and depth")
    parser.add_argument("--oak-width", type=int, default=640, help="OAK-D RGB width (default: 640)")
    parser.add_argument("--oak-height", type=int, default=480, help="OAK-D RGB height (default: 480)")
    parser.add_argument("--depth-ksize", type=int, default=5, help="Kernel size for median depth at bbox center (odd, default: 5)")
    return parser.parse_args()


def list_cameras(max_devices: int = 10) -> None:
    sys_name = platform.system()
    found = []
    logger.info(f"Probing camera indices 0..{max_devices-1} (OS={sys_name})")
    for cam_id in range(max_devices):
        opened = False
        backend = "default"
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            # Try platform-specific backend
            if sys_name == "Darwin":
                cap = cv2.VideoCapture(cam_id, cv2.CAP_AVFOUNDATION)
                backend = "AVFOUNDATION"
            elif sys_name == "Linux":
                cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
                backend = "V4L2"
        if cap.isOpened():
            # Try a quick read to ensure it streams
            ret, _ = cap.read()
            if ret:
                opened = True
        cap.release()
        if opened:
            logger.info(f"Camera index {cam_id} available (backend={backend})")
            found.append(cam_id)
    if not found:
        logger.warning("No working camera indices found. Check USB connection and permissions.")
    else:
        logger.info(f"Usable camera indices: {found}")


class OAKDCamera:
    """DepthAI OAK-D Lite helper to stream RGB and aligned depth, with intrinsics.

    Provides read() -> (rgb_bgr, depth_mm) and intrinsics for projection.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        try:
            import depthai as dai  # type: ignore
        except Exception as e:
            logger.error("depthai is not installed. Install with 'pip install depthai'.")
            raise

        self.dai = dai
        self.width = int(width)
        self.height = int(height)

        pipeline = dai.Pipeline()

        color = pipeline.create(dai.node.ColorCamera)
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setPreviewSize(self.width, self.height)
        color.setInterleaved(False)
        color.setFps(30)

        monoL = pipeline.create(dai.node.MonoCamera)
        monoR = pipeline.create(dai.node.MonoCamera)
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        depth = pipeline.create(dai.node.StereoDepth)
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.setDepthAlign(dai.CameraBoardSocket.RGB)
        depth.setSubpixel(False)
        depth.setLeftRightCheck(True)
        depth.setExtendedDisparity(False)

        monoL.out.link(depth.left)
        monoR.out.link(depth.right)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        color.preview.link(xout_rgb.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        depth.depth.link(xout_depth.input)

        self.device = dai.Device(pipeline)
        self.q_rgb = self.device.getOutputQueue("rgb", 1, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", 1, blocking=False)

        calib = self.device.readCalibration()
        # Intrinsics scaled to our preview size
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self.width, self.height)
        # K is 3x3
        self.fx = float(K[0][0])
        self.fy = float(K[1][1])
        self.cx = float(K[0][2])
        self.cy = float(K[1][2])

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rgb_msg = self.q_rgb.get() if self.q_rgb.has() else None
        depth_msg = self.q_depth.get() if self.q_depth.has() else None
        rgb = rgb_msg.getCvFrame() if rgb_msg is not None else None
        depth = depth_msg.getFrame() if depth_msg is not None else None  # uint16 in mm
        return rgb, depth

    def close(self):
        if self.device is not None:
            self.device.close()

    def project_pixel_to_3d(self, u: int, v: int, z_m: float) -> Tuple[float, float, float]:
        x = (u - self.cx) / self.fx * z_m
        y = (v - self.cy) / self.fy * z_m
        return x, y, z_m


def main():
    args = parse_args()
    if args.list_cameras:
        list_cameras(args.max_devices)
        return
    device = DeviceManager.get_best_device() if args.device == "auto" else args.device
    detector = GarbageDetector(device=device, conf_threshold=args.conf, weights=args.weights)

    # If camera-id is 1 (reserved for OAK-D Lite), enable OAK-D automatically unless overridden
    use_oakd = args.use_oakd or (args.camera_id == 1)
    oak = None
    if use_oakd:
        try:
            oak = OAKDCamera(width=args.oak_width, height=args.oak_height)
            logger.info("OAK-D Lite pipeline started (RGB+Depth aligned)")
        except Exception:
            logger.error("Failed to start OAK-D pipeline. Falling back to standard camera.")
            use_oakd = False

    if not use_oakd:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            # Retry with platform-specific backends
            sys_name = platform.system()
            logger.warning("Default camera backend failed. Retrying with platform-specific backend...")
            if sys_name == "Darwin":
                cap = cv2.VideoCapture(args.camera_id, cv2.CAP_AVFOUNDATION)
            elif sys_name == "Linux":
                cap = cv2.VideoCapture(args.camera_id, cv2.CAP_V4L2)
            # Re-check
            if not cap.isOpened():
                logger.error(f"Failed to open camera (ID {args.camera_id}) with fallback backend.")
                sys.exit(1)

    logger.info("Press 'q' to quit.")
    while True:
        if use_oakd and oak is not None:
            frame, depth = oak.read()
            if frame is None or depth is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            detections = detector.detect(frame)

            # Ensure depth matches frame size for sampling
            if depth.shape[:2] != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Compute XYZ for each detection at bbox center using median Z in a kxk window
            k = args.depth_ksize
            if k % 2 == 0:
                k += 1
            half = k // 2
            h, w = depth.shape[:2]
            for det in detections:
                x1, y1, x2, y2 = det.bbox_xyxy
                cx = max(0, min((x1 + x2) // 2, w - 1))
                cy = max(0, min((y1 + y2) // 2, h - 1))
                x0 = max(0, cx - half)
                y0 = max(0, cy - half)
                x1w = min(w, cx + half + 1)
                y1w = min(h, cy + half + 1)
                roi = depth[y0:y1w, x0:x1w]
                if roi.size == 0:
                    continue
                z_mm = int(np.median(roi))
                if z_mm <= 0:
                    continue
                z_m = z_mm / 1000.0
                if z_m > 10 or isnan(z_m):
                    continue
                try:
                    x_m, y_m, z_m = oak.project_pixel_to_3d(int(cx), int(cy), float(z_m))
                    det.xyz_m = (x_m, y_m, z_m)
                except Exception:
                    pass

            detector.draw(frame, detections)

            win_name = args.window_name or f"YOLOv8 Garbage Sort [{device}] conf={args.conf} OAK-D"
            cv2.imshow(win_name, frame)
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera.")
                break

            detections = detector.detect(frame)
            detector.draw(frame, detections)

            # Optionally visualize target zones or UI hints in the future
            win_name = args.window_name or f"YOLOv8 Garbage Sort [{device}] conf={args.conf}"
            cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if use_oakd and oak is not None:
        try:
            oak.close()
        except Exception:
            pass
    else:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
