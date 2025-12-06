# ゴミ仕分け・ロボット連携リポジトリ

本リポジトリは、YOLOv8 によるリアルタイム物体検出と、ロボットアーム（SO-100 / LeRobot）制御を疎結合で連携するためのサンプル実装です。

- 物体検出のみ: `garbage_sort_yolo.py`
- OAK-D Lite（深度）検出: `--use-oakd` または `run_3d_oakd.sh`
- タスク管理（検出→ロボット実行）: `manager.py` + `run_task.sh` + `config.yaml`

---

## セットアップ

- Python 3.10+（Mac M1: 3.11 / Ubuntu: 3.10 を想定）
- 仮想環境（.venv）を利用

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python torch depthai numpy pillow pyyaml tqdm pandas matplotlib seaborn psutil
```

- 学習済み重み
  - `yolov8n.pt` をルートに置く（初回は自動ダウンロードも可）。

---

## カメラ検出（番号の把握）

- OpenCV 経由でスキャン:
```
.venv/bin/python garbage_sort_yolo.py --list-cameras
```
- OSごとの補助コマンド
  - macOS: `ffmpeg -f avfoundation -list_devices true -i ""`
  - Linux: `ls -l /dev/video*`, `v4l2-ctl --list-devices`

---

## 物体検出（単体）

- 基本（カメラ0、しきい値0.5、ローカル重み）
```
.venv/bin/python garbage_sort_yolo.py -c 0 -t 0.5 -w yolov8n.pt
```
- デバイス指定（自動: `cuda→mps→cpu`）
```
.venv/bin/python garbage_sort_yolo.py -d auto   # 既定
.venv/bin/python garbage_sort_yolo.py -d mps    # Mac M1
.venv/bin/python garbage_sort_yolo.py -d cuda   # Ubuntu + NVIDIA
```
- OAK-D LITE（深度＋3D座標表示）
```
# スクリプト（既定: CAMERA_ID=1）
bash run_3d_oakd.sh
# 直接指定
.venv/bin/python garbage_sort_yolo.py --use-oakd -c 1 -t 0.5 -w yolov8n.pt
```

---

## タスク管理（検出→ロボット実行）

構成:
- `config.yaml`: システム全体設定とターゲット別のロボット実行設定
- `manager.py`: カメラ監視→YOLO推論→検出時に `run_task.sh` を呼び出し
- `run_task.sh`: `lerobot-control` を `timeout` 付きで実行（見つからなければ DRY-RUN）

### 設定
`config.yaml`（抜粋）
```
system:
  camera_index: 0
  robot_port: "/dev/ttyACM0"   # 環境変数 ROBOT_PORT で上書き可
  mode: "validate"              # validate(検証)/production(本番)
  log_file: "task_runs.log"     # ドライラン時のログ

yolo:
  weights: "yolov8n.pt"
  conf: 0.35
  imgsz: 640
  draw_all: true

targets:
  bottle:
    policy_path: "user/so100_bottle_policy"
    task_text: "Pick up the bottle and place it in recycle bin"
    duration: 15
  # cup / banana / apple なども同様
```

### 実行（検証／ドライラン：Mac mini）
- カメラ0で監視、検出があれば緑枠で表示＋標準ログ出力、1秒待機後に `run_task.sh` をDRY-RUN実行（`task_runs.log`に追記）
```
RUN_MODE=validate .venv/bin/python manager.py
# ログ監視
tail -f task_runs.log
```
- 検出が弱い場合の調整（環境変数で上書き）
```
CONF=0.25 IMGSZ=960 RUN_MODE=validate .venv/bin/python manager.py
WEIGHTS=yolov8s.pt CONF=0.30 IMGSZ=960 RUN_MODE=validate .venv/bin/python manager.py
```

### 実行（本番：Ubuntu + NVIDIA）
- 実機制御を有効化（RUN_MODE=production）
```
export ROBOT_PORT=/dev/ttyACM0
RUN_MODE=production .venv/bin/python manager.py
```
- `run_task.sh` が `timeout` 付きで `lerobot-control` を起動
  - 例: `timeout 15 lerobot-control --robot.type=so100_follower --robot.port=/dev/ttyACM0 --policy.path=user/so100_bottle_policy --fps 30`

---

## ログと可視化
- `manager.py`（検出時）
  - 標準出力: `[manager] Detected target: bottle conf=0.87 bbox=(122,45,250,300)`
  - 画面: 緑枠＋ラベル（検証モードでは他クラスの検出も黄色枠で重畳）
- ドライラン（検証）
  - `task_runs.log` に実行予定コマンドを追記

---

## トラブルシュート
- カメラが開けない
  - macOS: AVFoundation、Linux: V4L2 バックエンドで自動再試行
  - Linux: `sudo usermod -a -G video $USER`（再ログイン）
- OAK-D が動かない
  - `pip install depthai`
  - Ubuntu: udev ルール/plugdev グループ設定が必要な場合あり（Luxonisの手順参照）
- GPU で遅い/使われない
  - CUDA 対応の PyTorch を導入（`pip` ではなく PyTorch の専用 index を利用）
- `lerobot-control` が見つからない
  - 検証モード/DRY-RUNにフォールバック（実行せずログのみ）。本番環境で PATH を整備。

---

## ファイル一覧
- `garbage_sort_yolo.py` … YOLOv8リアルタイム検出（対象クラスの色分け表示、OAK-D対応）
- `run_3d_oakd.sh` … OAK-D 3D座標表示の起動スクリプト
- `manager.py` … 検出→ロボット実行の司令塔（検証/本番切替、ログ出力、枠描画）
- `run_task.sh` … `lerobot-control` のラッパー（timeout / DRY-RUN ログ）
- `config.yaml` … システム設定とターゲット別実行設定

---

## メモ（チューニングの目安）
- 小物・俯瞰配置で検出が弱い → `CONF` を下げる（0.25〜0.35）、`IMGSZ` を上げる（960〜1280）、`yolov8s.pt` を試す
- 処理が重い → `IMGSZ` を下げる、`yolov8n.pt` に戻す、GPU を使う（`-d cuda`）
- 検出の可視化を抑える → `yolo.draw_all: false` または `DRAW_ALL=0`

