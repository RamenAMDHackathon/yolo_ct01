# ゴミ仕分け・ロボット連携リポジトリ

本リポジトリは、YOLOv8 によるリアルタイム物体検出と、ロボットアーム（SO-100 / LeRobot）制御を疎結合で連携するためのサンプル実装です。

- 物体検出のみ: `garbage_sort_yolo.py`
- タスク管理（検出→ロボット実行）: `manager.py` + `run_task.sh` + `config.yaml`

---

## セットアップ

- Python 3.10+（Mac M1: 3.11 / Ubuntu: 3.10 を想定）
- 仮想環境（.venv）を利用

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python torch numpy pillow pyyaml tqdm pandas matplotlib seaborn psutil
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
---

## タスク管理（検出→コマンド生成/実行）

- `config.yaml`: システム設定とターゲット別（cup/bottle）の `policy_path`、実行既定（control_bin/extra_args）を定義
- `manager.py`: カメラ監視→YOLO推論→20秒ごとにタスクを1件ずつフラッシュ
  - 検証: DRY-RUN 行を `task_runs.log` に追記（実行しない）
  - 本番: `run_task.sh` を呼び出して実行
- `run_task.sh`: `CONTROL_BIN`（既定: lerobot-record）等を使ってコマンドを実行／DRY-RUN記録

### cup/bottle → policy 切り替え
- cup 検出時: `--policy.path=AmdRamen/mission2_cup`
- bottle 検出時: `--policy.path=AmdRamen/mission2_bottle`

### 設定（抜粋）
```
system:
  camera_index: 0
  robot_port: "/dev/ttyACM1"
  mode: "validate"              # validate(検証)/production(本番)
  log_file: "task_runs.log"
  control_bin: "lerobot-record" # 既定の実行バイナリ
  robot_type: "so101_follower"
  extra_args: "--robot.id=my_awesome_follower_arm --robot.cameras=\"{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}\" --dataset.single_task=\"put the pen in the white square\" --dataset.repo_id=AmdRamen/eval_mission_1 --dataset.root=/home/amddemo/hackathon_ramen/outputs/eval_lerobot_dataset/ --dataset.episode_time_s=20 --dataset.num_episodes=5 --dataset.push_to_hub=false"

targets:
  cup:
    policy_path: "AmdRamen/mission2_cup"
    duration: 12
  bottle:
    policy_path: "AmdRamen/mission2_bottle"
    duration: 15
```

### 実行（検証／Mac mini）
- 20秒刻み(0/20/40s…)で、cup/bottle のキューから1件だけ DRY-RUN を出力（`task_runs.log` 追記）
```
CONF=0.5 RUN_MODE=validate .venv/bin/python manager.py
tail -f task_runs.log

CONF=0.5 RUN_MODE=production .venv/bin/python manager.py
```

#### 引数（環境変数）の説明
- CONF: YOLO の信頼度しきい値（例: `0.5`。小さいほど検出しやすい）
- RUN_MODE: `validate`（ログのみ）/ `production`（実行する）
- TASK_INTERVAL_S: コマンドを出す間隔（秒）。既定 `20`
- ROBOT_PORT: ロボットのポート。既定は `config.yaml` の `/dev/ttyACM1`
- CONTROL_BIN: 既定の実行バイナリ。既定 `lerobot-record`
- ROBOT_TYPE: ロボット種別。既定 `so101_follower`
- EXTRA_ARGS: カメラ/データセット等の追加引数（config.yaml に既定を定義済み）
- 出力例（log.txt より）
```
[DRY-RUN 2025-12-06 16:49:08] lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM1 --policy.path=AmdRamen/misson2_cup --fps 30 --robot.id=my_awesome_follower_arm --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" --dataset.single_task="put the pen in the white square" --dataset.repo_id=AmdRamen/eval_mission_1 --dataset.root=/home/amddemo/hackathon_ramen/outputs/eval_lerobot_dataset/ --dataset.episode_time_s=20 --dataset.num_episodes=5 --dataset.push_to_hub=false (for 12s)
[DRY-RUN 2025-12-06 16:49:26] lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM1 --policy.path=AmdRamen/misshon2_bottle --fps 30 --robot.id=my_awesome_follower_arm --robot.cameras="{top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" --dataset.single_task="put the pen in the white square" --dataset.repo_id=AmdRamen/eval_mission_1 --dataset.root=/home/amddemo/hackathon_ramen/outputs/eval_lerobot_dataset/ --dataset.episode_time_s=20 --dataset.num_episodes=5 --dataset.push_to_hub=false (for 15s)
```

### 実行（本番／Ubuntu + NVIDIA）
- 実機制御を有効化（RUN_MODE=production）。既定の `control_bin=lerobot-record` と `extra_args` が使われます。
```
# 必要に応じてポートを上書き
# export ROBOT_PORT=/dev/ttyACM1
CONF=0.5 RUN_MODE=production .venv/bin/python manager.py
```
※ 必要に応じて `CONTROL_BIN`, `ROBOT_TYPE`, `EXTRA_ARGS`, `TASK_INTERVAL_S` を環境変数で上書き可能です。

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
- GPU で遅い/使われない
  - CUDA 対応の PyTorch を導入（`pip` ではなく PyTorch の専用 index を利用）
- `lerobot-control` が見つからない
  - 検証モード/DRY-RUNにフォールバック（実行せずログのみ）。本番環境で PATH を整備。

---

## ファイル一覧
- `garbage_sort_yolo.py` … YOLOv8リアルタイム検出（対象クラスの色分け表示）
- `manager.py` … 検出→ロボット実行の司令塔（検証/本番切替、ログ出力、枠描画）
- `run_task.sh` … LeRobot 実行ラッパー（`CONTROL_BIN` で制御、timeout / DRY-RUN ログ）
- `config.yaml` … システム設定とターゲット別実行設定

---

## メモ（チューニングの目安）
- 小物・俯瞰配置で検出が弱い → `CONF` を下げる（0.25〜0.35）、`IMGSZ` を上げる（960〜1280）、`yolov8s.pt` を試す
- 処理が重い → `IMGSZ` を下げる、`yolov8n.pt` に戻す、GPU を使う（`-d cuda`）
- 検出の可視化を抑える → `yolo.draw_all: false` または `DRAW_ALL=0`
