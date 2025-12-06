# ロボットアーム仕分けシステム仕様書

## 1. システム構成
- **Manager (Python)**: カメラ映像を監視し、対象物を見つけたらロボットの実行スクリプトを呼び出す。
- **Worker (Shell Script)**: 指定された学習済みモデルをロードして、一定時間ロボットを制御する。

## 2. ファイル別要件

### A. `config.yaml` (設定ファイル)
以下の構造を持つ設定ファイルを作成する。
- **system**:
    - `camera_index`: 0 (またはOAK-Dの設定)
    - `robot_port`: "/dev/ttyACM0" (環境変数で上書き可能に)
- **targets**: ゴミのラベル（YOLOのクラス名）をキーとする辞書。
    - 各エントリには以下を含める:
        - `policy_path`: モデルのパス (例: "user/so100_bottle_policy")
        - `task_text`: テキスト指示 (例: "Pick up the bottle")
        - `duration`: 実行時間（秒） (例: 15)

### B. `run_task.sh` (実行スクリプト)
LeRobotの `lerobot-control` コマンドをラップするシェルスクリプト。
- **引数**:
    1. `POLICY_PATH`: モデルへのパス
    2. `ROBOT_PORT`: ロボットのUSBポート
    3. `DURATION`: 実行時間（秒）
- **機能**:
    - Linuxの `timeout` コマンドを使用して、指定された `DURATION` 秒後に強制終了するようにする。
    - **コマンド構成例**:
      ```bash
      timeout $DURATION lerobot-control \
        --robot.type=so100_follower \
        --robot.port=$ROBOT_PORT \
        --policy.path=$POLICY_PATH \
        --fps 30
      ```
      (※カメラ設定は引数化せず、スクリプト内にハードコードで良い)

### C. `manager.py` (司令塔)
- **ライブラリ**: `ultralytics` (YOLO), `pyyaml`, `subprocess`
- **処理フロー**:
    1. `config.yaml` を読み込む。
    2. カメラを起動し、リアルタイムで推論を行う。
    3. 設定ファイルにあるターゲット（例: bottle）が検出された場合:
        - バウンディングボックスを描画して画面表示。
        - ユーザーが視認できるように1秒ほど待機（任意）。
        - カメラを開放する（重要: ロボット制御側とカメラが競合しないため）。
        - `subprocess.run` で `run_task.sh` を呼び出す。
        - 実行が終わったら、再度カメラを接続して監視に戻る。

## 3. 環境差異への配慮
- 開発はMac、本番はUbuntuで行う。
- `run_task.sh` はUbuntu環境（GPUあり）での動作を前提とするが、コマンドが見つからない場合はダミーのecho出力で動作確認できるようにエラーハンドリングを入れること。