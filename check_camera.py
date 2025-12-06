import cv2

def list_cameras(max_check=5):
    print("カメラを検索しています...")
    available_cameras = []

    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[OK] Camera ID {i}: 接続成功 ({int(cap.get(3))}x{int(cap.get(4))})")
                available_cameras.append(i)
            else:
                print(f"[NG] Camera ID {i}: 開けましたが映像が取得できません (権限エラーの可能性あり)")
            cap.release()
        else:
            print(f"[--] Camera ID {i}: 見つかりません")
    
    print("\n--- 結果 ---")
    if available_cameras:
        print(f"使えるカメラID: {available_cameras}")
        print(f"コマンド例: python garbage_sort_yolo.py -c {available_cameras[0]}")
    else:
        print("有効なカメラが見つかりませんでした。接続や権限を確認してください。")

if __name__ == "__main__":
    list_cameras()