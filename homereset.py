from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        
        print("SO-101に接続中...")
        
        # 設定を作成
        config = SO101FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
        )
        
        # ロボットを初期化して接続
        robot = SO101Follower(config)
        robot.connect()