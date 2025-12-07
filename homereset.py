from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        
print("SO-101に接続中...")

# 險ｭ螳壹ｒ菴懈�
config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="my_awesome_follower_arm",
)

# ロボットを初期化して接続
robot = SO101Follower(config)
robot.connect()

HOME={
    "shoulder_pan.pos": 14.42,
    "shoulder_lift.pos": -96.63,
    "elbow_flex.pos": 91.40,
    "wrist_flex.pos": 74.78,
    "wrist_roll.pos": 1.38,
    "gripper.pos": 2.26,
}

robot.send_action(HOME)