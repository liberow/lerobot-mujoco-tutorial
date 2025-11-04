#!/usr/bin/env python3
"""
语言环境：键盘采集演示数据

在包含语言指令的场景中采集演示数据。
右上/右下叠加图像来自数据集；侧视图显示在左上。
"""

# ==== 导入依赖 ====
import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env2 import SimpleEnv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ==== 配置 ====
# 若想随机化物体位置，设置为 None；固定种子则每次相同
SEED = 0
# SEED = None  # 取消注释以随机化物体位置

REPO_NAME = 'omy_pnp_language'
NUM_DEMO = 20  # 采集的演示数量
ROOT = "./demo_data_language"  # 数据集保存根目录

# ==== 创建环境 ====
xml_path = './asset/example_scene_y2.xml'
PnPEnv = SimpleEnv2(xml_path, seed=SEED, state_type='joint_angle')

# ==== 数据集特征说明（注释保留以便查阅） ====
# fps = 20,
# features={
#     "observation.image": {"dtype": "image", "shape": (256, 256, 3)},
#     "observation.wrist_image": {"dtype": "image", "shape": (256, 256, 3)},
#     "observation.state": {"dtype": "float32", "shape": (6,)},
#     "action": {"dtype": "float32", "shape": (7,)},
#     "obj_init": {"dtype": "float32", "shape": (9,)},
# }
# 生成的数据目录结构参见 notebook 注释。

# ==== 创建/加载数据集 ====
create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False

if create_new:
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=ROOT,
        robot_type="omy",
        fps=20,
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],  # x, y, z, roll, pitch, yaw
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],  # 6 关节角 + 1 夹爪
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["obj_init"],  # 物体初始位姿（语言场景）
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# ==== 键盘控制说明（中文化） ====
# XY 平面: W 后退 / S 前进 / A 左移 / D 右移
# Z 轴:    R 上升 / F 下降
# 旋转:    Q 左倾 / E 右倾 / 方向键控制俯仰与偏航
# 空格:    切换夹爪
# Z:      重置（丢弃当前回合）

# ==== 采集循环 ====
action = np.zeros(7)
episode_id = 0
record_flag = False  # 当机器人开始移动时，开始录制
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 判断回合是否完成
        done = PnPEnv.check_success()
        if done:
            # 保存并重置
            dataset.save_episode()
            PnPEnv.reset()
            episode_id += 1
        # 键盘遥操作用
        action, reset = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # 重置环境并清空缓冲
            # PnPEnv.reset(seed=SEED)
            PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # 读取图像并缩放到 256x256
        agent_image, wrist_image = PnPEnv.grab_image()
        agent_image = Image.fromarray(agent_image).resize((256, 256))
        wrist_image = Image.fromarray(wrist_image).resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        # 推进一步，并从关节状态构建 action（保持与 notebook 一致）
        joint_q = PnPEnv.step(action)
        action = PnPEnv.q[:7].astype(np.float32)
        if record_flag:
            dataset.add_frame({
                "observation.image": agent_image,
                "observation.wrist_image": wrist_image,
                "observation.state": joint_q[:6],
                "action": action,
                "obj_init": PnPEnv.obj_init_pose,
                # "task": PnPEnv.instruction,
            }, task=PnPEnv.instruction)
        PnPEnv.render(teleop=True, idx=episode_id)

# 关闭查看器
PnPEnv.env.close_viewer()

# 可选：清理 images 目录（按需启用）
# import shutil
# shutil.rmtree(dataset.root / 'images')
