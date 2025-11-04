#!/usr/bin/env python3
"""
通过键盘采集演示数据

为给定环境采集演示数据。
任务是抓取杯子并放置在盘子上。
环境识别成功的条件：杯子在盘子上、夹爪打开、末端执行器位于杯子上方。

控制方式:
  WASD - 在 XY 平面移动
  R/F  - 上下移动
  Q/E  - 倾斜旋转
  方向键 - 其他旋转
  空格键 - 切换夹爪
  Z - 重置环境（丢弃当前回合）
  ESC - 退出

叠加图像:
- 右上: 代理视角
- 右下: 自我中心视角
- 左上: 侧视图
- 左下: (本脚本未使用)
"""

import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ============================================================================
# 配置参数
# ============================================================================

# 如果想随机化物体位置，设置为 None
# 如果固定种子，每次物体位置都相同
SEED = 0 
# SEED = None  # <- 取消注释此行以随机化物体位置

REPO_NAME = 'liberow'
NUM_DEMO = 5  # 要采集的演示数量
ROOT = "./demo_data"  # 保存演示数据的根目录

TASK_NAME = 'Put mug cup on the plate' 
xml_path = './asset/example_scene_y.xml'

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 64)
    print("数据采集脚本")
    print("=" * 64)
    print(f"任务: {TASK_NAME}")
    print(f"要采集的演示数量: {NUM_DEMO}")
    print(f"数据保存路径: {ROOT}")
    print(f"随机种子: {SEED}")
    print("=" * 64)
    
    # ========================================================================
    # 步骤 1: 创建环境
    # ========================================================================
    print("\n[1/4] 正在创建环境...")
    PnPEnv = SimpleEnv(xml_path, seed=SEED, state_type='joint_angle')
    print("✓ 环境创建成功")
    
    # ========================================================================
    # 步骤 2: 创建或加载数据集
    # ========================================================================
    print("\n[2/4] 正在设置数据集...")
    create_new = True
    if os.path.exists(ROOT):
        print(f"目录 {ROOT} 已存在。")
        while True:
            ans = input("是否删除并重新开始？(y/n): ").lower().strip()
            if ans in ['y', 'n']:
                break
            print("请回答 'y' 或 'n'")
        
        if ans == 'y':
            import shutil
            shutil.rmtree(ROOT)
            print(f"✓ 已删除现有目录: {ROOT}")
        else:
            create_new = False
    
    if create_new:
        print("正在创建新数据集...")
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            root=ROOT, 
            robot_type="liber",
            fps=20,  # 每秒 20 帧
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
                    "names": ["action"],  # 6 个关节角度和 1 个夹爪状态
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["obj_init"],  # 物体的初始位置
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
        print("✓ 新数据集创建成功")
    else:
        print(f"正在从 {ROOT} 加载现有数据集...")
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)
        print("✓ 现有数据集加载成功")
    
    # ========================================================================
    # 步骤 3: 显示控制说明
    # ========================================================================
    print("\n[3/4] 键盘控制说明:")
    print("-" * 64)
    print("  移动 (XY 平面):")
    print("    W - 后退      |  S - 前进")
    print("    A - 左移      |  D - 右移")
    print()
    print("  移动 (Z 轴):")
    print("    R - 上升      |  F - 下降")
    print()
    print("  旋转:")
    print("    Q - 向左倾斜          |  E - 向右倾斜")
    print("    上方向键 - 向上看     |  下方向键 - 向下看")
    print("    左方向键 - 向左转     |  右方向键 - 向右转")
    print()
    print("  动作:")
    print("    空格键 - 切换夹爪 (开/关)")
    print("    Z - 重置环境 (丢弃当前回合)")
    print("    ESC - 退出程序")
    print("-" * 64)
    print()
    print("重要提示:")
    print("  - 当您移动机器人时，录制会自动开始")
    print("  - 要成功完成一个回合:")
    print("    1. 抓起杯子")
    print("    2. 放到盘子上")
    print("    3. 打开夹爪 (空格键)")
    print("    4. 将末端执行器向上移动到杯子上方")
    print("-" * 64)
    
    input("\n按回车键开始数据采集...")
    
    # ========================================================================
    # 步骤 4: 数据采集主循环
    # ========================================================================
    print("\n[4/4] 开始数据采集...")
    print(f"采集 {NUM_DEMO} 个演示。窗口现在已打开！")
    print("=" * 64)
    
    action = np.zeros(7)
    episode_id = 0
    record_flag = False  # 当机器人开始移动时开始录制
    
    try:
        while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
            PnPEnv.step_env()
            
            if PnPEnv.env.loop_every(HZ=20):
                # 检查回合是否完成
                done = PnPEnv.check_success()
                if done: 
                    print(f"\n✓ 回合 {episode_id + 1}/{NUM_DEMO} 成功完成！")
                    # 保存回合数据并重置环境
                    dataset.save_episode()
                    PnPEnv.reset(seed=SEED)
                    episode_id += 1
                    record_flag = False
                    
                    if episode_id < NUM_DEMO:
                        print(f"开始回合 {episode_id + 1}/{NUM_DEMO}...")
                    else:
                        print(f"\n{'=' * 64}")
                        print(f"所有 {NUM_DEMO} 个演示已采集完成！")
                        print(f"{'=' * 64}")
                        break
                
                # 通过键盘遥控机器人，获取末端执行器增量位姿和夹爪状态
                action, reset = PnPEnv.teleop_robot()
                
                if not record_flag and sum(action) != 0:
                    record_flag = True
                    print(f"\n→ 正在录制回合 {episode_id + 1}/{NUM_DEMO}...")
                
                if reset:
                    # 重置环境并清空回合缓冲区
                    # 按 'z' 键可触发
                    print("  ⟲ 收到重置请求，清空当前回合...")
                    PnPEnv.reset(seed=SEED)
                    dataset.clear_episode_buffer()
                    record_flag = False
                
                # 执行环境步进
                # 获取末端执行器位姿和图像
                ee_pose = PnPEnv.get_ee_pose()
                agent_image, wrist_image = PnPEnv.grab_image()
                
                # 调整图像大小为 256x256
                agent_image = Image.fromarray(agent_image)
                wrist_image = Image.fromarray(wrist_image)
                agent_image = agent_image.resize((256, 256))
                wrist_image = wrist_image.resize((256, 256))
                agent_image = np.array(agent_image)
                wrist_image = np.array(wrist_image)
                
                joint_q = PnPEnv.step(action)
                
                if record_flag:
                    # 将帧添加到数据集
                    dataset.add_frame({
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": ee_pose, 
                        "action": joint_q,
                        "obj_init": PnPEnv.obj_init_pose,
                    }, task=TASK_NAME)
                
                PnPEnv.render(teleop=True)
    
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断 (Ctrl+C)")
    
    finally:
        # ====================================================================
        # 清理工作
        # ====================================================================
        print("\n" + "=" * 64)
        print("正在清理...")
        PnPEnv.env.close_viewer()
        print("✓ 查看器已关闭")
        
        # 清理图像文件夹（可选，需要时取消注释）
        # import shutil
        # if os.path.exists(dataset.root / 'images'):
        #     shutil.rmtree(dataset.root / 'images')
        #     print("✓ 临时图像文件夹已删除")
        
        print(f"\n数据集已保存到: {ROOT}")
        print("数据集结构:")
        print("  ./demo_data/")
        print("    ├── data/")
        print("    │   └── chunk-000/")
        print("    │       ├── episode_000000.parquet")
        print("    │       └── ...")
        print("    └── meta/")
        print("        ├── episodes.jsonl")
        print("        ├── info.json")
        print("        ├── stats.json")
        print("        └── tasks.jsonl")
        print("\n" + "=" * 64)
        print("数据采集完成！")
        print("=" * 64)


if __name__ == "__main__":
    main()
