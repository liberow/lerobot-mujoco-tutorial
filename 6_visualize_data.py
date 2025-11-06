#!/usr/bin/env python3
"""
可视化语言数据集

在模拟环境中回放带语言指令的数据集。
主窗口回放动作，右上/右下叠加来自数据集的图像。
"""

# ==== 从 Hugging Face 下载数据集 ====
# git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language

import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from mujoco_env.y_env2 import SimpleEnv2

# ROOT: 数据集根目录（本地采集则 './demo_data_language'；若用 HF 下载，则指向其目录）
ROOT = "./datasets"
# ROOT = './omy_pnp_language'

dataset = LeRobotDataset('omy_grasp_mug', root=ROOT)

# ==== 单回合采样器 ====
class EpisodeSampler(torch.utils.data.Sampler):
    """
    单个回合的采样器
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

# 选择要可视化的回合索引
episode_index = 0

episode_sampler = EpisodeSampler(dataset, episode_index)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)

# ==== 在仿真中可视化 ====
xml_path = './asset/example_scene_y2.xml'
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

step = 0
iter_dataloader = iter(dataloader)
PnPEnv.reset()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 从数据集中读取一帧
        data = next(iter_dataloader)
        if step == 0:
            # 根据数据集初始化物体位姿/指令
            instruction = data['task'][0]
            PnPEnv.set_instruction(instruction)
            PnPEnv.set_obj_pose(data['obj_init'][0, :3], data['obj_init'][0, 3:6], data['obj_init'][0, 6:9])

        # 获取动作并执行
        action = data['action'].numpy()
        _obs = PnPEnv.step(action[0])

        # 将数据集图像转为叠加图
        PnPEnv.rgb_agent = (data['observation.image'][0].numpy() * 255).astype(np.uint8)
        PnPEnv.rgb_ego = (data['observation.wrist_image'][0].numpy() * 255).astype(np.uint8)
        # (C,H,W) -> (H,W,C)
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1, 2, 0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1, 2, 0))
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)

        PnPEnv.render()
        step += 1

        if step == len(episode_sampler):
            # 从头开始循环播放
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0

# 关闭查看器
PnPEnv.env.close_viewer()

# ==== 推送数据集到 Hub ====
# dataset.push_to_hub(upload_large_folder=True)
