#!/usr/bin/env python3
"""
可视化采集的数据

在重建的模拟场景中可视化基于动作的回放。
主模拟窗口会重放动作序列。
右上和右下的叠加图像来自数据集。
"""

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_json, serialize_dict
from mujoco_env.y_env import SimpleEnv

# ============================================================================
# 配置参数
# ============================================================================

# 数据集配置
REPO_NAME = 'liberow'  # 与采集时使用的名称一致
ROOT = './demo_data'  # 数据集路径
# 如果要使用提供的示例数据，使用: ROOT = './demo_data_example'

# 可视化配置
EPISODE_INDEX = 0  # 要可视化的回合索引

# 环境配置
xml_path = './asset/example_scene_y.xml'

# ============================================================================
# Episode采样器
# ============================================================================

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

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 64)
    print("数据可视化脚本")
    print("=" * 64)
    print(f"数据集: {REPO_NAME}")
    print(f"数据路径: {ROOT}")
    print(f"可视化回合: {EPISODE_INDEX}")
    print("=" * 64)
    
    # ========================================================================
    # 步骤 1: 加载数据集
    # ========================================================================
    print("\n[1/3] 正在加载数据集...")
    try:
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)
        print(f"✓ 数据集加载成功")
        print(f"  - 总回合数: {len(dataset.episode_data_index['from'])}")
        print(f"  - 总帧数: {len(dataset)}")
    except Exception as e:
        print(f"✗ 加载数据集失败: {e}")
        print(f"  请确保路径 {ROOT} 包含有效的数据集")
        return
    
    # ========================================================================
    # 步骤 2: 创建数据加载器
    # ========================================================================
    print(f"\n[2/3] 正在创建回合 {EPISODE_INDEX} 的数据加载器...")
    
    # 检查回合索引是否有效
    num_episodes = len(dataset.episode_data_index['from'])
    if EPISODE_INDEX >= num_episodes:
        print(f"✗ 回合索引 {EPISODE_INDEX} 超出范围 (总共 {num_episodes} 个回合)")
        print(f"  请选择 0 到 {num_episodes - 1} 之间的索引")
        return
    
    episode_sampler = EpisodeSampler(dataset, EPISODE_INDEX)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        sampler=episode_sampler,
    )
    
    print(f"✓ 数据加载器创建成功")
    print(f"  - 回合帧数: {len(episode_sampler)}")
    
    # ========================================================================
    # 步骤 3: 在模拟中可视化数据集
    # ========================================================================
    print("\n[3/3] 正在创建模拟环境...")
    PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')
    print("✓ 环境创建成功")
    
    print("\n" + "=" * 64)
    print("开始可视化回放")
    print("=" * 64)
    print("说明:")
    print("  - 主窗口: 模拟环境中的动作回放")
    print("  - 右上叠加图像: 数据集中的代理视角")
    print("  - 右下叠加图像: 数据集中的自我中心视角")
    print("  - 回放会循环播放")
    print("  - 按 ESC 退出")
    print("=" * 64)
    
    input("\n按回车键开始可视化...")
    
    step = 0
    iter_dataloader = iter(dataloader)
    PnPEnv.reset()
    
    try:
        while PnPEnv.env.is_viewer_alive():
            PnPEnv.step_env()
            if PnPEnv.env.loop_every(HZ=20):
                # 从数据集获取数据
                data = next(iter_dataloader)
                
                if step == 0:
                    # 根据数据集重置物体位姿
                    PnPEnv.set_obj_pose(data['obj_init'][0, :3], data['obj_init'][0, 3:])
                
                # 从数据集获取动作
                action = data['action'].numpy()
                obs = PnPEnv.step(action[0])
                
                # 将数据集中的图像可视化为叠加层
                PnPEnv.rgb_agent = data['observation.image'][0].numpy() * 255
                PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy() * 255
                PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
                PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
                
                # 转换维度: 3 x 256 x 256 -> 256 x 256 x 3
                PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1, 2, 0))
                PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1, 2, 0))
                PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
                
                PnPEnv.render()
                step += 1
                
                if step == len(episode_sampler):
                    # 从头开始循环播放
                    print(f"\n→ 回合 {EPISODE_INDEX} 播放完成，重新开始...")
                    iter_dataloader = iter(dataloader)
                    PnPEnv.reset()
                    step = 0
    
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断 (Ctrl+C)")
    
    except StopIteration:
        print("\n✓ 数据集播放完成")
    
    finally:
        # ====================================================================
        # 清理工作
        # ====================================================================
        print("\n" + "=" * 64)
        print("正在清理...")
        PnPEnv.env.close_viewer()
        print("✓ 查看器已关闭")
        print("\n" + "=" * 64)
        print("可视化完成！")
        print("=" * 64)


# ============================================================================
# 可选功能: 保存统计信息
# ============================================================================

def save_stats(dataset_root):
    """
    保存数据集统计信息到 stats.json
    这是一个可选功能，用于其他版本的兼容性
    
    参数:
        dataset_root: 数据集根目录路径
    """
    print("\n正在保存统计信息...")
    try:
        dataset = LeRobotDataset(REPO_NAME, root=dataset_root)
        stats = dataset.meta.stats
        PATH = dataset.root / 'meta' / 'stats.json'
        stats = serialize_dict(stats)
        write_json(stats, PATH)
        print(f"✓ 统计信息已保存到: {PATH}")
    except Exception as e:
        print(f"✗ 保存统计信息失败: {e}")


if __name__ == "__main__":
    main()
    
    # 如果需要保存统计信息，取消下面这行的注释
    # save_stats(ROOT)

