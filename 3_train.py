#!/usr/bin/env python3
"""
训练 Action-Chunking-Transformer (ACT) 模型

在自定义数据集上训练 ACT 模型。本示例将 chunk_size 设为 10。
"""

# ==== 导入依赖 ====
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
import torchvision

# ==== 设备与训练配置 ====
device = torch.device("cuda")

# 仅进行离线训练步数（可按需调整，建议 ≥ 5000 步才有较好效果）
training_steps = 3000
log_freq = 100

# ==== 策略配置与初始化 ====
# 从零开始（非预训练）时，需要提供：
#  - 输入/输出的特征形状（用于正确构建网络）
#  - 数据集统计信息（用于归一化/反归一化）
chunk_size = 10

dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# 去掉腕部图像（与 notebook 一致）
input_features.pop("observation.wrist_image")

# 使用 ACTConfig，默认超参；仅传入输入/输出特征与 chunk 配置
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size=10, n_action_steps=10)
# 根据配置解析 delta_timestamps，构造带动作分块的数据
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

# 实例化策略并切换到训练模式
policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
policy.train()
policy.to(device)

# ==== 数据集与数据增强 ====
from torchvision import transforms

class AddGaussianNoise(object):
    """
    向张量添加高斯噪声
    """
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 添加噪声（保持为张量）
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# 图像增强流水线：添加噪声并截断到 [0,1]
transform = transforms.Compose([
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])

# 使用 delta_timestamps 创建数据集
dataset = LeRobotDataset("omy_pnp", delta_timestamps=delta_timestamps, root='./demo_data', image_transforms=transform)

# 优化器与 DataLoader（离线训练）
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)

# ==== 训练 ====
# 训练好的权重将保存到 './ckpt/act_y'
step = 0
done = False
while not done:
    for batch in dataloader:
        # 将张量移动到设备
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, _ = policy.forward(inp_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# 保存策略
policy.save_pretrained('./ckpt/act_y')

# ==== 推理测试 ====
# 在数据集上评估策略：计算预测动作与 GT 动作的误差
class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

policy.eval()
actions = []
gt_actions = []
images = []
episode_index = 0
episode_sampler = EpisodeSampler(dataset, episode_index)
test_dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=1,
    shuffle=False,
    pin_memory=device.type != "cpu",
    sampler=episode_sampler,
)
policy.reset()
for batch in test_dataloader:
    inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    action = policy.select_action(inp_batch)
    actions.append(action)
    gt_actions.append(inp_batch["action"][:, 0, :])
    images.append(inp_batch["observation.image"])

actions = torch.cat(actions, dim=0)
gt_actions = torch.cat(gt_actions, dim=0)
print(f"Mean action error: {torch.mean(torch.abs(actions - gt_actions)).item():.3f}")

# ==== 可选：绘制预测与 GT 的曲线 ====
# （需要 matplotlib）
# import matplotlib.pyplot as plt
# action_dim = 7
# fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))
# for i in range(action_dim):
#     axs[i].plot(actions[:, i].cpu().detach().numpy(), label="pred")
#     axs[i].plot(gt_actions[:, i].cpu().detach().numpy(), label="gt")
#     axs[i].legend()
# plt.show()
