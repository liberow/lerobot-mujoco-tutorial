#!/usr/bin/env python3
"""
部署已训练的 ACT 策略

在模拟环境中部署并回放已训练的策略。
"""

# ==== 导入依赖 ====
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision

# ==== 加载策略配置 ====
device = 'cuda'

dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# 与 notebook 一致：去掉腕部图像
dinput = input_features.pop("observation.wrist_image", None)

# 使用 ACT 配置；启用时间集成以平滑轨迹
cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    chunk_size=10,
    n_action_steps=1,
    temporal_ensemble_coeff=0.9,
)
# 解析时间戳（若用于带分块的数据加载）
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

# 从检查点加载策略
policy = ACTPolicy.from_pretrained('./ckpt/act_y', config=cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)

# ==== 创建仿真环境 ====
from mujoco_env.y_env import SimpleEnv
xml_path = './asset/example_scene_y.xml'
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

# ==== 策略回滚 ====
step = 0
PnPEnv.reset(seed=0)
policy.reset()
policy.eval()
save_image = True
img_transform = torchvision.transforms.ToTensor()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 检查任务是否成功
        success = PnPEnv.check_success()
        if success:
            print('Success')
            # 重置策略与环境
            policy.reset()
            PnPEnv.reset(seed=0)
            step = 0
            save_image = False

        # 读取当前状态
        state = PnPEnv.get_ee_pose()

        # 读取相机图像，并处理为 256x256 张量
        image, wirst_image = PnPEnv.grab_image()
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = img_transform(image)
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = img_transform(wrist_image)

        # 组装策略输入
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': ['Put mug cup on the plate'],
            'timestamp': torch.tensor([step/20]).to(device),
        }

        # 推理并执行一步
        action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1

        # 再次检查是否成功
        if PnPEnv.check_success():
            print('Success')
            break
