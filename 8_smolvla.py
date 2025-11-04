#!/usr/bin/env python3
"""
部署已训练的 SmolVLA 策略

包含（可选）下载数据集、训练与部署步骤。
"""

# ==== 可选：安装依赖（在终端中执行） ====
# pip install transformers==4.50.3
# pip install num2words
# pip install accelerate
# pip install 'safetensors>=0.4.3'

# ==== 可选：下载数据集（在终端中执行） ====
# git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language

# ==== 可选：训练（在终端中执行） ====
# python train_model.py --config_path smolvla_omy.yaml

# ==== 部署 ====
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision

# 设备
device = 'cuda'

# 加载数据集元信息（优先从本地采集路径，否则从 HF 下载目录）
try:
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./demo_data_language')
except Exception:
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./omy_pnp_language')

features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

# 配置与 delta_timestamps
cfg = SmolVLAConfig(input_features=input_features, output_features=output_features, chunk_size=5, n_action_steps=5)
_ = resolve_delta_timestamps(cfg, dataset_metadata)

# 从本地检查点加载策略（或从 Hub 加载，见注释）
policy = SmolVLAPolicy.from_pretrained('./ckpt/smolvla_omy/checkpoints/last/pretrained_model', dataset_stats=dataset_metadata.stats)
# policy = SmolVLAPolicy.from_pretrained("Jeongeun/omy_pnp_pi0", config=cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)

# 创建环境
from mujoco_env.y_env2 import SimpleEnv2
xml_path = './asset/example_scene_y2.xml'
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

# 简单的图像预处理
def get_default_transform(image_size: int = 224):
    """
    返回将 PIL 图像转换为 FloatTensor（[0,1]）的 transform
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
    ])

# 回放
step = 0
PnPEnv.reset(seed=0)
policy.reset()
policy.eval()
IMG_TRANSFORM = get_default_transform()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 成功判定
        if PnPEnv.check_success():
            print('Success')
            policy.reset()
            PnPEnv.reset()
            step = 0

        # 状态与图像
        state = PnPEnv.get_joint_state()[:6]
        image, wirst_image = PnPEnv.grab_image()
        image = IMG_TRANSFORM(Image.fromarray(image).resize((256, 256)))
        wrist_image = IMG_TRANSFORM(Image.fromarray(wirst_image).resize((256, 256)))

        # 推理输入
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [PnPEnv.instruction],
        }
        action = policy.select_action(data)
        action = action[0, :7].cpu().detach().numpy()

        # 执行
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1
        if PnPEnv.check_success():
            print('Success')
            break

# 可选：推送到 Hub
# policy.push_to_hub(
#     repo_id='Jeongeun/omy_pnp_smolvla',
#     commit_message='Add trained policy for PnP task',
# )
