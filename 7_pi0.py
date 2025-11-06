#!/usr/bin/env python3
"""
部署已训练的 pi_0 策略
"""
# ==== 下载数据集 ====
# git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language

# ==== 训练 ====
# python train_model.py --config_path pi0_omy.yaml

# ==== 部署 ====
import torch
from PIL import Image

from lerobot.configs.types import FeatureType
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features

# 加载策略配置
device = 'cuda'
try:
    dataset_metadata = LeRobotDatasetMetadata("omy_grasp_mug", root='./datasets')
except Exception:
    dataset_metadata = LeRobotDatasetMetadata("omy_grasp_mug", root='./datasets')

features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

cfg = PI0Config(input_features=input_features, output_features=output_features, chunk_size=5, n_action_steps=5)
_ = resolve_delta_timestamps(cfg, dataset_metadata)

# 从本地检查点加载策略（或从 Hub 加载）
# policy = PI0Policy.from_pretrained('./ckpt/pi0_omy/checkpoints/last/pretrained_model', dataset_stats=dataset_metadata.stats)
policy = PI0Policy.from_pretrained("Jeongeun/omy_pnp_pi0", config=cfg, dataset_stats=dataset_metadata.stats)
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
auth_step = 0
PnPEnv.reset(seed=0)
policy.reset()
policy.eval()
IMG_TRANSFORM = get_default_transform()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 是否完成任务
        if PnPEnv.check_success():
            print('Success')
            policy.reset()
            PnPEnv.reset()
            auth_step = 0

        # 读取当前状态与图像
        state = PnPEnv.get_joint_state()[:6]
        image, wirst_image = PnPEnv.grab_image()
        image = IMG_TRANSFORM(Image.fromarray(image).resize((256, 256)))
        wrist_image = IMG_TRANSFORM(Image.fromarray(wirst_image).resize((256, 256)))

        # 组装输入并推理
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [PnPEnv.instruction],
        }
        action = policy.select_action(data)
        action = action[0, :7].cpu().detach().numpy()

        # 执行并渲染
        _ = PnPEnv.step(action)
        PnPEnv.render()
        auth_step += 1
        if PnPEnv.check_success():
            print('Success')
            break

# 推送至 Hub（大文件请注意网络与空间）
# policy.push_to_hub(
#     repo_id='Jeongeun/omy_pnp_pi0',
#     commit_message='Add trained policy for PnP task',
# )
