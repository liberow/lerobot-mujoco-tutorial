# lerobot simulation

这是一个使用 mujoco 实现的仿真环境，数据集和模型策略均集成了 lerobot 的标准。

## 1. 数据采集

1. 在包含语言指令的场景中采集演示数据。右上/右下叠加图像来自数据集；侧视图显示在左上。

2. 数据集特征
```
fps = 20,
features={
    "observation.image": {"dtype": "image", "shape": (256, 256, 3)},
    "observation.wrist_image": {"dtype": "image", "shape": (256, 256, 3)},
    "observation.state": {"dtype": "float32", "shape": (6,)},
    "action": {"dtype": "float32", "shape": (7,)},
    "obj_init": {"dtype": "float32", "shape": (9,)},
}
```

3. 键盘控制说明

XY 平面: W 后退 / S 前进 / A 左移 / D 右移
Z 轴:    R 上升 / F 下降
旋转:    Q 左倾 / E 右倾 / 方向键控制俯仰与偏航
空格:    切换夹爪
Z:      重置（丢弃当前回合）

## 2. 可视化

## 3. 训练

### 3.1. install 
```
pip install transformers==4.50.3
pip install num2words
pip install accelerate
pip install 'safetensors>=0.4.3'
```

### 3.2. download dataset
1. git 
```
git clone https://huggingface.co/datasets/Liberow/omy_pickplace
``` 

2. cli
```
# install package
pip install huggingface_hub

# login
huggingface-cli login 

# download
huggingface-cli download Liberow/omy_pickplace \
    --repo-type dataset \
    --local-dir ./datasets
```

### 3.3. train 
```
python scripts/training.py --config_path configs/train_smolvla.yaml
```

## 4. 部署

### 4.1. download models
```
# download
huggingface-cli download Liberow/omy_pickplace \
    --repo-type model \
    --local-dir ./models
```
