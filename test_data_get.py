# 新增的 inference.py 代码
import torch
from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet
import os # 导入os模块
import hydra # 导入hydra模块
import torch # 导入torch模块
from omegaconf import OmegaConf # 导入OmegaConf类
import pathlib # 导入pathlib模块
from torch.utils.data import DataLoader # 导入DataLoader类
import copy # 导入copy模块
import random # 导入random模块
import wandb # 导入wandb模块
import tqdm # 导入tqdm模块
import numpy as np # 导入numpy模块
import shutil # 导入shutil模块
from diffusion_policy.workspace.base_workspace import BaseWorkspace # 导入BaseWorkspace类
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy # 导入DiffusionUnetImagePolicy类
from diffusion_policy.dataset.base_dataset import BaseImageDataset # 导入BaseImageDataset类
from diffusion_policy.dataset.real_pusht_image_dataset import RealPushTImageDataset # 导入BaseImageDataset类
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner # 导入BaseImageRunner类
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager # 导入TopKCheckpointManager类
from diffusion_policy.common.json_logger import JsonLogger # 导入JsonLogger类
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to # 导入dict_apply和optimizer_to函数
from diffusion_policy.model.diffusion.ema_model import EMAModel # 导入EMAModel类
from diffusion_policy.model.common.lr_scheduler import get_scheduler # 导入get_scheduler函数

# Traceback (most recent call last):                                                                            
#   File "/home/hy/dp/test_predict.py", line 191, in <module>
#     result = policy.predict_action(obs_dict)
#   File "/home/hy/dp/diffusion_policy/policy/diffusion_unet_image_policy.py", line 131, in predict_action
#     nobs = self.normalizer.normalize(obs_dict)
#   File "/home/hy/dp/diffusion_policy/model/common/normalizer.py", line 68, in normalize
#     return self._normalize_impl(x, forward=True)
#   File "/home/hy/dp/diffusion_policy/model/common/normalizer.py", line 59, in _normalize_impl
#     result[key] = _normalize(value, params, forward=forward)
#   File "/home/hy/dp/diffusion_policy/model/common/normalizer.py", line 272, in _normalize
#     x = x.reshape(-1, scale.shape[0])
# RuntimeError: shape '[-1, 16997]' is invalid for input of size 128

# 修改 shape_meta 定义为标准字典
shape_meta = {
    'obs': {
        'camera_0': {
            'shape': [3, 240, 320],
            'type': 'rgb'
        },
        'camera_1': {
            'shape': [3, 240, 320],
            'type': 'rgb'
        },
        'robot_eef_pose': {
            'shape': [6],
            'type': 'low_dim'
        },
        'stage': {
            'shape': [1],
            'type': 'low_dim'
        },
        'gripper': {
            'shape': [1],
            'type': 'low_dim'
        }
    },
    'action': {
        'shape': [7]
    }
}

# 初始化组件 -------------------------------------------------
# 1. 创建噪声调度器
noise_scheduler = DDIMScheduler(
    num_train_timesteps=100,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    set_alpha_to_one=True,
    steps_offset=0,
    prediction_type='epsilon',
)

# 2. 创建观测编码器 (需要根据实际输入图像尺寸调整参数)
obs_encoder = MultiImageObsEncoder(
    shape_meta=shape_meta,  # 直接使用字典
    rgb_model=get_resnet(name='resnet18'),
    resize_shape=(240, 320),
    crop_shape=(216, 288),
    random_crop=True,
    use_group_norm=True,
    share_rgb_model=False,
    imagenet_norm=True,
)


# 3. 创建策略模型
policy = DiffusionUnetImagePolicy(
    shape_meta=shape_meta,  # 直接使用字典
    noise_scheduler=noise_scheduler,
    obs_encoder=obs_encoder,
    horizon=16,
    n_action_steps=2,
    n_obs_steps=2,
    num_inference_steps=100,
    down_dims=(512, 1024, 2048),
    obs_as_global_cond=True,
    diffusion_step_embed_dim=128,
    kernel_size= 5,
    n_groups= 8,
    cond_predict_scale=True,
)

shape_meta_test = OmegaConf.create({
    'obs': {
        'camera_0': {
            'shape': [3, 240, 320],
            'type': 'rgb'
        },
        'camera_1': {
            'shape': [3, 240, 320],
            'type': 'rgb'
        },
        'robot_eef_pose': {
            'shape': [6],
            'type': 'low_dim'
        },
        'stage':{
            'shape': [1],
            'type': 'low_dim'
            
        },
        'gripper': {
            'shape': [1],
            'type': 'low_dim'
        }},
    'action': {
        'shape': [7]
    }
})

# 加载数据 -------------------------------------------------
# 4. 创建数据集（复用dataset.py的参数）
params = {
    "shape_meta": shape_meta_test,
    "dataset_path": "data/demo_test_02253",
    "horizon": 16,
    "pad_before": 1,
    "pad_after": 7,
    "n_obs_steps": 2,
    "use_cache": True,
    "val_ratio": 0.1,
    "max_train_episodes": 1000,
    "delta_action": False,
}

# 5. 创建数据集和加载器
train_dataset = RealPushTImageDataset(**params)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=64,  # 小批量测试
    num_workers=8,
    shuffle=True
)

# 关键配置步骤 ---------------------------------------------
# 6. 设置归一化器（必须在预测前执行）
policy.set_normalizer(train_dataset.get_normalizer())

# 7. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = policy.to(device)
policy.eval() # 设置模型为评估模式

# 执行预测 -------------------------------------------------
# with torch.no_grad():
#     # 8. 获取第一个batch的数据
#     batch = next(iter(train_loader))
#     # obs_dict = {
#     #     'obs': batch['obs']  # 保持原始数据结构
#     # }
#     obs_dict = batch['obs'] # 构建观测字典
    
#     # 9. 转换数据到设备
#     obs_dict = dict_apply(obs_dict, lambda x: x.to(device))
    
#     # 10. 执行预测
#     result = policy.predict_action(obs_dict)
#     pred_action = result['action_pred'] # 获取预测动作

with torch.no_grad(), \
     tqdm.tqdm(train_dataloader, desc="Running prediction", 
              leave=False, mininterval=1.0) as pbar:  # 添加进度条

    for batch_idx, batch in enumerate(pbar):  # 遍历所有批次
        # 分解数据
        obs_dict = batch['obs']  # 获取观测字典
        gt_action = batch['action']  # 获取真实动作（可选）

        # 设备转移（带非阻塞传输优化）
        # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        
        # 执行预测
        result = policy.predict_action(obs_dict)
        pred_action = result['action_pred']  # 提取预测动作
        
        # 可选：在进度条显示统计信息
        pbar.set_postfix({
            'pred_mean': pred_action.mean().item(),
            'pred_std': pred_action.std().item()
        })

# with torch.no_grad(), \
#      tqdm.tqdm(train_dataloader, desc="Running prediction", 
#               leave=False, mininterval=1.0) as pbar:

#     for batch_idx, batch in enumerate(pbar):
#         # 确保观测数据是字典结构
#         if not isinstance(batch['obs'], dict):  # 添加类型检查
#             raise TypeError(f"Expected obs_dict to be dict, got {type(batch['obs'])}")
        
#         # 正确传递观测字典
#         obs_dict = {
#             'camera_0': batch['obs']['camera_0'],  # 明确指定每个传感器
#             'camera_1': batch['obs']['camera_1'],
#             'robot_eef_pose': batch['obs']['robot_eef_pose'],
#             'stage': batch['obs']['stage'],
#             'gripper': batch['obs']['gripper']
#         }
        
#         # 设备转移（保持字典结构）
#         obs_dict = dict_apply(obs_dict, lambda x: x.to(device, non_blocking=True))
        
#         # 执行预测
#         result = policy.predict_action(obs_dict)  # 现在传入的是正确字典
#         pred_action = result['action_pred']

# 解析结果 -------------------------------------------------
        # 11. 输出结果结构
        print("\n=== Prediction Result Structure ===")
        print(f"Result keys: {list(result.keys())}")
        print(f"action shape: {result['action'].shape}")
        print(f"action_pred shape: {result['action_pred'].shape}")

        # 12. 显示第一个预测样本
        print("\n=== First Sample Prediction ===")
        print("Action (mean):", result['action'][0].mean(dim=0).cpu().numpy())
        print("Action sequence:", result['action_pred'][0].cpu().numpy().round(3))