from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

# 定义PushTLowdimDataset类，继承自BaseLowdimDataset
class PushTLowdimDataset(BaseLowdimDataset):
    # 从指定路径复制数据并生成验证掩码和训练掩码，并初始化序列采样器
    def __init__(self,  # 定义初始化方法
            zarr_path,  # zarr数据路径
            horizon=1,  # 时间跨度
            pad_before=0,  # 前填充步数
            pad_after=0,  # 后填充步数
            obs_key='keypoint',  # 观测键
            state_key='state',  # 状态键
            action_key='action',  # 动作键
            seed=42,  # 随机种子
            val_ratio=0.0,  # 验证集比例
            max_train_episodes=None  # 最大训练集数
            ):
        super().__init__()  # 调用父类初始化方法
        self.replay_buffer = ReplayBuffer.copy_from_path(  # 从路径复制ReplayBuffer
            zarr_path, keys=[obs_key, state_key, action_key])  # 指定键列表

        val_mask = get_val_mask(  # 获取验证掩码
            n_episodes=self.replay_buffer.n_episodes,  # 重放缓冲区中的剧集数
            val_ratio=val_ratio,  # 验证集比例
            seed=seed)  # 随机种子
        train_mask = ~val_mask  # 获取训练掩码
        train_mask = downsample_mask(  # 下采样训练掩码
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(  # 创建SequenceSampler实例
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.obs_key = obs_key  # 设置观测键
        self.state_key = state_key  # 设置状态键
        self.action_key = action_key  # 设置动作键
        self.train_mask = train_mask  # 设置训练掩码
        self.horizon = horizon  # 设置时间跨度
        self.pad_before = pad_before  # 设置前填充步数
        self.pad_after = pad_after  # 设置后填充步数

    # 复制当前实例，并创建一个新的序列采样器，使用验证掩码来生成验证数据集
    def get_validation_dataset(self):  # 获取验证数据集的方法
        val_set = copy.copy(self)  # 复制当前实例
        val_set.sampler = SequenceSampler(  # 创建新的SequenceSampler实例用于验证集
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask  # 设置验证集的训练掩码
        return val_set  # 返回验证数据集

    # 从重放缓冲区采样数据，并使用 LinearNormalizer 对数据进行归一化
    def get_normalizer(self, mode='limits', **kwargs):  # 获取归一化器的方法
        data = self._sample_to_data(self.replay_buffer)  # 从重放缓冲区采样数据
        normalizer = LinearNormalizer()  # 创建LinearNormalizer实例
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)  # 拟合数据
        return normalizer  # 返回归一化器

    # 从重放缓冲区获取动作并转换为 torch.Tensor
    def get_all_actions(self) -> torch.Tensor:  # 获取所有动作的方法
        return torch.from_numpy(self.replay_buffer[self.action_key])  # 从重放缓冲区获取动作并转换为Tensor


    # 返回采样器的长度
    def __len__(self) -> int:  # 获取数据集长度的方法
        return len(self.sampler)  # 返回采样器的长度

    # 将采样数据转换为所需格式
    def _sample_to_data(self, sample):  # 将采样数据转换为所需格式的方法
        keypoint = sample[self.obs_key]  # 获取观测数据中的关键点
        state = sample[self.state_key]  # 获取状态数据
        agent_pos = state[:,:2]  # 获取代理位置
        obs = np.concatenate([  # 拼接观测数据和代理位置
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            'obs': obs, # T, D_o 观测数据
            'action': sample[self.action_key], # T, D_a 动作数据
        }
        return data  # 返回数据

    # 根据索引采样序列，并将数据转换为 torch.Tensor
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # 获取单个数据项的方法
        sample = self.sampler.sample_sequence(idx)  # 根据索引采样序列
        data = self._sample_to_data(sample)  # 转换采样数据

        torch_data = dict_apply(data, torch.from_numpy)  # 将数据转换为Tensor
        return torch_data  # 返回Tensor数据
