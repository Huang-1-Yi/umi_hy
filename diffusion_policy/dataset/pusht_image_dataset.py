from typing import Dict # 导入Dict类型提示
import torch # 导入torch模块
import numpy as np # 导入numpy模块
import copy # 导入copy模块
from diffusion_policy.common.pytorch_util import dict_apply # 导入dict_apply函数
from diffusion_policy.common.replay_buffer import ReplayBuffer # 导入ReplayBuffer类
from diffusion_policy.common.sampler import ( # 从sampler模块导入SequenceSampler, get_val_mask, downsample_mask函数
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer # 导入LinearNormalizer类
from diffusion_policy.dataset.base_dataset import BaseImageDataset # 导入BaseImageDataset类
from diffusion_policy.common.normalize_util import get_image_range_normalizer # 导入get_image_range_normalizer函数

class PushTImageDataset(BaseImageDataset): # 定义PushTImageDataset类，继承自BaseImageDataset
    def __init__(self,              # 初始化方法
            zarr_path,              # zarr数据路径
            horizon=1,              # 时间跨度
            pad_before=0,           # 前填充步数
            pad_after=0,            # 后填充步数
            seed=42,                # 随机种子
            val_ratio=0.0,          # 验证集比例
            max_train_episodes=None # 最大训练集数
            ):
        
        super().__init__()          # 调用父类初始化方法
        self.replay_buffer = ReplayBuffer.copy_from_path(   # 从路径复制ReplayBuffer
            zarr_path, keys=['img', 'state', 'action'])     # 指定键列表
        val_mask = get_val_mask(                            # 获取验证掩码
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask                              # 获取训练掩码
        train_mask = downsample_mask(                       # 下采样训练掩码
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(                     # 创建SequenceSampler实例
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask                        # 设置训练掩码
        self.horizon = horizon                              # 设置时间跨度
        self.pad_before = pad_before                        # 设置前填充步数
        self.pad_after = pad_after                          # 设置后填充步数

    def get_validation_dataset(self):               # 获取验证数据集的方法
        val_set = copy.copy(self)                   # 复制当前实例
        val_set.sampler = SequenceSampler(          # 创建新的SequenceSampler实例用于验证集
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask       # 设置验证集的训练掩码
        return val_set                              # 返回验证数据集

    def get_normalizer(self, mode='limits', **kwargs):      # 获取数据归一化器的方法
        data = {                                            # 从重放缓冲区采样数据
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()                     # 创建LinearNormalizer实例
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)  # 拟合数据
        normalizer['image'] = get_image_range_normalizer()  # 获取图像范围归一化器
        return normalizer                                   # 返回归一化器

    def __len__(self) -> int:                               # 获取数据集长度的方法
        return len(self.sampler)                            # 返回采样器的长度

    def _sample_to_data(self, sample):                      # 将采样数据转换为所需格式的方法
        agent_pos = sample['state'][:,:2].astype(np.float32)# 获取代理位置数据，并转换为float32类型
        image = np.moveaxis(sample['img'],-1,1)/255         # 调整图像轴顺序，并归一化到[0,1]范围

        data = {                            # 构建数据字典
            'obs': {
                'image': image,             # T, 3, 96, 96
                'agent_pos': agent_pos,     # T, 2
            },
            'action': sample['action'].astype(np.float32)   # T, 2
        }
        return data                         # 返回数据
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: # 获取单个数据项的方法
        sample = self.sampler.sample_sequence(idx)              # 根据索引采样序列
        data = self._sample_to_data(sample)                     # 转换采样数据
        torch_data = dict_apply(data, torch.from_numpy)         # 将数据转换为Tensor
        return torch_data                                       # 返回Tensor数据

def test():     # 测试函数
    import os   # 导入os模块
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')  # 展开用户目录并指定zarr数据路径
    dataset = PushTImageDataset(zarr_path, horizon=16)          # 创建PushTImageDataset实例

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
