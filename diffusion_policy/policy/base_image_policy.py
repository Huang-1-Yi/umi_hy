from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin # 混合类，它可能包含了一些与 PyTorch 模块相关的额外属性和方法，用于扩展 BaseImagePolicy 类的功能
from diffusion_policy.model.common.normalizer import LinearNormalizer       # 用于数据标准化的类，它可能定义了如何将输入数据转换为具有特定均值和标准差的分布，这通常是为了提高模型训练的稳定性和性能

# 为图像输入的决策策略模型提供了一个框架，它定义了一些通用的接口和属性，具体的实现细节需要由子类来完成
# 这样的设计使得不同的图像策略模型可以共享一些通用的代码和接口，同时又能根据具体的需求进行定制化的扩展。
class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml
    # 抽象方法，接受观察字典 obs_dict，
    # 其中的键是字符串，值是形状为 B, To, * 的 PyTorch 张量（其中 B 是批量大小，To 是观察时间步长，* 是其他维度），然后返回一个字典，其中的键是字符串，值是形状为 B, Ta, Da 的 PyTorch 张量（其中 Ta 是动作时间步长，Da 是动作维度）
    # 这个方法需要子类来实现，以定义如何根据观察预测动作
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    # 可选的方法，用于重置策略模型的状态。
    # 这对于有状态的策略（例如那些包含循环神经网络或某些形式的内部状态的策略）是必要的
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    # 抽象方法，它接受一个 LinearNormalizer 实例，用于设置策略模型的数据标准化器。这个方法需要子类来实现，以定义如何设置和更新模型的标准化器
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
