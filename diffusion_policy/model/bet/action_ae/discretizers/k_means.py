import torch  # 导入torch库
import numpy as np  # 导入numpy库
import tqdm  # 导入tqdm库
from typing import Optional, Tuple, Union  # 导入类型提示
# 从diffusion_policy.model.common.dict_of_tensor_mixin导入DictOfTensorMixin类
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin


class KMeansDiscretizer(DictOfTensorMixin):
    """
    简化和修改版的KMeans算法，来源于sklearn
    Simplified and modified version of KMeans algorithm  from sklearn.
    """

    def __init__(
        self,
        action_dim: int,  # 动作维度
        num_bins: int = 100,  # 分箱数量，默认为100
        predict_offsets: bool = False,  # 是否预测偏移量，默认为False
    ):
        super().__init__()  # 初始化父类
        self.n_bins = num_bins  # 设置分箱数量
        self.action_dim = action_dim  # 设置动作维度
        self.predict_offsets = predict_offsets  # 设置是否预测偏移量

    def fit_discretizer(self, input_actions: torch.Tensor) -> None:
        assert (# 确保输入动作的维度匹配
            self.action_dim == input_actions.shape[-1]
        ), f"Input action dimension {self.action_dim} does not match fitted model {input_actions.shape[-1]}"

        flattened_actions = input_actions.view(-1, self.action_dim)  # 展平输入动作
        cluster_centers = KMeansDiscretizer._kmeans(
            flattened_actions, ncluster=self.n_bins  # 使用k-means算法找到聚类中心
        )
        self.params_dict['bin_centers'] = cluster_centers  # 存储聚类中心

    @property
    def suggested_actions(self) -> torch.Tensor:
        return self.params_dict['bin_centers']  # 返回建议的动作（聚类中心）

    @classmethod
    def _kmeans(cls, x: torch.Tensor, ncluster: int = 512, niter: int = 50):
        """
        简单的k-means聚类算法，改编自Karpathy的minGPT库
        https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
        """
        N, D = x.size()  # 获取输入的大小
        c = x[torch.randperm(N)[:ncluster]]  # 随机初始化聚类中心

        pbar = tqdm.trange(niter)  # 创建进度条
        pbar.set_description("K-means clustering")  # 设置进度条描述
        for i in pbar:
            # 将所有点分配给最近的聚类中心
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # 将每个聚类中心移动到分配给它的点的均值位置
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            # 重新分配位置不佳的聚类中心
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            if ndead:
                tqdm.tqdm.write(
                    "done step %d/%d, re-initialized %d dead clusters"
                    % (i + 1, niter, ndead)
                )
            c[nanix] = x[torch.randperm(N)[:ndead]]  # 重新初始化失效的聚类中心
        return c  # 返回聚类中心

    def encode_into_latent(
        self, input_action: torch.Tensor, input_rep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        给定输入动作，使用k-means聚类算法将其离散化。

        输入:
        input_action (形状: ... x action_dim): 要离散化的输入动作。可以是批处理，通常假设最后一个维度是动作维度。

        输出:
        discretized_action (形状: ... x num_tokens): 离散化的动作。
        如果self.predict_offsets为True，则返回偏移量。
        """
        assert (
            input_action.shape[-1] == self.action_dim
        ), "Input action dimension does not match fitted model"  # 确保输入动作的维度匹配

        flattened_actions = input_action.view(-1, self.action_dim)  # 展平输入动作

        closest_cluster_center = torch.argmin(
            torch.sum(
                (flattened_actions[:, None, :] - self.params_dict['bin_centers'][None, :, :]) ** 2,
                dim=2,
            ),
            dim=1,
        )  # 获取最近的聚类中心
        discretized_action = closest_cluster_center.view(input_action.shape[:-1] + (1,))  # 重新调整形状

        if self.predict_offsets:
            reconstructed_action = self.decode_actions(discretized_action)  # 从潜在空间解码并获取差值
            offsets = input_action - reconstructed_action  # 计算偏移量
            return (discretized_action, offsets)  # 返回离散化的动作和偏移量
        else:
            return discretized_action  # 返回离散化的动作

    def decode_actions(
        self,
        latent_action_batch: torch.Tensor,
        input_rep_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        给定潜在动作，重建原始动作。

        输入:
        latent_action (形状: ... x 1): 要重建的潜在动作。可以是批处理，通常假设最后一个维度是动作维度。
        如果latent_action_batch是一个元组，则假定为(discretized_action, offsets)。

        输出:
        reconstructed_action (形状: ... x action_dim): 重建的动作。
        """
        offsets = None
        if type(latent_action_batch) == tuple:
            latent_action_batch, offsets = latent_action_batch  # 解析潜在动作和偏移量
        closest_cluster_center = self.params_dict['bin_centers'][latent_action_batch]  # 获取最近的聚类中心
        reconstructed_action = closest_cluster_center.view(
            latent_action_batch.shape[:-1] + (self.action_dim,)
        )  # 重新调整形状
        if offsets is not None:
            reconstructed_action += offsets  # 添加偏移量
        return reconstructed_action  # 返回重建的动作

    @property
    def discretized_space(self) -> int:
        return self.n_bins  # 返回离散空间的大小

    @property
    def latent_dim(self) -> int:
        return 1  # 返回潜在维度

    @property
    def num_latents(self) -> int:
        return self.n_bins  # 返回潜在数量
