"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # 为标准输出使用行缓冲
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)  # 为标准错误使用行缓冲

import hydra  # 导入Hydra库，用于配置管理
from omegaconf import OmegaConf  # 导入OmegaConf，用于处理配置文件
import pathlib  # 导入Pathlib，用于路径操作
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # 导入BaseWorkspace类

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)  # 注册一个新的解析器，允许在配置文件中使用${eval:''}执行任意Python代码

@hydra.main(
    version_base=None,  # 设置Hydra版本基准为None
    config_path=str(pathlib.Path(__file__).parent.joinpath(  # 设置配置文件路径
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)  # 立即解析配置，以便所有的${now:}解析器使用相同的时间

    cls = hydra.utils.get_class(cfg._target_)  # 从配置中获取目标类
    workspace: BaseWorkspace = cls(cfg)  # 实例化工作区
    workspace.run()  # 运行工作区

if __name__ == "__main__":
    main()  # 如果脚本作为主程序运行，则调用main函数

