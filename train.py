"""
Usage:
Training:使用说明，说明了如何训练模型
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# import os
# os.environ["WANDB_API_KEY"] = YOUR_API_KEY
# os.environ["WANDB_MODE"] = "offline"
# 设置为行缓冲模式。标准输出（stdout）和标准错误输出（stderr）每次输出新的一行时都会刷新缓冲区
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra    #  配置管理和参数化
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# 注册一个新的解析器eval，允许在配置文件中使用${eval:''}语法执行任意Python代码。这使得配置文件更加灵活和动态    allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# 这是一个Hydra的装饰器，标记main函数为程序的入口点，并指定配置文件的路径。
# version_base=None表示不使用特定版本的Hydra，config_path指定配置文件的路径。
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)


def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)                      # 解析配置文件中的所有引用和动态值，确保所有配置项在使用时已经被解析。
    # 这一步确保所有${eval:''}和其他动态配置项在使用前都已解析。
    
    cls = hydra.utils.get_class(cfg._target_)   # 从配置中获取目标类的名称，并动态加载这个类。
    workspace: BaseWorkspace = cls(cfg)         # 实例化目标类（这里假设是BaseWorkspace或其子类）的对象，并传入配置
    workspace.run()                             # 开始执行主程序逻辑

if __name__ == "__main__":
    main()
