from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading

# os.environ["WANDB_API_KEY"] = "b5c048fbf1ca69b0e7b89958a2cd76488741e601"
# os.environ["WANDB_MODE"] = "offline"

class BaseWorkspace:
    include_keys = tuple()              # 包含的键的元组
    exclude_keys = tuple()              # 排除的键的元组

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None): # 初始化方法，接收配置和输出目录参数
        self.cfg = cfg                  # 保存配置
        self._output_dir = output_dir   # 保存输出目录
        self._saving_thread = None      # 初始化保存线程为None

    @property
    def output_dir(self):               # 定义输出目录的属性方法
        output_dir = self._output_dir   # 获取输出目录
        if output_dir is None:          # 如果输出目录为None
            output_dir = HydraConfig.get().runtime.output_dir # 从Hydra配置获取运行时输出目录
        return output_dir               # 返回输出目录
    
    def run(self): # 定义运行方法
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):           # 定义保存检查点的方法
        if path is None:                # 如果路径为None
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt') # 设置默认路径
        else:
            path = pathlib.Path(path)   # 否则使用提供的路径
        if exclude_keys is None:        # 如果排除的键为None
            exclude_keys = tuple(self.exclude_keys) # 使用类属性中的排除键
        if include_keys is None:        # 如果包含的键为None
            include_keys = tuple(self.include_keys) + ('_output_dir',) # 使用类属性中的包含键并添加'_output_dir'

        path.parent.mkdir(parents=False, exist_ok=True) # 创建路径的父目录
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }  # 初始化保存的数据

        for key, value in self.__dict__.items():    # 遍历对象的属性
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'): # 如果属性有state_dict和load_state_dict方法
                if key not in exclude_keys:         # 如果键不在排除键列表中
                    if use_thread:                  # 如果使用线程
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict()) # 将状态字典复制到CPU
                    else:
                        payload['state_dicts'][key] = value.state_dict() # 直接使用状态字典
            elif key in include_keys:               # 如果键在包含键列表中
                payload['pickles'][key] = dill.dumps(value) # 使用dill序列化属性
        if use_thread:                              # 如果使用线程
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill)) # 创建保存线程
            self._saving_thread.start()             # 启动保存线程
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill) # 直接保存数据
        return str(path.absolute())                 # 返回绝对路径
    
    def get_checkpoint_path(self, tag='latest'):        # 获取检查点路径的方法
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt') # 返回检查点路径

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs): # 加载数据的方法
        if exclude_keys is None:                            # 如果排除键为None
            exclude_keys = tuple()                          # 初始化为空元组
        if include_keys is None:                            # 如果包含键为None
            include_keys = payload['pickles'].keys()        # 使用数据中的键

        for key, value in payload['state_dicts'].items():   # 遍历状态字典
            if key not in exclude_keys:                     # 如果键不在排除键列表中
                self.__dict__[key].load_state_dict(value, **kwargs) # 加载状态字典
        for key in include_keys:                            # 遍历包含键
            if key in payload['pickles']:                   # 如果键在数据中
                self.__dict__[key] = dill.loads(payload['pickles'][key]) # 反序列化并加载属性
    
    #  从检查点文件中加载模型的状态和其他训练信息 定义了一个名为 load_checkpoint 的方法，它接受一个可选的 path 参数、一个默认值为 'latest' 的 tag 参数、以及可选的 exclude_keys 和 include_keys 参数。**kwargs 允许传递任意其他关键字参数
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs): # 加载检查点的方法
        if path is None: # 如果路径为None
            path = self.get_checkpoint_path(tag=tag) # 获取默认路径：动态生成检查点路径
        else:
            path = pathlib.Path(path)   # 否则使用提供的路径，将字符串路径转换为 pathlib.Path 对象
        # 确保路径指向一个存在的文件：如果检查点路径指向一个不存在的文件，方法将抛出一个错误
        if path.is_file():              # 检查点路径指向一个存在的文件，继续加载检查点
            payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)# 使用 torch.load 函数加载检查点文件。pickle_module=dill 指定使用 dill 模块进行序列化和反序列化。**kwargs 允许传递任意其他关键字参数。
        else:                           # 检查点路径指向一个不存在的文件，抛出错误
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)  # 将检查点中的数据加载到模型和优化器中
        return payload                  # 返回数据
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):                  # 从检查点创建实例的方法
        payload = torch.load(open(path, 'rb'), pickle_module=dill) # 加载检查点数据
        instance = cls(payload['cfg'])  # 创建实例
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)                   # 加载数据
        return instance                 # 返回实例

    def save_snapshot(self, tag='latest'):                  # 保存快照的方法
        """
        Quick loading and saving for research, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl') # 设置快照路径
        path.parent.mkdir(parents=False, exist_ok=True)         # 创建路径的父目录
        torch.save(self, path.open('wb'), pickle_module=dill)   # 保存快照
        return str(path.absolute())                             # 返回快照的绝对路径
    
    @classmethod
    def create_from_snapshot(cls, path):                        # 从快照创建实例的方法
        return torch.load(open(path, 'rb'), pickle_module=dill) # 加载快照并返回实例

def _copy_to_cpu(x):                    # 定义将数据复制到CPU的方法
    if isinstance(x, torch.Tensor):     # 如果是张量,复制到CPU
        return x.detach().to('cpu')
    elif isinstance(x, dict):           # 如果是字典,初始化结果字典,遍历字典的键和值,递归复制到CPU
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result                   # 返回结果字典
    elif isinstance(x, list):           # 如果是列表,递归复制到CPU
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)         # 深拷贝并返回
