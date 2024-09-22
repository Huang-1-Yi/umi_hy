from typing import Dict, Callable, List # 导入需要的类型注解
import collections          # 导入collections模块
import torch                # 导入torch模块
import torch.nn as nn       # 导入torch.nn模块

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:   # 定义一个对字典中的每个值应用函数的函数
    result = dict()                     # 初始化结果字典
    for key, value in x.items():        # 遍历字典的键和值
        if isinstance(value, dict):     # 如果值是一个字典
            result[key] = dict_apply(value, func) # 递归调用自身
        else:
            result[key] = func(value)   # 否则对值应用函数
    return result # 返回结果字典

def pad_remaining_dims(x, target): # 定义一个函数，将x的形状填充到与target一致
    assert x.shape == target.shape[:len(x.shape)] # 确保x的形状与target的前几个维度一致
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape))) # 重新形状化x

def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:       # 定义一个函数，对字典中的每个值应用split_func并合并结果
    results = collections.defaultdict(dict) # 使用defaultdict初始化结果字典
    for key, value in x.items():            # 遍历字典的键和值
        result = split_func(value)          # 对值应用split_func
        for k, v in result.items():         # 遍历split_func的结果
            results[k][key] = v             # 将结果添加到results字典中
    return results # 返回结果字典

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:           # 定义一个函数，对字典列表中的每个值应用reduce_func并合并结果
    result = dict()                             # 初始化结果字典
    for key in x[0].keys():                     # 遍历字典的键
        result[key] = reduce_func([x_[key] for x_ in x]) # 对每个键对应的值应用reduce_func
    return result # 返回结果字典

def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module: # 定义一个函数，用新的子模块替换满足条件的子模块
    """
    predicate: 如果模块被替换,则返回true。
    func: 返回要使用的新模块。
    """
    if predicate(root_module):                  # 如果根模块满足条件
        return func(root_module)                # 返回新的模块

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]                        # 获取满足条件的模块的名称列表
    for *parent, k in bn_list:                  # 遍历名称列表
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent)) # 获取父模块
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]  # 获取原始子模块
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)           # 获取新的子模块
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module  # 替换子模块
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]                        # 再次获取满足条件的模块的名称列表
    assert len(bn_list) == 0                    # 确保所有满足条件的模块都被替换
    return root_module                          # 返回根模块

def optimizer_to(optimizer, device):            # 定义一个函数，将优化器的状态移动到指定设备
    for state in optimizer.state.values():      # 遍历优化器的状态
        for k, v in state.items():              # 遍历状态的键和值
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)  # 将张量移动到指定设备
    return optimizer                            # 返回优化器

