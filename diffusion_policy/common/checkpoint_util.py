from typing import Optional, Dict  # 导入类型提示模块
import os  # 导入操作系统模块

# 提供了一个简单的方法来管理模型训练过程中的检查点，确保只保留性能最好的前k个检查点，以节省存储空间并便于追踪最佳模型
class TopKCheckpointManager:
    def __init__(self,
            save_dir,  # 保存目录
            monitor_key: str,  # 监控键
            mode='min',  # 模式，默认为'min'
            k=1,  # 保留的检查点数量
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'  # 文件名格式
        ):
        """
        初始化TopKCheckpointManager的实例
        - save_dir: 检查点保存的目录
        - monitor_key: 监控的性能指标键
        - mode: 要监控的性能指标是最大化还是最小化，可选值为'max'或'min'
        - k: 要保留的最佳检查点数量
        - format_str: 检查点文件名的格式
        """
        assert mode in ['max', 'min']   # 确保mode参数是'max'或'min'
        assert k >= 0                   # 确保k是非负数
        self.save_dir = save_dir        # 保存检查点的目录
        self.monitor_key = monitor_key  # 监控的性能指标键
        self.mode = mode                # 性能指标是最大化还是最小化
        self.k = k                      # 要保留的最佳检查点数量
        self.format_str = format_str    # 检查点文件名的格式
        self.path_value_map = dict()    # 存储检查点路径和对应性能指标的字典
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        """
        根据监控的性能指标，获取或创建新的检查点路径
        - data: 包含性能指标的字典
        返回: 新的检查点路径或None
        """
        if self.k == 0:
            return None                 # 如果k为0，则不保存检查点
        value = data[self.monitor_key]              # 获取监控的性能指标值
        ckpt_path = os.path.join(                   # 构建检查点文件的完整路径
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # 如果当前保存的检查点数量少于k个
            self.path_value_map[ckpt_path] = value  # 添加新的检查点和性能值,添加到映射字典
            return ckpt_path                        # 返回新的检查点路径
        # 如果当前保存的检查点数量已达到k个
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])  # 根据性能值对检查点进行排序
        min_path, min_value = sorted_map[0]         # 最小性能值的检查点,最小值路径和最小值
        max_path, max_value = sorted_map[-1]        # 最大性能值的检查点,最大值路径和最大值
        delete_path = None                          # 初始化要删除的检查点路径
        if self.mode == 'max':
            # 如果是最大化模式
            if value > min_value:
                delete_path = min_path  # 如果新值大于最小值，则删除最小值的检查点
        else:
            # 如果是最小化模式
            if value < max_value:
                delete_path = max_path  # 如果新值小于最大值，则删除最大值的检查点
        if delete_path is None:
            return None                 # 如果没有检查点需要删除，则不保存新的检查点
        else:
            del self.path_value_map[delete_path]    # 从字典中删除要删除的检查点
            self.path_value_map[ckpt_path] = value  # 添加新的检查点和性能值
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)             # 如果保存目录不存在，则创建它
            if os.path.exists(delete_path):
                os.remove(delete_path)              # 删除旧的检查点文件
            return ckpt_path                        # 返回新的检查点路径