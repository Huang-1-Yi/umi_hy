from typing import Optional, Callable, Any, Sequence
import os
import copy
import json
import numbers
import pandas as pd


def read_json_log(path: str, 
        required_keys: Sequence[str]=tuple(), 
        **kwargs) -> pd.DataFrame:
    """
    Read json-per-line file, with potentially incomplete lines.
    kwargs passed to pd.read_json
    读取 json-per-line 文件，可能不完整的 rows.kwargs 传递给 pd.read_json
    """
    lines = list()  # 创建一个空列表用于存储行
    with open(path, 'r') as f:
        while True:
            # 每行一个JSON
            line = f.readline()
            if len(line) == 0:
                # 文件结束
                break
            elif not line.endswith('\n'):
                # 不完整的行
                break
            is_relevant = False
            for k in required_keys:
                if k in line:
                    is_relevant = True
                    break
            if is_relevant:
                lines.append(line)  # 添加相关行
    if len(lines) < 1:
        return pd.DataFrame()  # 如果没有相关行，返回空的DataFrame
    json_buf = f'[{",".join([line for line in (line.strip() for line in lines) if line])}]'  # 创建JSON缓冲区
    df = pd.read_json(json_buf, **kwargs)  # 读取JSON并转换为DataFrame
    return df  # 返回DataFrame

class JsonLogger:
    def __init__(self, path: str, 
            filter_fn: Optional[Callable[[str, Any], bool]] = None):
        if filter_fn is None:
            filter_fn = lambda k, v: isinstance(v, numbers.Number)  # 默认过滤函数，仅保留数值类型

        # 默认追加模式
        self.path = path  # 文件路径
        self.filter_fn = filter_fn  # 过滤函数
        self.file = None  # 文件对象
        self.last_log = None  # 最后一次日志记录
    
    def start(self):
        # 使用行缓冲
        try:
            self.file = file = open(self.path, 'r+', buffering=1)  # 尝试以读写模式打开文件
        except FileNotFoundError:
            self.file = file = open(self.path, 'w+', buffering=1)  # 如果文件不存在，以写读模式创建文件

        # 将指针移动到文件末尾
        pos = file.seek(0, os.SEEK_END)

        # Read each character in the file one at a time from the last
        # character going backwards, searching for a newline character
        # If we find a new line, exit the search
        # 从最后一个字符开始，逐个字符向前读取，查找换行符
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        # now the file pointer is at one past the last '\n'
        # and pos is at the last '\n'.
        # 现在文件指针位于最后一个换行符之后
        last_line_end = file.tell()
        
        # find the start of second last line
        # 查找倒数第二行的开始位置
        pos = max(0, pos-1)
        file.seek(pos, os.SEEK_SET)
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        # now the file pointer is at one past the second last '\n'
        # 现在文件指针位于倒数第二个换行符之后
        last_line_start = file.tell()

        if last_line_start < last_line_end:
            # 如果存在最后一行JSON
            last_line = file.readline()
            self.last_log = json.loads(last_line)
        
        # 移除最后一行不完整的行
        file.seek(last_line_end)
        file.truncate()
    
    def stop(self):
        self.file.close()  # 关闭文件
        self.file = None
    
    def __enter__(self):
        self.start()  # 启动日志记录器
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()  # 停止日志记录器
    
    def log(self, data: dict):
        filtered_data = dict(
            filter(lambda x: self.filter_fn(*x), data.items()))  # 过滤数据
        # 保存当前日志记录为最后一次日志记录
        self.last_log = filtered_data
        for k, v in filtered_data.items():
            if isinstance(v, numbers.Integral):
                filtered_data[k] = int(v)  # 转换为整数
            elif isinstance(v, numbers.Number):
                filtered_data[k] = float(v)  # 转换为浮点数
        buf = json.dumps(filtered_data)  # 将过滤后的数据转换为JSON字符串
        # 确保每个JSON记录占一行
        buf = buf.replace('\n', '') + '\n'
        self.file.write(buf)  # 写入文件
    
    def get_last_log(self):
        return copy.deepcopy(self.last_log)  # 返回最后一次日志记录的深拷贝
