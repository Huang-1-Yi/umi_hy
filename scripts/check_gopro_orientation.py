# %%
# Python编写的命令行工具，它使用Click库来处理命令行参数。
# 这个脚本的主要目的是遍历一系列的会话目录，检查每个目录中的MP4视频文件，并使用ExifTool来获取视频的旋转信息。如果视频文件不是正立的（即旋转角度不是0度），打印出相应的信息
import sys
import os
# 设置项目的根目录，并更新Python的路径，以便能够正确地导入项目中的模块
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
# 导入click库用于命令行参数处理，json模块用于处理JSON数据，pathlib模块用于处理文件系统路径，numpy模块用于数学运算，tqdm模块用于显示进度条，以及exiftool模块用于读取和写入图像、视频和音频文件的元数据
import click
import json
import pathlib
import numpy as np
from tqdm import tqdm
from exiftool import ExifToolHelper

# %%
# 定义main接受名为session_dir的参数，参数传递一个或多个会话目录的路径
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    # 创建一个ExifToolHelper对象，并将其上下文管理器用于自动处理资源的打开和关闭
    with ExifToolHelper() as et:
        # 遍历命令行参数中提供的会话目录
        for session in tqdm(session_dir):
            # 将相对路径转换为绝对路径，pathlib对象
            session = pathlib.Path(os.path.expanduser(session)).absolute()
            # hardcode subdirs 
            demos_dir = session.joinpath('demos')# 硬编码子目录结构，创建一个名为"demos"的目录路径
            mp4_paths = list(demos_dir.glob("demo_*/raw_video.mp4"))# 查找"demos"目录下所有名为"raw_video.mp4"的文件
            if len(mp4_paths) < 1: # 如果找到的MP4文件数量小于1，则跳过当前循环
                continue
            all_meta = et.get_tags(mp4_paths, ['QuickTime:AutoRotation'])# 使用ExifTool获取所有MP4文件的视频旋转信息
            for mp4_path, meta in zip(mp4_paths, all_meta):# 遍历MP4文件和它们的元数据，检查AutoRotation值是否为U（表示未旋转
                rot = meta['QuickTime:AutoRotation']
                if rot != 'U':# 如果发现旋转的视频，打印会话名称和演示目录名称
                    demo_dir = mp4_path.parent
                    print(f"Found rotated video: {session.name} {demo_dir.name}")

"""
如果这个脚本作为主程序运行，检查命令行参数的数量。
如果没有提供参数，显示帮助信息。
如果提供了参数，调用main函数
"""
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
