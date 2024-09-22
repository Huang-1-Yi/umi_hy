# %%
import sys
import os

# 设置项目的根目录，并更新Python的路径，以便能够正确地导入项目中的模块
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %% 导入Click库用于命令行参数处理,OpenCV库（虽然在这个脚本中似乎没有使用），time模块用于时间相关操作，NumPy库用于数学运算，deque用于创建一个双向队列，tqdm用于在命令行中显示进度条，SharedMemoryManager用于在多进程之间共享内存，WSGController是一个与机器人抓手交互的控制器类，precise_sleep是一个用于精确控制睡眠时间的函数，get_latency用于计算延迟，matplotlib用于绘图
import click
import cv2
import time
import numpy as np
from collections import deque
from tqdm import tqdm
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.wsg_controller import WSGController
from umi.common.precise_sleep import precise_sleep
from umi.common.latency_util import get_latency
from matplotlib import pyplot as plt

# %%使用Click库定义命令行接口，并设置三个命令行选项：hostname（抓手控制器的IP地址），port（端口号），frequency（抓手操作的频率）。
@click.command()
@click.option('-h', '--hostname', default='192.168.1.20')
@click.option('-p', '--port', type=int, default=1000)
@click.option('-f', '--frequency', type=float, default=30)
def main(hostname, port, frequency):
    # 定义测试的持续时间，采样时间间隔，计算采样点的数量，并创建一个时间数组
    # 然后，根据正弦函数计算抓手宽度变化的目标值
    duration = 10.0
    sample_dt = 1 / 100
    k = int(duration / sample_dt)
    sample_t = np.linspace(0, duration, k)
    value = np.sin(sample_t * duration / 1.5) * 0.5 + 0.5
    width = value * 80

    # 使用SharedMemoryManager创建一个共享内存管理器，并实例化WSGController。设置抓手控制器的参数，并启动等待
    with SharedMemoryManager() as shm_manager:
        with WSGController(
            
            shm_manager=shm_manager,
            hostname=hostname,
            port=port,
            frequency=frequency,
            move_max_speed=200.0,
            get_max_k=int(k*1.2),
            command_queue_size=int(k*1.2),
            verbose=False) as gripper:
            gripper.start_wait()
            # 为抓手设置初始目标宽度，并在1秒后开始移动
            gripper.schedule_waypoint(width[0], time.time() + 0.3)
            precise_sleep(1.0)
            # 为抓手设置一系列的目标宽度，并计算对应的时间戳。
            # 将这些目标宽度和时间戳发送给抓手控制器，并在测试结束后等待一段时间
            timestamps = time.time() + sample_t + 1.0
            for i in range(k):
                gripper.schedule_waypoint(width[i], timestamps[i])
                time.sleep(0.0)
            precise_sleep(duration + 1.0)
            # 从抓手控制器获取所有的状态信息
            states = gripper.get_all_state()
    # 使用get_latency函数计算系统的端到端延迟，并打印出来
    latency, info = get_latency(
        x_target=width,
        t_target=timestamps,
        x_actual=states['gripper_position'],
        t_actual=states['gripper_receive_timestamp']
    )
    print(f"End-to-end latency: {latency}sec")

    # plot everything 创建一个matplotlib图窗，并设置其大小
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(15, 5, forward=True)
    # 使用matplotlib绘制了三幅图表：第一幅图显示了延迟与相关性的关系，
    ax = axes[0]
    ax.plot(info['lags'], info['correlation'])
    ax.set_xlabel('lag')
    ax.set_ylabel('cross-correlation')
    ax.set_title("Cross Correlation")
    # 第二幅图显示了抓手宽度的目标值与实际值随时间的变化
    ax = axes[1]
    ax.plot(timestamps, width, label='target')
    ax.plot(states['gripper_receive_timestamp'], states['gripper_position'], label='actual')
    ax.set_xlabel('time')
    ax.set_ylabel('gripper-width')
    ax.legend()
    ax.set_title("Raw observation")
    # 第三幅图显示了在考虑延迟的情况下，目标值与实际值的对齐情况。
    ax=axes[2]
    t_samples = info['t_samples'] - info['t_samples'][0]
    ax.plot(t_samples, info['x_target'], label='target')
    ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
    ax.set_xlabel('time')
    ax.set_ylabel('gripper-width')
    ax.legend()
    ax.set_title(f"Aligned with latency={latency}")
    # 最后，显示所有的图表
    plt.show()

# %%
if __name__ == '__main__':
    main()
