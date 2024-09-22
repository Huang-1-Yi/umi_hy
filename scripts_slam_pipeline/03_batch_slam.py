"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/fold_cloth_20231214/demos
用于处理视频文件并运行SLAM（Simultaneous Localization and Mapping）以生成相机轨迹和地图
"""
# %%
# 导入必要的Python模块
import sys
import os
# 设置项目的根目录，并将该目录添加到Python的模块搜索路径中，以确保可以正确导入项目中的模块。
# 同时将当前工作目录切换到根目录。
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
# 导入更多模块，这些模块用于文件操作、命令行接口、子进程调用、多进程处理、并发执行、图像处理和视频处理。
# %%
import av
import click
import concurrent.futures
import cv2
import multiprocessing
import numpy as np
import pathlib
import subprocess
from tqdm import tqdm
from umi.common.cv_util import draw_predefined_mask
# %% 定义一个函数runner，用于执行子进程命令，并处理输出和超时
def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
    try:
        return subprocess.run(cmd,                       
            cwd=str(cwd),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w'),
            timeout=timeout,
            **kwargs)
    except subprocess.TimeoutExpired as e:
        return e

# %% 使用click模块定义命令行接口。
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=True, help="pull docker image from docker hub")
# 定义主函数，它将接收命令行参数
def main(input_dir, map_path, docker_image, num_workers, max_lost_frames, timeout_multiple, no_docker_pull):
    # 设置输入目录的绝对路径
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')
    # 如果没有指定地图文件路径，则默认为输入目录下的map文件夹中的map_atlas.osa文件
    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file()
    # 如果未指定工作进程数，则使用CPU的核心数的一半
    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2
    # 如果没有指定不拉取Docker镜像，则从Docker Hub拉取指定的Docker镜像
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)
    # 使用tqdm创建一个进度条，总进度为找到的视频目录数
    with tqdm(total=len(input_video_dirs)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:    # 创建一个线程池执行器，最大工作线程数为num_workers
            futures = set()
            for video_dir in tqdm(input_video_dirs):                # 遍历视频目录，为每个目录提交一个任务到线程池
                video_dir = video_dir.absolute()
                if video_dir.joinpath('camera_trajectory.csv').is_file():
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    continue
                mount_target = pathlib.Path('/data')                # 设置Docker容器中的挂载路径和文件名
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                video_path = mount_target.joinpath('raw_video.mp4')
                json_path = mount_target.joinpath('imu_data.json')
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container: # 查找视频时长
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple
                # 创建一个掩模图像来遮挡抓手和镜子，并保存到文件中
                slam_mask = np.zeros((2028, 2704), dtype=np.uint8)  # 创建一个名为slam_mask的二维numpy数组，大小为2028x2704，数据类型为无符号8位整数（即每个元素的值范围为0-255），初始化为0
                slam_mask = draw_predefined_mask(   # 调用draw_predefined_mask函数，在slam_mask上绘制预定义的掩码
                    slam_mask, color=255, mirror=True, gripper=False, finger=True)# color=255表示绘制的颜色为白色（255表示最大亮度），mirror=True表示掩码是对称的，gripper=False表示不包含夹具部分，finger=True表示包含手指部分
                # 将处理后的掩码图像保存到磁盘，文件路径由mask_write_path提供
                cv2.imwrite(str(mask_write_path.absolute()), slam_mask)
                # 设置源地图路径map_mount_source和目标地图路径map_mount_target，后者通常是在容器内的路径
                map_mount_source = map_path
                map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

                # 构建命令行参数列表cmd，用于调用docker run命令来执行SLAM
                cmd = [
                    'docker',
                    'run',  
                    '--rm',                                     # 容器退出后自动删除
                    '--volume', str(video_dir) + ':' + '/data', # 挂载宿主机上的目录到容器内
                    '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
                    docker_image,                               # 要运行的Docker镜像名称
                    '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',# 容器内SLAM程序的路径
                    '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',                                                 # 词袋文件的路径
                    '--setting', '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml',  # 配置文件的路径
                    '--input_video', str(video_path),           # 输入视频的路径
                    '--input_imu_json', str(json_path),         # IMU（惯性测量单元）数据的路径
                    '--output_trajectory_csv', str(csv_path),   # 输出轨迹CSV文件的路径
                    '--load_map', str(map_mount_target),        # 加载地图的路径
                    '--mask_img', str(mask_path),               # 掩码图像的路径
                    '--max_lost_frames', str(max_lost_frames)   # 最大丢失帧数
                ]
                # 设置标准输出和错误输出的文件路径
                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')
                # 如果当前运行的任务数达到或超过设定的最大工作线程数num_workers，则等待至少一个任务完成
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))
                # 使用executor提交一个新的任务runner，runner是一个函数，它将执行命令行cmd，并将标准输出和错误输出重定向到指定的文件。timeout设置执行命令的超时时间
                # （可能是concurrent.futures.ThreadPoolExecutor或concurrent.futures.ProcessPoolExecutor的实例）
                futures.add(executor.submit(runner,
                    cmd, str(video_dir), stdout_path, stderr_path, timeout))
                # print(' '.join(cmd))
            # 等待所有任务完成，并更新进度条pbar
            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))
    # 最后，打印所有已完成任务的结果。每个任务的结果是从x.result()获得的，其中x是concurrent.futures.Future对象
    print("Done! Result:")
    print([x.result() for x in completed])

# %%
if __name__ == "__main__":
    main()
