"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
使用命令行参数和Docker来处理视频文件，并运行SLAM（Simultaneous Localization and Mapping）以生成相机轨迹和地图
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

# %%
# 导入更多模块，这些模块用于文件操作、命令行接口、子进程调用、多进程处理、并发执行、数值计算和图像处理。
import click
import concurrent.futures
import cv2
import multiprocessing
import numpy as np
import pathlib
import subprocess
from tqdm import tqdm
from umi.common.cv_util import draw_predefined_mask

# %%
# 使用click模块定义命令行接口
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=True, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
# 定义主函数，它将接收命令行参数
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask):
    # 确认输入目录中存在视频文件和IMU数据文件
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()
    # 设置地图文件的路径，并创建父目录（如果不存在）
    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)
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
    # 设置Docker容器中的挂载路径和文件名
    mount_target = pathlib.Path('/data')
    csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
    video_path = mount_target.joinpath('raw_video.mp4')
    json_path = mount_target.joinpath('imu_data.json')
    mask_path = mount_target.joinpath('slam_mask.png')
    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
        slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
        slam_mask = draw_predefined_mask(
            slam_mask, color=255, mirror=True, gripper=False, finger=True)
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)
    # 如果需要，创建一个掩模图像来遮挡抓手和镜子，并保存到文件中
    map_mount_source = pathlib.Path(map_path)
    map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

    # 构建运行SLAM的Docker命令
    cmd = [
        'docker',
        'run',
        '--rm', # delete after finish
        '--volume', str(video_dir) + ':' + '/data',
        '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml',
        '--input_video', str(video_path),
        '--input_imu_json', str(json_path),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_target)
    ]
    # 如果使用了掩模，则将掩模参数添加到命令中
    if not no_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])
    # 设置标准输出和错误输出的文件路径。这些文件将保存在视频目录中，分别用于存储Docker命令的标准输出和错误输出
    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')
    # 使用`subprocess.run()`函数执行Docker命令。`cmd`是之前构建的命令列表。`cwd`参数指定了子进程的工作目录，这里是视频目录的路径。`stdout`和`stderr`参数分别指定了子进程的标准输出和错误输出的文件对象，这些文件将被写入到之前设置的`stdout_path`和`stderr_path`。
    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    # 打印`subprocess.run()`函数的返回结果。这个结果是一个`CompletedProcess`对象，它包含了命令的退出码、标准输出和错误输出的内容（如果命令成功执行）
    print(result)

# %%
if __name__ == "__main__":
    main()
