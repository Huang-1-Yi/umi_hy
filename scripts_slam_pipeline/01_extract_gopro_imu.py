"""
python scripts_slam_pipeline/01_extract_gopro_imu.py data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room
"""
# %%
# 使用命令行参数和Docker来并行处理GoPro视频文件，提取IMU数据
import sys
import os
# 设置项目的根目录，并将该目录添加到Python的模块搜索路径中，以确保可以正确导入项目中的模块。
# 同时将当前工作目录切换到根目录。
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
# 导入更多模块，这些模块用于文件操作、命令行接口、子进程调用、多进程处理和并发执行。
import click
import concurrent.futures
import multiprocessing
import pathlib
import subprocess
from tqdm import tqdm

# %%
# 使用click模块定义命令行接口。
@click.command()
@click.option('-d', '--docker_image', default="chicheng/openicc:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-np', '--no_docker_pull', is_flag=True, default=True, help="pull docker image from docker hub")
@click.argument('session_dir', nargs=-1)
# 定义主函数，它将接收命令行参数。
def main(docker_image, num_workers, no_docker_pull, session_dir):
    # 如果未指定工作进程数，则使用CPU的核心数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    # 如果没有指定不拉取Docker镜像，则从Docker Hub拉取指定的Docker镜像。
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
    # 遍历指定的会话目录
    for session in session_dir:
        input_dir = pathlib.Path(os.path.expanduser(session)).joinpath('demos')
        input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
        print(f'Found {len(input_video_dirs)} video dirs')
         # 使用tqdm创建一个进度条，总进度为找到的视频目录数
        with tqdm(total=len(input_video_dirs)) as pbar:
            # one chunk per thread, therefore no synchronization needed 每个线程一个块，因此不需要同步
            # 创建一个线程池执行器，最大工作线程数为num_workers 
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                # 遍历视频目录，为每个目录提交一个任务到线程池
                for video_dir in tqdm(input_video_dirs):
                    video_dir = video_dir.absolute()
                    # 如果IMU数据文件已存在，则跳过该目录。
                    if video_dir.joinpath('imu_data.json').is_file():
                        print(f"imu_data.json already exists, skipping {video_dir.name}")
                        continue
                    # 设置Docker容器中的挂载路径和文件名
                    mount_target = pathlib.Path('/data')
                    video_path = mount_target.joinpath('raw_video.mp4')
                    json_path = mount_target.joinpath('imu_data.json')
                    # run imu extractor # 构建提取IMU数据的Docker命令
                    cmd = [
                        'docker',
                        'run',
                        '--rm', # delete after finish
                        '--volume', str(video_dir) + ':' + '/data',
                        docker_image,
                        'node',
                        '/OpenImuCameraCalibrator/javascript/extract_metadata_single.js',
                        str(video_path),
                        str(json_path)
                    ]
                    # 设置标准输出和错误输出的文件路径
                    stdout_path = video_dir.joinpath('extract_gopro_imu_stdout.txt')
                    stderr_path = video_dir.joinpath('extract_gopro_imu_stderr.txt')
                    # 如果当前运行的任务数达到工作线程数，等待至少一个任务完成
                    if len(futures) >= num_workers:
                        # 提交任务到线程池，并记录未来对象
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))
                    # 打印Docker命令，用于调试
                    futures.add(executor.submit(
                        lambda x, stdo, stde: subprocess.run(x, 
                            cwd=str(video_dir),
                            stdout=stdo.open('w'),
                            stderr=stde.open('w')), 
                        cmd, stdout_path, stderr_path))
                    # print(' '.join(cmd)) # 打印Docker命令，用于调试
                 # 等待所有任务完成，并更新进度条
                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))
        # 处理完成，打印结果
        print("Done! Result:")
        print([x.result() for x in completed])
# %%
if __name__ == "__main__":
    main()
