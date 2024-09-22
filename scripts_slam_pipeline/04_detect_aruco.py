"""
这段代码通过定义一个命令行界面，允许用户指定视频文件夹、相机内参文件、ArUco配置文件，以及并发工作线程数。它首先搜索指定目录下的视频文件，然后为每个找到的视频启动一个处理任务，这些任务在一个线程池中并发运行。
每个任务调用一个外部脚本(`detect_aruco.py`)来处理单个视频文件，识别视频中的ArUco标记，并将检测到的标记信息保存为`.pkl`文件。
python scripts_slam_pipeline/04_detect_aruco.py \
-i data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room/demos \
-ci data_workspace/toss_objects/20231113/calibration/gopro_intrinsics_2_7k.json \
-ac data_workspace/toss_objects/20231113/calibration/aruco_config.yaml
处理通过摄像头捕捉的视频，识别其中的 ArUco 标记，并将检测到的标记信息保存到文件中
"""
# 导入必要的模块
import sys
import os
# 获取脚本的根目录，并将其添加到系统路径中，以便可以导入其他模块
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)# 将当前工作目录更改为脚本的根目录

# %%
import click  # 用于创建命令行界面
import concurrent.futures  # 用于并发执行
import multiprocessing  # 用于并行处理
import pathlib  # 用于处理路径
import subprocess  # 用于调用子进程
from tqdm import tqdm  # 用于显示进度条


# 需要标定相机内参文件和ArUco配置文件


# %%
@click.command()  # 定义一个click命令
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')    # 定义输入目录选项
@click.option('-ci', '--camera_intrinsics', required=True, help='Camera intrinsics json file (2.7k)')  # 定义相机内参文件选项
@click.option('-ac', '--aruco_yaml', required=True, help='Aruco config yaml file')      # 定义ArUco配置文件选项
@click.option('-n', '--num_workers', type=int, default=None)                            # 定义工作线程数选项
def main(input_dir, camera_intrinsics, aruco_yaml, num_workers):                        # 主函数
    input_dir = pathlib.Path(os.path.expanduser(input_dir))                             # 解析并扩展输入目录路径
    input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]            # 查找含有raw_video.mp4的目录
    print(f'Found {len(input_video_dirs)} video dirs')                                  # 打印找到的视频目录数
    # 确保相机内参文件和ArUco配置文件存在
    assert os.path.isfile(camera_intrinsics)
    assert os.path.isfile(aruco_yaml)
    # 如果没有指定工作线程数，则使用CPU的核心数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco.py')  # 获取detect_aruco.py脚本的路径

    with tqdm(total=len(input_video_dirs)) as pbar:                 # 创建一个进度条
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:        # 使用ThreadPoolExecutor并行处理每个视频
            futures = set()
            for video_dir in tqdm(input_video_dirs):                # 遍历每个视频目录
                video_dir = video_dir.absolute()                    # 获取绝对路径
                video_path = video_dir.joinpath('raw_video.mp4')    # 获取视频文件路径
                pkl_path = video_dir.joinpath('tag_detection.pkl')  # 设置输出的pkl文件路径
                if pkl_path.is_file():                              # 如果pkl文件已存在，则跳过
                    print(f"tag_detection.pkl already exists, skipping {video_dir.name}")
                    continue
                # 构建运行detect_aruco.py脚本的命令
                cmd = [
                    'python', script_path,
                    '--input', str(video_path),
                    '--output', str(pkl_path),
                    '--intrinsics_json', camera_intrinsics,
                    '--aruco_yaml', aruco_yaml,
                    '--num_workers', '1'
                ]
                # 如果当前任务数达到了线程池上限，则等待至少一个任务完成
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))                     # 更新进度条
                # 提交任务到线程池
                futures.add(executor.submit(
                    lambda x: subprocess.run(x, 
                        capture_output=True), 
                    cmd))
            # 等待所有任务完成
            completed, futures = concurrent.futures.wait(futures)            
            pbar.update(len(completed))  # 更新进度条

    print("Done! Result:") # 打印完成后的结果
    print([x.result() for x in completed])# 打印每个任务的结果

# %%
if __name__ == "__main__":
    main()
