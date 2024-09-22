"""
python scripts_slam_pipeline/00_process_videos.py data_workspace/toss_objects/20231113
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import shutil
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:# 遍历 session_dir 中的每个目录,将其转换为 pathlib 对象,并确保目录是绝对路径
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        # hardcode subdirs 硬编码两个子目录：input_dir 和 output_dir。这些目录将用于存放原始视频和演示视频
        input_dir = session.joinpath('raw_videos')
        output_dir = session.joinpath('demos')
        
        # create raw_videos if don't exist 如果input_dir不存在则创建该目录，并将 session 目录中所有的 .mp4 文件移动到 input_dir 中。
        if not input_dir.is_dir():
            input_dir.mkdir(parents=True, exist_ok=True)# input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside.")
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)
                # 检查目标目录中是否已经存在同名文件
                if out_path.exists():
                    # 生成一个唯一的新文件名
                    new_name = f"{out_path.stem}_copy{out_path.suffix}"
                    new_out_path = out_path.with_name(new_name)
                    
                    # 移动文件到新路径
                    shutil.move(mp4_path, new_out_path)
                    print(f"File {mp4_path.name} already exists. Renamed to {new_name}.")
                else:
                    # 如果没有同名文件，则正常移动文件
                    shutil.move(mp4_path, out_path)
                    print(f"Moved file {mp4_path.name}.")
        
        # create mapping video if don't exist 如果mapping.mp4不存在，查找 input_dir 中最大的 .mp4 文件，并将它重命名为 mapping.mp4
        # huangyi: 确保在尝试查找最大文件之前，input_dir 目录中确实存在文件
        if not input_dir.is_dir() or not any(input_dir.glob('**/*.MP4')):
            print(f"No .mp4 files found in {input_dir}")
            continue
        mapping_vid_path = input_dir.joinpath('mapping.mp4')
        if (not mapping_vid_path.exists()) and not(mapping_vid_path.is_symlink()):
            max_size = -1
            max_path = None
            for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                size = mp4_path.stat().st_size
                if size > max_size:
                    max_size = size
                    max_path = mp4_path

            if max_path is None:# huangyi: 确保 max_path 不是 None
                print("max_path is None. No valid .mp4 files found.")
                continue

            print("max_path:", max_path)  # huangyi:添加这行代码来打印出 max_path 的值
            shutil.move(max_path, mapping_vid_path)
            print(f"raw_videos/mapping.mp4 don't exist! Renaming largest file {max_path.name}.")
        
        # create gripper calibration video if don't exist如果不存在gripper_calibration目录，它将创建该目录，并准备使用每个相机序列的第一个视频进行填充
        gripper_cal_dir = input_dir.joinpath('gripper_calibration')
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print("raw_videos/gripper_calibration don't exist! Creating one with the first video of each camera serial.")
            # 始化了两个字典 serial_start_dict 和 serial_path_dict，并将 ExifToolHelper 上下文管理器用于处理 .mp4 文件的元数据。对于每个 .mp4 文件，如果它的名称以 map 开头，则跳过。
            serial_start_dict = dict()
            serial_path_dict = dict()
            with ExifToolHelper() as et:
                for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                    if mp4_path.name.startswith('map'):
                        continue
                    # 从每个 .mp4 文件中提取开始日期和相机序列号
                    start_date = mp4_get_start_datetime(str(mp4_path))
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    cam_serial = meta['QuickTime:CameraSerialNumber']
                    # 更新字典 serial_start_dict 和 serial_path_dict，其中包含每个相机序列号及其对应的最早开始日期和 .mp4 文件路径
                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path
            # 遍历 serial_path_dict，为每个相机序列号移动对应的 .mp4 文件到 gripper_cal_dir 目录。
            for serial, path in serial_path_dict.items():
                print(f"Selected {path.name} for camera serial {serial}")
                out_path = gripper_cal_dir.joinpath(path.name)
                shutil.move(path, out_path)

        # look for mp4 video in all subdirectories in input_dir 查找 input_dir 及其所有子目录中的所有 .mp4 视频文件，并打印找到的视频数量。
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')
        
        # 使用 ExifToolHelper 来处理剩余的 .mp4 文件m为每个文件创建一个目录，然后移动文件并创建一个指向原始位置的符号链接。如果文件是一个符号链接，则跳过它，因为它们已经被移动过。
        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue
                # 从每个 .mp4 文件中提取开始日期和相机序列号，并创建一个输出目录的名称
                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # special folders 根据文件名或父目录名称，为 .mp4 文件创建一个特殊的输出目录名称。
                if mp4_path.name.startswith('mapping'):
                    out_dname = "mapping"
                elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                
                # create directory 这行代码为每个 .mp4 文件创建一个输出目录
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # move videos 将 .mp4 文件移动到为其创建的输出目录
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # create symlink back from original location 为移动后的 .mp4 文件创建一个符号链接，指向其在原始位置的相对路径。
                # relative_to's walk_up argument is not avaliable until python 3.12
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                mp4_path.symlink_to(symlink_path)

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
