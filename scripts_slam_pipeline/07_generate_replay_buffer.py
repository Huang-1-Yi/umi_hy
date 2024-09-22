# 用于处理视频数据和生成机器学习训练数据集的脚本。
# 从多源视频数据中自动提取和处理机器学习训练所需的数据,包括机器人的末端执行器(End-Effector, EEF)位置、朝向、夹持器宽度以及相应的视频帧,
# 并以高效的Zarr格式存储作为数据集,以便于后续的训练过程使用，大大简化和加速机器学习项目的数据准备阶段。
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import av
import click
import concurrent.futures
import cv2
import json
import multiprocessing
import numpy as np
import pathlib
import pickle
import zarr
from tqdm import tqdm
from collections import defaultdict
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

"""
导入用于基本文件和系统操作的模块（sys, os, pathlib）,处理JSON格式数据的json模块,命令行界面创建的click模块,高性能数组库numpy,图像处理的cv2和视频处理的av,以及用于多进程和并发执行的模块。
from umi.common.cv_util import ... 导入了一些自定义的计算机视觉工具函数,用于图像处理。
导入并初始化用于机器学习数据预处理和压缩的模块和函数。
"""

"""
使用Click定义了一个命令行界面,用于处理输入参数和选项。
参数包括输入路径、输出路径、输出分辨率、视场角（FOV）、压缩级别、是否禁用镜像、是否交换镜像以及工作线程数
"""
"""
主函数执行流程
准备输出目录和处理输入参数：检查输出文件是否存在并可能提示覆盖,解析输出分辨率等。

设置多线程和图像处理参数：根据需要的输出视场角和分辨率初始化鱼眼相机的校正转换器,创建空的重放缓冲区。

处理每个输入路径：遍历输入路径,加载数据集计划,并对每个计划中的视频进行处理。对于每个视频,提取关于机器人夹持器的信息和相应的视频帧。

视频帧处理：定义video_to_zarr函数用于将视频帧转换并存储到Zarr数据集中。这包括标签去除、镜像处理、图像压缩等步骤。

并行处理视频：使用ThreadPoolExecutor并发处理多个视频,提高数据处理效率。

保存数据到Zarr格式：将处理后的数据保存到指定的Zarr输出路径。
"""

# %%
# 参数包括输入路径、输出路径、输出分辨率、视场角（FOV）、压缩级别、是否禁用镜像、是否交换镜像以及工作线程数
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, out_res, out_fov, compression_level, 
         no_mirror, mirror_swap, num_workers):
    # 检查输出文件是否存在并提示是否覆盖
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
    # 解析输出分辨率    
    out_res = tuple(int(x) for x in out_res.split(','))
    # 设置多线程和图像处理参数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
    # 如果指定了输出视场角，初始化鱼眼相机的校正转换器。        
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
    # 创建一个空的重放缓冲区，用于存储处理后的数据
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # 遍历输入路径，加载数据集计划，并对每个计划中的视频进行处理 dump lowdim data to replay buffer generate argumnet for videos
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        plan = pickle.load(plan_path.open('rb'))
        # 对每个视频，提取关于机器人夹持器的信息和相应的视频帧。
        # 检查所有集的夹持器数量是否相同。
        # 为每个夹持器和摄像头提取数据，并将其添加到重放缓冲区中。
        videos_dict = defaultdict(list)
        for plan_episode in plan:
            grippers = plan_episode['grippers']
            
            # check that all episodes have the same number of grippers 
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
                
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
                
            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):    
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[...,:3]
                eef_rot = eef_pose[...,3:]
                gripper_widths = gripper['gripper_width']
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']
                
                robot_name = f'robot{gripper_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose
            
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # aggregate video gen aguments
            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()
                
                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)
                
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })
            buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")
    
    # 获取视频的图像大小。get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    # 创建用于存储图像数据的 Zarr 数据集，并设置压缩级别。dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    # 将视频帧转换并存储到 Zarr 数据集中。这包括标签去除、镜像处理、图像压缩等步骤。
    def video_to_zarr(replay_buffer, mp4_path, tasks):
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        curr_task_idx = 0
        
        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
            is_mirror = (mirror_mask[...,0] == 0)
        
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')

                    # inpaint tags
                    this_det = tag_detection_results[frame_idx]
                    all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    for corners in all_corners:
                        img = inpaint_tag(img, corners)
                        
                    # mask out gripper
                    img = draw_predefined_mask(img, color=(0,0,0), 
                        mirror=no_mirror, gripper=True, finger=False)
                    # resize
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)
                        
                    # handle mirror swap
                    if mirror_swap:
                        img[is_mirror] = img[:,::-1,:][is_mirror]
                        
                    # compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False
    
    # 使用 ThreadPoolExecutor 并发处理多个视频，提高数据处理效率                
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # 将处理后的数据保存到指定的 Zarr 输出路径 dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")

# %%
if __name__ == "__main__":
    main()
