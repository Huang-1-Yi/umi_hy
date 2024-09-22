# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib# pathlib 是一个现代的文件路径操作库。
import pickle# pickle 用于序列化和反序列化Python对象。
import collections# collections 提供了额外的数据结构，如defaultdict。
import click# click 是一个用于创建命令行接口的库。

# %%
"""
接受一个或多个输入路径作为参数。
函数内部，它遍历每个输入路径，检查是否存在dataset_plan.pkl文件。如果存在，它读取该文件中的计划，并遍历每个计划中的剧集。
对于每个剧集，它检查是否有相同数量的抓取器和摄像机，并收集视频生成所需的参数。
最后，它计算总共有多少个视频和剧集，并打印出来。
"""
@click.command()
@click.argument('input', nargs=-1)
def main(input):
    n_videos = 0
    n_episode = 0
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))
        
        # generate argumnet for videos 为视频生成argument
        n_grippers = None
        n_cameras = None
        buffer_start = 0
        videos_dict = collections.defaultdict(list)
        for plan_episode in plan:
            n_episode += 1
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
            
            # aggregate video gen aguments聚合视频生成
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
        
        # vid_args.extend(videos_dict.items()) 将 videos_dict.items() 返回的所有键值对元组添加到 vid_args 列表
        n_videos += len(videos_dict)
    print(f"{n_videos} videos and {n_episode} episodes in total.")

# %%
if __name__ == "__main__":
    main()
