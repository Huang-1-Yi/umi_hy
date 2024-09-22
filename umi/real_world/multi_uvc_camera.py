# 导入必要的模块和类型定义。typing模块用于类型注解，numbers用于数值类型检查，copy用于复制对象，time用于时间操作，pathlib用于文件路径操作，
# SharedMemoryManager用于共享内存管理，numpy用于数值计算，UvcCamera和VideoRecorder可能是自定义的类
from typing import List, Optional, Union, Dict, Callable
import numbers
import copy
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi.real_world.uvc_camera import UvcCamera
from umi.real_world.video_recorder import VideoRecorder

class MultiUvcCamera:
    # 设备文件路径列表、共享内存管理器、分辨率、帧率等设置，以及一些转换函数和视频录制器
    def __init__(self,
            # v4l2 device file path
            # e.g. /dev/video0
            # or /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0
            dev_video_paths: List[str],
            shm_manager: Optional[SharedMemoryManager]=None,
            resolution=(1280,720),
            capture_fps=60,
            put_fps=None,
            put_downsample=True,
            get_max_k=30,
            receive_latency=0.0,
            cap_buffer_size=1,
            transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
            verbose=False
        ):
        super().__init__()  # 初始化继承自的任何父类属性

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        n_cameras = len(dev_video_paths)    # 获取摄像头数量，即设备文件路径列表的长度

        
        # verbose =True # 强行调试
        # if verbose == True:
        #     print("MultiUvcCamera.dev_video_paths == {}".format(dev_video_paths)) 
        #     print("MultiUvcCamera.resolution == {}".format(resolution)) 
        #     print("MultiUvcCamera.capture_fps == {}".format(capture_fps))
        #     print("MultiUvcCamera.get_max_k == {}".format(get_max_k))
        #     print("MultiUvcCamera.receive_latency == {}".format(receive_latency))
        #     print("MultiUvcCamera.cap_buffer_size == {}".format(cap_buffer_size))
        #     # print("dev_video_paths == {}".format(dev_video_paths))
        #     # print("dev_video_paths == {}".format(dev_video_paths))       
        
        
        
        # 初始化cameras字典，并为每个摄像头创建一个UvcCamera实例
        resolution          = repeat_to_list(resolution, n_cameras, tuple)
        capture_fps         = repeat_to_list(capture_fps, n_cameras, (int, float))
        cap_buffer_size     = repeat_to_list(cap_buffer_size, n_cameras, int)
        transform           = repeat_to_list(transform, n_cameras, Callable)
        vis_transform       = repeat_to_list(vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(recording_transform, n_cameras, Callable)
        video_recorder      = repeat_to_list(video_recorder, n_cameras, VideoRecorder)
        
        cameras = dict()
        for i, path in enumerate(dev_video_paths):
            # print("enumerate(),dev_video_paths== {}".format(dev_video_paths))
            cameras[path] = UvcCamera(
                shm_manager=shm_manager,
                dev_video_path=path,
                resolution=resolution[i],
                capture_fps=capture_fps[i],
                put_fps=put_fps,
                put_downsample=put_downsample,
                get_max_k=get_max_k,
                receive_latency=receive_latency,
                cap_buffer_size=cap_buffer_size[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose
            )
        self.cameras = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        results = list()                    # 初始化一个空的列表 results 来存储每个摄像头的输出
        for i, camera in enumerate(self.cameras.values()):  # 遍历 self.cameras 字典中的每个摄像头
            this_out = None
            if out is not None:             # 如果 out 参数不为 None，则使用切片技巧来保持数组
                this_out = dict()
                for key, v in out.items():  # 使用切片技巧来保持数组，当 v 是 1D 时,将切片后的数据放入 this_out 字典中
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out) # 调用 camera.get_vis 方法获取每个摄像头的视频数据
            # print(f"Camera {i} output:", this_out)
            # print(f"Camera {i} output shape:", this_out['color'].shape)
            if out is None:                 # 如果 out 参数为 None，则将每个摄像头的输出添加到 results 列表中
                results.append(this_out)
        if out is None:                     # 如果 out 参数为 None，则创建一个字典 out，并合并所有摄像头的数据
            out = dict()
            for key in results[0].keys():   # 遍历第一个结果字典的键，创建 out 字典
                out[key] = np.stack([x[key] for x in results])# 将所有结果字典的值合并到 out 字典中
        return out                          # 返回包含所有摄像头的视频数据的 out 字典


    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)
    
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [copy.deepcopy(x) for _ in range(n)]
    assert len(x) == n
    return x
