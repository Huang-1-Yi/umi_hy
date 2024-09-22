from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class RealEnv:
    def __init__(self, 
            # required params
            output_dir,         # 输出目录
            robot_ip,           # 机器人IP地址
            # env params
            frequency=10,       # 控制频率
            n_obs_steps=2,      # 观察步数
            # obs
            obs_image_resolution=(640,480),         # 观察图像分辨率
            max_obs_buffer_size=30,                 # 最大观察缓冲区大小
            camera_serial_numbers=None,             # 相机序列号
            obs_key_map=DEFAULT_OBS_KEY_MAP,        # 观察键映射
            obs_float32=False,                      # 观察数据类型
            # action
            max_pos_speed=0.25,                     # 最大位置速度
            max_rot_speed=0.6,                      # 最大旋转速度
            # robot
            tcp_offset=0.13,                        # TCP偏移
            init_joints=False,                      # 是否初始化关节
            # video capture params
            video_capture_fps=30,                   # 视频捕捉帧率
            video_capture_resolution=(1280,720),    # 视频捕捉分辨率
            # saving params
            record_raw_video=True,                  # 是否记录原始视频，默认是
            thread_per_video=2,                     # 每个视频的线程数
            video_crf=21,                           # 视频质量
            # vis params
            enable_multi_cam_vis=True,              # 是否启用多相机可视化
            multi_cam_vis_resolution=(1280,720),    # 多相机可视化分辨率
            # shared memory
            shm_manager=None # 共享内存管理器
            ):
        assert frequency <= video_capture_fps           # 确保控制频率不超过视频捕捉帧率
        output_dir = pathlib.Path(output_dir)           # 将输出目录转换为Path对象
        assert output_dir.parent.is_dir()               # 确保输出目录的父目录存在
        video_dir = output_dir.joinpath('videos')       # 创建视频目录
        video_dir.mkdir(parents=True, exist_ok=True)    # 确保视频目录存在
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute()) # 创建重放缓冲区路径
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')              # 创建重放缓冲区

        if shm_manager is None:                     # 如果没有提供共享内存管理器
            shm_manager = SharedMemoryManager()     # 创建一个新的共享内存管理器
            shm_manager.start()                     # 启动共享内存管理器
        if camera_serial_numbers is None:           # 如果没有提供相机序列号
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial() # 获取连接的设备序列号
        
        # obs output rgb，需要由bgr转换为rgb
        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution, 
            bgr_to_rgb=True)                    # 获取图像转换函数
        color_transform = color_tf
        if obs_float32:                         # 如果观察数据类型为float32
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255 # 将图像转换为float32类型并归一化

        def transform(data):
            data['color'] = color_transform(data['color']) # 转换颜色数据
            return data
        
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        ) # 计算最佳行列数
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        ) # 获取可视化颜色转换函数
        def vis_transform(data):
            data['color'] = vis_color_transform(data['color']) # 转换可视化颜色数据
            return data

        recording_transfrom = None          # 初始化录制转换函数
        recording_fps = video_capture_fps   # 初始化录制帧率
        recording_pix_fmt = 'bgr24'         # 初始化录制像素格式
        if not record_raw_video:            # 如果不记录原始视频
            recording_transfrom = transform # 使用转换函数
            recording_fps = frequency       # 使用控制频率作为录制帧率
            recording_pix_fmt = 'rgb24'     # 使用RGB24像素格式

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps, 
            codec='h264',
            input_pix_fmt=recording_pix_fmt, 
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)  # 创建视频录制器

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # 在每一帧到达后立即发送send every frame immediately after arrival
            # 不考虑设置的帧率限制ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False
            ) # 创建Realsense多相机管理器
        
        multi_cam_vis = None        # 初始化多相机可视化
        if enable_multi_cam_vis:    # 如果启用多相机可视化
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            ) # 创建多相机可视化器

        cube_diag = np.linalg.norm([1,1,1])                     # 计算单位立方体的对角线长度
        j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi   # 初始化关节角度
        if not init_joints:                                     # 如果不初始化关节
            j_init = None                                       # 关节初始化设置为None

        robot = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125, # UR5 CB3 RTDE
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            launch_timeout=3,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0],
            payload_mass=None,
            payload_cog=None,
            joints_init=j_init,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size
            ) # 创建机器人控制器
        self.realsense = realsense                  # 设置Realsense
        self.robot = robot                          # 设置机器人
        self.multi_cam_vis = multi_cam_vis          # 设置多相机可视化
        self.video_capture_fps = video_capture_fps  # 设置视频捕捉帧率
        self.frequency = frequency                  # 设置控制频率
        self.n_obs_steps = n_obs_steps              # 设置观察步数
        self.max_obs_buffer_size = max_obs_buffer_size # 设置最大观察缓冲区大小
        self.max_pos_speed = max_pos_speed          # 设置最大位置速度
        self.max_rot_speed = max_rot_speed          # 设置最大旋转速度
        self.obs_key_map = obs_key_map              # 设置观察键映射
        # recording
        self.output_dir = output_dir            # 设置输出目录
        self.video_dir = video_dir              # 设置视频目录
        self.replay_buffer = replay_buffer      # 设置重放缓冲区
        # temp memory buffers
        self.last_realsense_data = None         # 初始化最后一次Realsense数据
        # recording buffers
        self.obs_accumulator = None             # 初始化观察累加器
        self.action_accumulator = None          # 初始化动作累加器
        self.stage_accumulator = None           # 初始化阶段累加器

        self.start_time = None                  # 初始化开始时间
    
# ======== start-stop API =============
    @property
    # 检查 Realsense 相机和机器人是否准备好
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready # 检查Realsense和机器人是否准备好
    
    # 启动 Realsense 相机和机器人控制器，如果启用了多相机可视化，也启动多相机可视化
    def start(self, wait=True):
        self.realsense.start(wait=False)        # 启动Realsense
        self.robot.start(wait=False)            # 启动机器人
        if self.multi_cam_vis is not None:      # 如果启用了多相机可视化
            self.multi_cam_vis.start(wait=False)# 启动多相机可视化
        if wait:
            self.start_wait()                   # 等待启动完成

    # 停止 Realsense 相机和机器人控制器，如果启用了多相机可视化，也停止多相机可视化
    def stop(self, wait=True):
        self.end_episode()                      # 结束当前记录
        if self.multi_cam_vis is not None:      # 如果启用了多相机可视化
            self.multi_cam_vis.stop(wait=False) # 停止多相机可视化
        self.robot.stop(wait=False)             # 停止机器人
        self.realsense.stop(wait=False)         # 停止Realsense
        if wait:
            self.stop_wait()                    # 等待停止完成

    # 等待 Realsense 相机和机器人控制器启动完成，如果启用了多相机可视化，也等待其启动完成
    def start_wait(self):
        self.realsense.start_wait()             # 等待Realsense启动完成
        self.robot.start_wait()                 # 等待机器人启动完成
        if self.multi_cam_vis is not None:      # 如果启用了多相机可视化
            self.multi_cam_vis.start_wait()     # 等待多相机可视化启动完成
    
    # 等待 Realsense 相机和机器人控制器停止完成，如果启用了多相机可视化，也等待其停止完成
    def stop_wait(self):
        self.robot.stop_wait()                  # 等待机器人停止完成
        self.realsense.stop_wait()              # 等待Realsense停止完成
        if self.multi_cam_vis is not None:      # 如果启用了多相机可视化
            self.multi_cam_vis.stop_wait()      # 等待多相机可视化停止完成

    # ========= context manager ===========
    # 在上下文管理器使用时，启动环境
    def __enter__(self):
        self.start()
        return self
    
    # 在上下文管理器结束时，停止环境
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    # 获取当前的观察数据，包括相机图像和机器人状态
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready        # 确保环境已准备好

        # get data获取数据
        # 30 Hz, camera_receive_timestamp 摄像头接收时间戳
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(
            k=k, 
            out=self.last_realsense_data)       # 获取Realsense数据

        # 125 hz, robot_receive_timestamp 机器人接收时间戳 
        last_robot_data = self.robot.get_all_state() # 获取机器人所有状态
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        # 获取最后一个时间戳
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()]) 
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt) # 对齐观察时间戳

        camera_obs = dict() # 初始化相机观察字典
        for camera_idx, value in self.last_realsense_data.items(): # 遍历每个相机的数据
            this_timestamps = value['timestamp'] # 获取当前相机的时间戳
            this_idxs = list() # 初始化索引列表
            for t in obs_align_timestamps: # 遍历对齐的时间戳
                is_before_idxs = np.nonzero(this_timestamps < t)[0] # 找到所有小于当前时间戳的索引
                this_idx = 0 # 初始化当前索引为0
                if len(is_before_idxs) > 0: # 如果存在小于当前时间戳的索引
                    this_idx = is_before_idxs[-1] # 取最后一个小于当前时间戳的索引
                this_idxs.append(this_idx) # 将索引添加到索引列表中
            # remap key
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs] # 将颜色数据映射到相机观察字典中

        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp'] # 获取机器人的时间戳
        this_timestamps = robot_timestamps # 将机器人的时间戳赋值给当前时间戳
        this_idxs = list() # 初始化索引列表
        for t in obs_align_timestamps: # 遍历对齐的时间戳
            is_before_idxs = np.nonzero(this_timestamps < t)[0] # 找到所有小于当前时间戳的索引
            this_idx = 0 # 初始化当前索引为0
            if len(is_before_idxs) > 0: # 如果存在小于当前时间戳的索引
                this_idx = is_before_idxs[-1] # 取最后一个小于当前时间戳的索引
            this_idxs.append(this_idx) # 将索引添加到索引列表中

        robot_obs_raw = dict() # 初始化机器人原始观察字典
        for k, v in last_robot_data.items(): # 遍历机器人的数据
            if k in self.obs_key_map: # 如果键在观察键映射中
                robot_obs_raw[self.obs_key_map[k]] = v # 将值映射到机器人原始观察字典中

        robot_obs = dict() # 初始化机器人观察字典
        for k, v in robot_obs_raw.items(): # 遍历机器人原始观察字典
            robot_obs[k] = v[this_idxs] # 将数据映射到机器人观察字典中

        # accumulate obs
        if self.obs_accumulator is not None: # 如果观察累加器不为空
            self.obs_accumulator.put( # 将机器人原始观察数据和时间戳放入观察累加器
                robot_obs_raw,
                robot_timestamps
            )

        # return obs
        obs_data = dict(camera_obs) # 初始化观察数据为相机观察数据
        obs_data.update(robot_obs) # 更新观察数据为机器人观察数据
        obs_data['timestamp'] = obs_align_timestamps # 设置观察数据的时间戳
        return obs_data # 返回观察数据
    
    # 执行给定的动作序列
    # actions，动作序列；timestamps，动作对应的时间戳；stages，动作的阶段信息（可选）。
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None):
        assert self.is_ready                            # 确保环境已准备好
        if not isinstance(actions, np.ndarray):         # 如果动作不是numpy数组
            actions = np.array(actions)                 # 将动作转换为numpy数组
        if not isinstance(timestamps, np.ndarray):      # 如果时间戳不是numpy数组
            timestamps = np.array(timestamps)           # 将时间戳转换为numpy数组
        if stages is None:                              # 如果阶段为空
            stages = np.zeros_like(timestamps, dtype=np.int64) # 初始化阶段为与时间戳相同形状的零数组
        elif not isinstance(stages, np.ndarray):        # 如果阶段不是numpy数组
            stages = np.array(stages, dtype=np.int64)   # 将阶段转换为numpy数组

        # convert action to pose
        receive_time = time.time()          # 获取当前时间
        is_new = timestamps > receive_time  # 找到所有新时间戳
        new_actions = actions[is_new]       # 获取新动作
        new_timestamps = timestamps[is_new] # 获取新时间戳
        new_stages = stages[is_new]         # 获取新阶段

        # schedule waypoints
        for i in range(len(new_actions)): # 遍历新动作
            self.robot.schedule_waypoint( # 为每个动作调度路径点
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )
        
        # record actions
        if self.action_accumulator is not None: # 如果动作累加器不为空
            self.action_accumulator.put( # 将新动作和新时间戳放入动作累加器
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None: # 如果阶段累加器不为空
            self.stage_accumulator.put( # 将新阶段和新时间戳放入阶段累加器
                new_stages,
                new_timestamps
            )
    
    # 获取当前机器人的状态
    def get_robot_state(self):
        return self.robot.get_state() # 获取机器人状态

    # recording API
    # 开始一个新的记录集，初始化观察和动作累加器，并开始记录视频
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None: # 如果开始时间为空
            start_time = time.time() # 获取当前时间作为开始时间
        self.start_time = start_time # 设置开始时间

        assert self.is_ready # 确保环境已准备好

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes # 获取当前集ID
        this_video_dir = self.video_dir.joinpath(str(episode_id)) # 创建视频目录
        this_video_dir.mkdir(parents=True, exist_ok=True) # 确保视频目录存在
        n_cameras = self.realsense.n_cameras # 获取相机数量
        video_paths = list() # 初始化视频路径列表
        for i in range(n_cameras): # 遍历相机数量
            video_paths.append( # 添加每个相机的视频路径
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on realsense
        self.realsense.restart_put(start_time=start_time) # 重新启动Realsense
        self.realsense.start_recording(video_path=video_paths, start_time=start_time) # 开始记录视频

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator( # 创建观察累加器
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator( # 创建动作累加器
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator( # 创建阶段累加器
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!') # 打印开始消息

    # 结束当前的记录集，停止记录视频，并将记录的数据保存到重放缓冲区
    def end_episode(self):
        "Stop recording"
        assert self.is_ready # 确保环境已准备好
        
        # stop video recorder
        self.realsense.stop_recording() # 停止记录视频

        if self.obs_accumulator is not None: # 如果观察累加器不为空
            # recording
            assert self.action_accumulator is not None # 确保动作累加器不为空
            assert self.stage_accumulator is not None # 确保阶段累加器不为空

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data # 获取观察数据
            obs_timestamps = self.obs_accumulator.timestamps # 获取观察时间戳

            actions = self.action_accumulator.actions # 获取动作数据
            action_timestamps = self.action_accumulator.timestamps # 获取动作时间戳
            stages = self.stage_accumulator.actions # 获取阶段数据
            n_steps = min(len(obs_timestamps), len(action_timestamps)) # 获取最小的步骤数
            if n_steps > 0: # 如果步骤数大于0
                episode = dict() # 初始化集字典
                episode['timestamp'] = obs_timestamps[:n_steps] # 设置时间戳
                episode['action'] = actions[:n_steps] # 设置动作
                episode['stage'] = stages[:n_steps] # 设置阶段
                for key, value in obs_data.items(): # 遍历观察数据
                    episode[key] = value[:n_steps] # 设置观察数据
                self.replay_buffer.add_episode(episode, compressors='disk') # 添加集到重放缓冲区
                episode_id = self.replay_buffer.n_episodes - 1 # 获取当前集ID
                print(f'Episode {episode_id} saved!') # 打印保存消息
            
            self.obs_accumulator = None # 清空观察累加器
            self.action_accumulator = None # 清空动作累加器
            self.stage_accumulator = None # 清空阶段累加器

    # 删除最近的记录集，包括删除视频文件和缓冲区中的数据
    def drop_episode(self):
        self.end_episode() # 结束当前记录
        self.replay_buffer.drop_episode() # 删除最近的记录
        episode_id = self.replay_buffer.n_episodes # 获取当前集ID
        this_video_dir = self.video_dir.joinpath(str(episode_id)) # 获取视频目录路径
        if this_video_dir.exists(): # 如果视频目录存在
            shutil.rmtree(str(this_video_dir)) # 删除视频目录
        print(f'Episode {episode_id} dropped!') # 打印删除消息
