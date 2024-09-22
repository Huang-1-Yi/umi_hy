from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers                           import SharedMemoryManager
from umi.real_world.realman_interpolation_controller    import RealmanInterpolationController
from umi.real_world.rtde_interpolation_controller       import RTDEInterpolationController
from umi.real_world.franka_interpolation_controller     import FrankaInterpolationController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.multi_uvc_camera                    import MultiUvcCamera, VideoRecorder
from diffusion_policy.common.timestamp_accumulator      import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util                                 import draw_predefined_mask
from umi.real_world.multi_camera_visualizer             import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer              import ReplayBuffer
from diffusion_policy.common.cv2_util                   import (get_image_transform, optimal_row_cols)
from umi.common.usb_util                                import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util                               import pose_to_pos_rot
from umi.common.interpolation_util                      import get_interp1d, PoseInterpolator

# 定义BimanualUmiEnv类及其构造函数，接收多个参数用于配置环境
class BimanualUmiEnv:
    def __init__(self, 
            # 所需参数 required params
            output_dir,
            robots_config, # list of dict[{robot_type: 'ur5', robot_ip: XXX, obs_latency: 0.0001, action_latency: 0.1, tcp_offset: 0.21}]
            grippers_config, # list of dict[{gripper_ip: XXX, gripper_port: 1000, obs_latency: 0.01, , action_latency: 0.1}]
            # env params
            frequency=20,
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            no_mirror=False,
            fisheye_converter=None,
            mirror_swap=False,
            # 此延迟补偿接收时间戳  this latency compensates receive_timestamp
            camera_obs_latency=0.125,# 全部以秒为单位 all in seconds 
            # 全部以步数表示（相对于频率） all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # 全部以步数表示（相对于频率）all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # 动作 action
            max_pos_speed=0.025, #0.25/0.1=2，太高了 0.5
            max_rot_speed=0.06, #0.6/0.1=6，太高了   1.5
            init_joints=True,
            # init_joints=False,
            # 视觉参数vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # 共享内存 shared memory
            shm_manager=None):
        # 将输出目录转换为pathlib.Path对象，并创建视频目录
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        # 设置重播缓冲区的路径，并创建一个重播缓冲区对象
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')
        if shm_manager is None:             # 如果未提供共享内存管理器，则创建一个新的共享内存管理器并启动
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        
        
        # reset_all_elgato_devices()          # 重置所有Elgato捕获卡，以解决固件错误
        
        time.sleep(0.5)
        v4l_paths = get_sorted_v4l_paths()  # 等待所有v4l摄像头重新在线，并获取排序后的v4l路径
        v4l_paths = [path for path in v4l_paths if 'Elgato' in path]# 去掉笔记本相机
        # 打印过滤后的结果
        # for idx, path in enumerate(v4l_paths):
        #     print("idx == ", idx, "path == ", path)


        if camera_reorder is not None:
            paths = [v4l_paths[i] for i in camera_reorder]
            v4l_paths = paths
        rw, rh, col, row = optimal_row_cols(# 根据摄像头数量和最大分辨率计算可视化所需的行数和列数
            n_cameras=len(v4l_paths),
            in_wh_ratio=4/3,
            max_resolution=multi_cam_vis_resolution
        )
        # 遍历每个摄像头路径，并根据摄像头类型（例如’Cam_Link_4K’或其他）设置分辨率、帧率、缓冲区大小和视频录制器
        # 特定处理方式（ HACK 是指“临时解决方案”或“变通方法”），用于为每个摄像头分别设置视频录制参数 HACK: Separate video setting for each camera
        # Elagto Cam Link 4k 摄像头以 4K 分辨率（3840x2160）录制视频，帧率为 30fps（每秒30帧）      Elagto Cam Link 4k records at 4k 30fps
        # 其他捕获卡以 720p 分辨率（通常是1280x720）录制视频，帧率为 60fps  Other capture card records at 720p 60fps
        resolution      = list()    # 初始化一个空列表，用于存储每个摄像头的分辨率
        capture_fps     = list()    # 初始化一个空列表，用于存储每个摄像头的捕获帧率。
        cap_buffer_size = list()    # 初始化一个空列表，用于存储每个摄像头的缓冲区大小。
        video_recorder  = list()    # 初始化一个空列表，用于存储每个摄像头的视频记录器。
        transform       = list()    # 初始化一个空列表，用于存储每个摄像头的转换函数。
        vis_transform   = list()    # 初始化一个空列表，用于存储每个摄像头的可视化转换函数。
        
        # 方案1：捕获卡
        for idx, path in enumerate(v4l_paths):
            print("过滤后, idx == ",idx, "path == ",path)
        for path in v4l_paths:
            if 'Elgato' in path:
                print("5.1 {} init".format(path))  
                res = (1920, 1080)
                fps = 60
                buf = 3 # 缓冲区
                bit_rate = 3000*1000
                def tf(data, input_res=res):
                    img = data['color']
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution,
                        bgr_to_rgb=True)
                    img = np.ascontiguousarray(f(img))  # 应用转换函数并确保图像数据在内存中是连续的。
                    # 用预定义的掩码函数更新图像
                    # img = draw_predefined_mask(img, color=(0,0,0), mirror=no_mirror, gripper=True, finger=False, use_aa=True)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)
            # elif 'Cam_Link_4K' in path:
            #     res = (3840, 2160)
            #     fps = 30
            #     buf = 3
            #     bit_rate = 6000*1000
            #     def tf4k(data, input_res=res):
            #         img = data['color']
            #         f = get_image_transform(
            #             input_res=input_res,
            #             output_res=obs_image_resolution, 
            #             # obs output rgb
            #             bgr_to_rgb=True)
            #         img = f(img)
            #         if obs_float32:
            #             img = img.astype(np.float32) / 255
            #         data['color'] = img
            #         return data
            #     transform.append(tf4k)
            else:
                print("5.1 {} init".format(path)) 
                res = (1920, 1080)
                fps = 60
                buf = 1
                bit_rate = 3000*1000
                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
                    is_mirror = (mirror_mask[...,0] == 0)
                
                def tf(data, input_res=res):
                    img = data['color']
                    if fisheye_converter is None:
                        f = get_image_transform(
                            input_res=input_res,
                            output_res=obs_image_resolution, 
                            # obs output rgb
                            bgr_to_rgb=True)
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:,::-1,:][is_mirror]
                        img = draw_predefined_mask(img, color=(0,0,0), 
                            mirror=no_mirror, gripper=True, finger=False, use_aa=True)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[...,::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)

            # 定义普通数据和可视化数据的转换函数
            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))
            # 定义可视化转换函数，并添加到可视化转换列表中
            def vis_tf(data, input_res=res):
                img = data['color']         # 从数据中获取图像
                f = get_image_transform(    # 获取图像转换函数
                    input_res=input_res,
                    output_res=(rw,rh),
                    bgr_to_rgb=False
                )
                img = f(img)
                data['color'] = img         # 更新数据中的图像
                return data                 # 返回更新后的数据
            vis_transform.append(vis_tf)

        # 如果启用了多摄像头可视化，则创建一个多摄像头可视化对象
        camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            video_recorder=video_recorder,
            verbose=False
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        # 计算立方体的对角线长度，并定义初始关节角度（如果需要）
        cube_diag = np.linalg.norm([1,1,1])
        # j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
        j_init = np.array([0, 0, -90, 0, 0, 0]) / 180 * np.pi
        if not init_joints:
            j_init = None
        # 断言机器人的配置数量和夹具的配置数量相等
        assert len(robots_config) == len(grippers_config)   # 如果条件为假，程序将抛出一个AssertionError异常
        # 初始化一个列表，用于存储机器人控制器对象
        # print("bimanual_List[RealmanInterpolationController] = list()")  
        robots: List[RealmanInterpolationController] = list()
        grippers: List[WSGController] = list()
        for rc in robots_config:                            # 遍历robots_config列表，其中每个元素是一个字典，包含机器人的配置信息
            if rc['robot_type'].startswith('ur'):          # 如果机器人类型以'ur5'开头（包括'ur5e'），则执行以下代码
                assert rc['robot_type'] in ['ur5', 'ur5e', 'ur3']
                this_robot = RTDEInterpolationController(   # 创建一个RTDEInterpolationController对象，传入共享内存管理器、机器人IP、频率、前瞻时间、增益、最大速度、TCP偏移、负载信息等参数
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    frequency=500 if rc['robot_type'] == 'ur5e' else 125,
                    lookahead_time=0.1,
                    gain=300,
                    max_pos_speed=max_pos_speed*cube_diag,
                    max_rot_speed=max_rot_speed*cube_diag,
                    launch_timeout=3,
                    tcp_offset_pose=[0, 0, rc['tcp_offset'], 0, 0, 0],
                    payload_mass=None,
                    payload_cog=None,
                    joints_init=j_init,
                    joints_init_speed=1.05,
                    soft_real_time=False,
                    verbose=False,
                    receive_keys=None,
                    receive_latency=rc['robot_obs_latency']
                )
            elif rc['robot_type'].startswith('franka'):     # 创建一个FrankaInterpolationController对象，传入共享内存管理器、机器人IP、频率、Kx_scale、Kxd_scale等参数
                this_robot = FrankaInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    frequency=200,
                    Kx_scale=1.0,
                    Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
                    verbose=False,
                    receive_latency=rc['robot_obs_latency']
                )
            elif rc['robot_type'].startswith('realman'):     # 创建一个FrankaInterpolationController对象，传入共享内存管理器、机器人IP、频率、Kx_scale、Kxd_scale等参数
                # print("this_robot == RealmanInterpolationController")  
                this_robot = RealmanInterpolationController(
                    shm_manager = shm_manager,
                    frequency = 100,
                    robot_ip = rc['robot_ip'],
                    # joints_init = rc['j_init'],
                    joints_init_speed = 0.3,# rc['j_init_speed']
                    soft_real_time=False,
                    verbose = False,
                    receive_keys=None,
                    receive_latency = rc['robot_obs_latency']
                )# 初始化角度、   等
            else:
                raise NotImplementedError() # 抛出一个NotImplementedError异常，表示不支持这种类型的机器人
            robots.append(this_robot)       # 将创建的机器人控制器对象添加到robots列表中
            # print("this_robot ==", this_robot)          
            
        for gc in grippers_config:          # 遍历grippers_config列表，其中每个元素是一个字典，包含夹具的配置信息
            this_gripper = WSGController(   # 创建一个WSGController对象，传入共享内存管理器、夹具IP、端口、接收延迟、是否使用米制单位等参数。
                shm_manager=shm_manager,
                hostname=gc['gripper_ip'],
                port=gc['gripper_port'],
                receive_latency=gc['gripper_obs_latency'],
                use_meters=True
            )
            # 加断言，判定爪子类型
            grippers.append(this_gripper)   # 将创建的夹具控制器对象添加到grippers列表中

        self.camera = camera
        print("机器人创建成功,robots==",robots)   
        self.robots = robots
        self.robots_config = robots_config
        self.grippers = grippers
        self.grippers_config = grippers_config
        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        # timing
        self.camera_obs_latency = camera_obs_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.start_time = None
        self.last_time_step = 0 # 可能无效
    
    # ======== start-stop API =============
    @property
    # 所有相关的设备（相机、机器人和夹爪）是否都准备好开始操作
    def is_ready(self):
        ready_flag = self.camera.is_ready
        # print('ready_flag==', ready_flag)
        for robot in self.robots:
            # ready_flag = ready_flag and robot.is_ready
            ready_flag = ready_flag and ready_flag
            # print('robot==', robot, 'ready_flag==', ready_flag)
        for gripper in self.grippers:
            ready_flag = ready_flag and gripper.is_ready
            # print('gripper==', gripper)
        return ready_flag
    
    # 开始所有相关的设备（相机、机器人和夹爪）
    def start(self, wait=True):
        self.camera.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        for gripper in self.grippers:
            gripper.start(wait=False)

        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()
    
    # 停止所有相关的设备
    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        for robot in self.robots:
            robot.stop(wait=False)
        for gripper in self.grippers:
            gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()
    
    # 等待所有相关的设备启动完成
    def start_wait(self):
        self.camera.start_wait()
        for robot in self.robots:
            robot.start_wait()
        for gripper in self.grippers:
            gripper.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    # 等待所有相关的设备停止完成。
    def stop_wait(self):
        for robot in self.robots:
            robot.stop_wait()
        for gripper in self.grippers:
            gripper.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        时间戳对齐策略 
        我们假设用于观测的摄像头始终是[0, k-1], 其中k是机器人的数量
        所有其他摄像头,找到与最接近时间戳对应的帧，所有低维观测值,根据'当前'时间进行插值
        """
        # assert self.is_ready                  # "observation dict"
        k = math.ceil(                          # 获取数据 get data60 Hz, camera_calibrated_timestamp
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency)) + 2        # here 2 is adjustable, typically 1 should be enough
        # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.frequency)
        self.last_camera_data = self.camera.get(k=k, out=self.last_camera_data)
        last_robots_data = list()               # 两者都拥有超过 n_obs_steps 的数据
        last_grippers_data = list()
        for robot in self.robots:               # 125/500 hz, robot_receive_timestamp
            last_robots_data.append(robot.get_all_state())
        for gripper in self.grippers:           # 30 hz, gripper_receive_timestamp # 暂时备注掉！！！
            last_grippers_data.append(gripper.get_all_state())

        num_obs_cameras = len(self.robots)
        align_camera_idx = None                 # 选择对齐相机索引 select align_camera_idx
        running_best_error = np.inf
   
        for camera_idx in range(num_obs_cameras):
            this_error = 0
            this_timestamp = self.last_camera_data[camera_idx]['timestamp'][-1]
            for other_camera_idx in range(num_obs_cameras):
                if other_camera_idx == camera_idx:
                    continue
                other_timestep_idx = -1
                while True:
                    if self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx] < this_timestamp:
                        this_error += this_timestamp - self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx]
                        break
                    other_timestep_idx -= 1
            if align_camera_idx is None or this_error < running_best_error:
                running_best_error = this_error
                align_camera_idx = camera_idx
        last_timestamp = self.last_camera_data[align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # 对齐相机观测时间戳 align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                # if np.abs(this_timestamps - t)[nn_idx] > 1.0 / 120 and camera_idx != 3:
                #     print('ERROR!!!  ', camera_idx, len(this_timestamps), nn_idx, (this_timestamps - t)[nn_idx-1: nn_idx+2])
                this_idxs.append(nn_idx)
            # remap key
            camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]
        obs_data = dict(camera_obs)                         # 返回的观测数据（目前仅包括相机数据） obs_data to return (it only includes camera data at this stage)
        obs_data['timestamp'] = camera_obs_timestamps       # 包括相机时间步 include camera timesteps

        # 对齐机器人观测数据 align robot obs
        robot_obs_timestamps = last_timestamp - (np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        for robot_idx, last_robot_data in enumerate(last_robots_data):
            # print('last_robot_data[robot_timestamp]',last_robot_data['robot_timestamp'])
            # print('last_robot_data[ActualTCPPose]',last_robot_data['ActualTCPPose'])
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data['robot_timestamp'], 
                x=last_robot_data['ActualTCPPose'])
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            robot_obs = {
                f'robot{robot_idx}_eef_pos': robot_pose[...,:3],
                f'robot{robot_idx}_eef_rot_axis_angle': robot_pose[...,3:]}
            obs_data.update(robot_obs)                      # 更新观测数据 update obs_data

        # 处理机器人夹具观测数据的时间戳对齐和插值 align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        for robot_idx, last_gripper_data in enumerate(last_grippers_data):
            # align gripper obs
            gripper_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_position'][...,None]
            )
            gripper_obs = {
                f'robot{robot_idx}_gripper_width': gripper_interpolator(gripper_obs_timestamps)
            }
            obs_data.update(gripper_obs)# update obs_data
            # print(f"Gripper data for robot{robot_idx} added:", gripper_obs)  # 添加调试信息


        # 将机器人末端执行器的位置、关节位置、关节速度和夹具宽度等观测数据累积到一个ObsAccumulator对象中 accumulate obs
        if self.obs_accumulator is not None:
            for robot_idx, last_robot_data in enumerate(last_robots_data):
                self.obs_accumulator.put(               # self.obs_accumulator存在，则将其作为输入，将last_robot_data中的末端执行器位置、关节位置和关节速度等数据累积到ObsAccumulator对象中
                    data={
                        f'robot{robot_idx}_eef_pose': last_robot_data['ActualTCPPose'],
                        f'robot{robot_idx}_joint_pos': last_robot_data['ActualQ'],
                        f'robot{robot_idx}_joint_vel': last_robot_data['ActualQd'],
                    },
                    timestamps=last_robot_data['robot_timestamp']
                )
            for robot_idx, last_gripper_data in enumerate(last_grippers_data):
                self.obs_accumulator.put(   # self.obs_accumulator存在，则将其作为输入，将last_gripper_data中的夹具宽度数据累积到ObsAccumulator对象中
                    data={
                        f'robot{robot_idx}_gripper_width': last_gripper_data['gripper_position'][...,None]
                    },
                    timestamps=last_gripper_data['gripper_timestamp']
                )
        # print("Final obs_data:", obs_data)  # 打印最终的 obs_data
        return obs_data
    





    # 大修改
    # 执行一系列动作 actions（动作数组）、timestamps（时间戳数组）和 compensate_latency（是否补偿延迟的布尔值）
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready                        # 如果is_ready不是True，则引发一个 AssertionError。这可能意味着某些设备或系统资源没有准备好执行动作
        if not isinstance(actions, np.ndarray):     # 检查 actions 和 timestamps 是否为 numpy 数组。如果不是，则将它们转换为 numpy 数组
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # 检查每个时间戳是否晚于当前时间（receive_time），如果是，则认为这是一个新的动作。提取新的动作和时间戳。convert action to pose
        receive_time = time.time()          # 获取当前时间
        is_new = timestamps > receive_time  # 找到所有新时间戳
        new_actions = actions[is_new]       # 获取新动作
        new_timestamps = timestamps[is_new] # 获取新时间戳
        print("timestamps:{}".format(timestamps))
        print("new_actions:{}".format(new_actions))

        # # 断言每个机器人动作的数量是否正确（应该是7个），并且总数必须是机器人数量的整数倍
        # print("new_actions.shape[1]:{},len(self.robots):{}".format(new_actions.shape[1],len(self.robots)))           # 打印max_speed
        # print("new_actions.shape[0]:{}".format(new_actions.shape[0]))           # 打印max_speed
        # assert new_actions.shape[1] // len(self.robots) == 7
        # assert new_actions.shape[1] % len(self.robots) == 0


        # 对于每个新的动作，为每个机器人（及其夹爪）调度一个路点（waypoint），考虑了延迟（如果启用补偿延迟）。 schedule waypoints
        for i in range(len(new_actions)):       # 遍历新动作
            for robot_idx, (robot, gripper, rc, gc) in enumerate(zip(self.robots, self.grippers, self.robots_config, self.grippers_config)):
                # 表示机器人动作延迟的属性和抓手延迟
                # compensate_latency 为 False 时：r_latency 被设置为 0.0，表示不补偿延迟
                r_latency = rc['robot_action_latency'] if compensate_latency else 0.0
                r_actions = new_actions[i, 7 * robot_idx + 0: 7 * robot_idx + 6]
                robot.schedule_waypoint(        # 为每个动作调度路径点
                    pose=r_actions[i],
                    target_time=new_timestamps[i] - r_latency
                )
                g_latency = gc['gripper_action_latency'] if compensate_latency else 0.0
                print("len(new_actions[i]:{}".format(len(new_actions[i])))
                if (len(new_actions[i])% 7 ==0) :
                    g_actions = new_actions[i, 7 * robot_idx + 6]
                    gripper.schedule_waypoint(
                        pos=g_actions,
                        target_time=new_timestamps[i] - g_latency
                    )
        # 如果存在 action_accumulator，则将新的动作和时间戳记录下来 record actions 放入动作累加器
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    # 包含所有机器人状态的列表。每个状态可能是一个包含机器人当前位置、姿态、速度等信息的字典。
    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]
    # # 包含所有夹爪状态的列表。每个状态可能是一个包含夹爪当前位置、姿态、开合状态等信息的字典。
    def get_gripper_state(self):
        return [gripper.get_state() for gripper in self.grippers]

    # 开始一个新的记录会话（episode）recording API
    # 它创建一个唯一的文件夹来存储该会话的视频和观测数据，并为相机和动作创建两个累积器（accumulators）。
    # 累积器用于存储观测数据和动作数据，以便在会话结束时可以重新播放和分析。
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()                    # 会话的开始时间，如果未提供，则使用当前时间。
        self.start_time = start_time

        # assert self.is_ready

        # prepare recording stuff 
        episode_id = self.replay_buffer.n_episodes      # 一个对象replay_buffer，用于管理会话的视频和观测数据
        this_video_dir = self.video_dir.joinpath(str(episode_id))   # video_dir 视频文件的存储目录
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras               # camera 是一个相机对象，用于录制视频
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # 两个累积器对象，用于存储观测数据和动作数据 create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency # 间隔时间=1/每秒记录的次数
        )
        print(f'Episode {episode_id} started!')
    
    # 用于结束一个记录会话（episode）的
    # 它停止视频记录，并处理累积的观测数据和动作数据，然后将它们添加到重放缓冲区（replay buffer）中
    def end_episode(self):
        "Stop recording"
        # assert self.is_ready
        self.camera.stop_recording()# 停止视频记录 stop video recorder
        # TODO
        if self.obs_accumulator is not None:# 检查 `obs_accumulator` 和 `action_accumulator` 是否不为 `None`。
            assert self.action_accumulator is not None  # 如果它们都存在，说明有累积的观测数据和动作数据需要处理 recording

            # 由于唯一积累观测和动作的方法是通过调用
            # 这两个操作将在同一个线程中执行，不需要担心新数据会在这里到来
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            # - 计算会话的结束时间，这通常是观测数据和动作数据中最后一条记录的时间戳。
            # - 从动作累积器中提取动作数据和时间戳，并确定有效的步骤数量。如果动作时间戳在结束时间之前，则将其视为有效步骤。
            # - 创建一个包含时间戳和动作的 `episode` 字典。
            # - 对于每个机器人，它使用 `PoseInterpolator` 和 `get_interp1d` 方法来插值机器人末端执行器的姿态（位置和旋转）、关节位置和关节速度以及夹爪宽度，并将这些插值结果添加到 `episode` 字典中。
            # - 将处理好的 `episode` 添加到重放缓冲区中，并压缩数据。
            # - 打印保存的会话ID。
            # - 最后，清除累积器，以便开始新的会话。
            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                for robot_idx in range(len(self.robots)):
                    robot_pose_interpolator = PoseInterpolator(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_eef_pose']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_eef_pose'])
                    )
                    robot_pose = robot_pose_interpolator(timestamps)
                    episode[f'robot{robot_idx}_eef_pos'] = robot_pose[:,:3]
                    episode[f'robot{robot_idx}_eef_rot_axis_angle'] = robot_pose[:,3:]
                    joint_pos_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_pos']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_pos'])
                    )
                    joint_vel_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_vel']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_vel'])
                    )
                    episode[f'robot{robot_idx}_joint_pos'] = joint_pos_interpolator(timestamps)
                    episode[f'robot{robot_idx}_joint_vel'] = joint_vel_interpolator(timestamps)

                    gripper_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_gripper_width']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_gripper_width'])
                    )
                    episode[f'robot{robot_idx}_gripper_width'] = gripper_interpolator(timestamps)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    # 结束当前的会话，然后从重放缓冲区中删除该会话。
    # 获取当前会话的ID，并使用它来定位视频文件的存储目录。
    # 如果该目录存在，则删除整个目录（包括所有视频文件）。
    # 打印已删除的会话ID。
    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
