from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math
import cv2  # 单爪子特有
from multiprocessing.managers                           import SharedMemoryManager
from umi.real_world.realman_interpolation_controller    import RealmanInterpolationController
from umi.real_world.rtde_interpolation_controller       import RTDEInterpolationController
from umi.real_world.franka_interpolation_controller     import FrankaInterpolationController
from umi.real_world.wsg_controller                      import WSGController
from umi.real_world.multi_uvc_camera                    import MultiUvcCamera, VideoRecorder
from diffusion_policy.common.timestamp_accumulator      import (TimestampActionAccumulator, ObsAccumulator)
from umi.common.cv_util                                 import (draw_predefined_mask, get_mirror_crop_slices)
from umi.real_world.multi_camera_visualizer             import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer              import ReplayBuffer
from diffusion_policy.common.cv2_util                   import (get_image_transform, optimal_row_cols)
from umi.common.usb_util                                import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util                               import pose_to_pos_rot
from umi.common.interpolation_util                      import get_interp1d, PoseInterpolator

# 定义UmiEnv类及其构造函数，接收多个参数用于配置环境
class UmiEnv:
    def __init__(self, 
            # 所需参数 required params
            output_dir,
            # robots_config, # list of dict[{robot_type: 'ur5', robot_ip: XXX, obs_latency: 0.0001, action_latency: 0.1, tcp_offset: 0.21}]
            # grippers_config, # list of dict[{gripper_ip: XXX, gripper_port: 1000, obs_latency: 0.01, , action_latency: 0.1}]
            # 手动定义
            robot_ip,# ？
            # 抓手部分
            gripper_ip  = '192.168.1.20',# ？
            gripper_port=1000,          # ？
            # 环境参数 env params
            robot_type='realman',#      robot_type='ur5',# ？
            frequency = 20,
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            no_mirror=False,
            fisheye_converter=None,

            mirror_crop=False,# 
            mirror_swap=False,
            # timing
            align_camera_idx=0,# ？
            # 此延迟补偿接收时间戳 this latency compensates receive_timestamp
            # 全部以秒为单位all in seconds
            camera_obs_latency          =0.125,# 0.125  ---0.14
            robot_obs_latency           =0.0001,
            gripper_obs_latency         =0.01,
            robot_action_latency        =0.082,
            gripper_action_latency      =0.1,
            # all in steps (relative to frequency)
            camera_down_sample_steps    =1,
            robot_down_sample_steps     =1,
            gripper_down_sample_steps   =1,
            # all in steps (relative to frequency)
            camera_obs_horizon          =2,
            robot_obs_horizon           =2,
            # 抓手部分
            gripper_obs_horizon         =2,
            # action
            max_pos_speed               =0.01,#0.25/0.1=2，太高了 umi用的0.5  0.1
            max_rot_speed               =0.005,#0.6/0.1=6，太高了   umi用的1.5  0.5
            # robot
            tcp_offset                  =0.235, # ur用
            init_joints                 =False,
            # 视觉参数 vis params
            enable_multi_cam_vis        =True,
            multi_cam_vis_resolution    =(960, 960),
            # 共享内存 shared memory
            shm_manager                 =None
            ):
        print("robot_ip=={} init".format(robot_ip)) 
        print("robot_type=={} init".format(robot_type)) 
        print("gripper_ip=={} init".format(gripper_ip)) 
        print("gripper_port=={} init".format(gripper_port)) 
        
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

        reset_all_elgato_devices()          # 重置所有Elgato捕获卡，以解决固件错误
        time.sleep(0.5)
        v4l_paths = get_sorted_v4l_paths()  # 等待所有v4l摄像头重新在线，并获取排序后的v4l路径
        v4l_paths = [path for path in v4l_paths if 'Elgato' in path]# 去掉笔记本相机
        # 打印过滤后的结果
        # for idx, path in enumerate(v4l_paths):
        #     print("idx == ", idx, "path == ", path)

        # if camera_reorder is not None:
        #     paths = [v4l_paths[i] for i in camera_reorder]
        #     v4l_paths = paths
            
        rw, rh, col, row = optimal_row_cols(# 根据摄像头数量和最大分辨率计算可视化所需的行数和列数
            n_cameras=len(v4l_paths),
            in_wh_ratio=4/3,
            max_resolution=multi_cam_vis_resolution
        )

        # HACK: Separate video setting for each camera
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
        # for idx, path in enumerate(v4l_paths):
        #     print("过滤后, idx == ",idx, "path == ",path)
        for path in v4l_paths:
            if 'Elgato' in path:
                print("5.1 Elgato init")  
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
            elif 'Cam_Link_4K' in path:
                print("5.2.1Cam_Link_4K init")  
                res = (3840, 2160)  # 1920, 1080
                fps = 30            # 60
                buf = 3
                bit_rate = 6000*1000# 设置比特率为6000kbps
                def tf4k(data, input_res=res):
                    img = data['color']# 从数据中获取图像
                    f = get_image_transform(# 获取图像转换函数
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = f(img)# 应用转换函数
                    if obs_float32:# 如果 obs_float32 为真，将图像转换为 float32 类型并归一化。
                        img = img.astype(np.float32) / 255
                    data['color'] = img# 更新数据中的图像
                    return data
                transform.append(tf4k)# 将转换函数添加到转换列表中
            else:
                print("5.3.1 {} init".format(path)) 
                print("5.3.2 Cam_Link_4K not found!")  
                res = (1920, 1080)
                fps = 15
                buf = 1
                bit_rate = 3000*1000
                # stack_crop = (idx==0) and mirror_crop   ##？？？？？
                is_mirror = None
                if mirror_swap:# 创建一个全为1的掩码
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
                        img = fisheye_converter.forward(img)#  使用鱼眼转换函数更新图像。
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

        # 单机械臂，无需断言机器人的配置数量和夹具的配置数量相等
        if robot_type.startswith('realman'):
            print("6.robot == RealmanInterpolationController")  
            robot = RealmanInterpolationController(
                shm_manager = shm_manager,
                frequency = 500,
                robot_ip = robot_ip,
                # joints_init = rc['j_init'],
                joints_init_speed = 0.3,# rc['j_init_speed']
                soft_real_time=False,
                verbose = False,
                receive_keys=None,
                receive_latency = robot_obs_latency
            )
            # print("this_robot ==", this_robot)
        elif robot_type.startswith('franka'):
            robot = FrankaInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=200,
                Kx_scale=1.0,
                Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
                verbose=False,
                receive_latency=robot_obs_latency
            )
        elif robot_type.startswith('ur'):
            robot = RTDEInterpolationController(    # 创建一个RTDEInterpolationController对象，传入共享内存管理器、机器人IP、频率、前瞻时间、增益、最大速度、TCP偏移、负载信息等参数
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=500,                      # UR5 CB3 RTDE
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
                receive_latency=robot_obs_latency
                )
        else:
            raise NotImplementedError() # 抛出一个NotImplementedError异常，表示不支持这种类型的机器人
    
        # 抓手部分
        gripper = WSGController(
            shm_manager=shm_manager,
            hostname=gripper_ip,
            port=gripper_port,
            receive_latency=gripper_obs_latency,
            use_meters=True
        )

        self.camera = camera
        self.robot = robot
        print("机器人创建成功")   
        # print("robot==", robot)
        # 抓手部分
        self.gripper = gripper
        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed



        self.mirror_crop = mirror_crop      #????
        # timing
        self.align_camera_idx           = align_camera_idx  # 以下5行为单机特有
        self.robot_obs_latency          = robot_obs_latency
        self.gripper_obs_latency        = gripper_obs_latency
        self.robot_action_latency       = robot_action_latency
        self.gripper_action_latency     = gripper_action_latency
        # timing
        self.camera_obs_latency         = camera_obs_latency
        self.camera_down_sample_steps   = camera_down_sample_steps
        self.robot_down_sample_steps    = robot_down_sample_steps
        self.gripper_down_sample_steps  = gripper_down_sample_steps
        self.camera_obs_horizon         = camera_obs_horizon
        self.robot_obs_horizon          = robot_obs_horizon
        self.gripper_obs_horizon        = gripper_obs_horizon
        # recording
        self.output_dir                 = output_dir
        self.video_dir                  = video_dir
        self.replay_buffer              = replay_buffer
        # temp memory buffers
        self.last_camera_data           = None
        # recording buffers
        self.obs_accumulator            = None
        self.action_accumulator         = None
        self.start_time                 = None
        self.last_time_step             = 0         # 可能无效


    # ======== start-stop API =============
    @property
    # 所有相关的设备（相机、机器人和夹爪）是否都准备好开始操作
    def is_ready(self):
        # return self.camera.is_ready and self.robot.is_ready
        ready_flag_camera = self.camera.is_ready
        ready_flag_robot = ready_flag_camera and self.robot.is_ready
        ready_flag = ready_flag_robot and self.gripper.is_ready
        return ready_flag
        # 抓手部分
        # return self.camera.is_ready and self.robot.is_ready and self.gripper.is_ready
    
    # 开始所有相关的设备（相机、机器人和夹爪）   
    def start(self, wait=True):
        self.camera.start(wait=False)
        self.robot.start(wait=False)
        # 抓手部分
        self.gripper.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()
    
    # 停止所有相关的设备
    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        # 抓手部分
        self.gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()
    # 等待所有相关的设备启动完成
    def start_wait(self):
        self.camera.start_wait()
        # 抓手部分
        self.gripper.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    # 等待所有相关的设备停止完成。    
    def stop_wait(self):
        self.robot.stop_wait()
        # 抓手部分
        self.gripper.stop_wait()
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
        assert self.is_ready
        k = math.ceil(          # 获取数据 get data60 Hz, camera_calibrated_timestamp
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency)) + 2         # here 2 is adjustable, typically 1 should be enough
        # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.frequency)

        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        # 两者都拥有超过 n_obs_steps 的数据
        last_robot_data = self.robot.get_all_state()# 125/500 hz, robot_receive_timestamp
        # both have more than n_obs_steps data

        # 抓手部分
        last_gripper_data = self.gripper.get_all_state()# 30 hz, gripper_receive_timestamp
        # last_gripper_data = {
        # 'gripper_state': np.array([0]),
        # 'gripper_position': np.array([100.0]),
        # 'gripper_velocity': np.array([0.0]),
        # 'gripper_force': np.array([0.0]),
        # 'gripper_measure_timestamp': np.array([time.time()]),
        # 'gripper_receive_timestamp': np.array([time.time()]),
        # 'gripper_timestamp': np.array([time.time()])
        # }
        # 检查+++ last_gripper_data 是否为空或包含空数组
        # print('过去的爪子信息last_gripper_data == ',last_gripper_data)
        
        # if not last_gripper_data or 'gripper_timestamp' not in last_gripper_data or 'gripper_position' not in last_gripper_data:
        #     raise ValueError("Gripper data is missing required keys or is empty.")
        # if len(last_gripper_data['gripper_timestamp']) == 0:
        #     raise ValueError("Gripper timestamp array is empty.")
        # if len(last_gripper_data['gripper_position']) == 0:
        #     raise ValueError("Gripper position array is empty.")


        # 多相机时候，选择对齐相机索引 select align_camera_idx
        # num_obs_cameras = len(self.robots)
        # align_camera_idx = None
        # running_best_error = np.inf
   
        # for camera_idx in range(num_obs_cameras):
        #     this_error = 0
        #     this_timestamp = self.last_camera_data[camera_idx]['timestamp'][-1]
        #     for other_camera_idx in range(num_obs_cameras):
        #         if other_camera_idx == camera_idx:
        #             continue
        #         other_timestep_idx = -1
        #         while True:
        #             if self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx] < this_timestamp:
        #                 this_error += this_timestamp - self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx]
        #                 break
        #             other_timestep_idx -= 1
        #     if align_camera_idx is None or this_error < running_best_error:
        #         running_best_error = this_error
        #         align_camera_idx = camera_idx


        last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]
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
                this_idxs.append(nn_idx)
            # remap key
            camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]
            # if camera_idx == 0 and self.mirror_crop:
            #     camera_obs['camera0_rgb'] = value['color'][...,:3][this_idxs]
            #     camera_obs['camera0_rgb_mirror_crop'] = value['color'][...,3:][this_idxs]
            # else:
            #     camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]
                
        obs_data = dict(camera_obs)                         # 返回的观测数据（目前仅包括相机数据） obs_data to return (it only includes camera data at this stage)
        obs_data['timestamp'] = camera_obs_timestamps       # 包括相机时间步 include camera timesteps


        # 对齐机器人观测数据 align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        # print('last_robot_data[robot_timestamp]',last_robot_data['robot_timestamp'])
        # print('last_robot_data[ActualTCPPose]',last_robot_data['ActualTCPPose'])
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'], 
            x=last_robot_data['ActualTCPPose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }
        obs_data.update(robot_obs)                      # 更新观测数据 update obs_data

        # 抓手部分
        # 处理机器人夹具观测数据的时间戳对齐和插值 align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        # print("1,get_interp1d")
        # 检查++++++再次检查传递给 get_interp1d 的数据
        # print("gripper_timestamp == ", last_gripper_data['gripper_timestamp'])
        # print("gripper_position == ", last_gripper_data['gripper_position'])

        gripper_interpolator = get_interp1d(
            t=last_gripper_data['gripper_timestamp'],
            x=last_gripper_data['gripper_position'][...,None]
        )
        gripper_obs = {
            'robot0_gripper_width': gripper_interpolator(gripper_obs_timestamps)
        }
        # gripper_obs = {
        #     'robot0_gripper_width': 0.1
        # }
        # update obs_data
        obs_data.update(gripper_obs)

        # 将机器人末端执行器的位置、关节位置、关节速度和夹具宽度等观测数据累积到一个ObsAccumulator对象中 accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            # 抓手部分
            self.obs_accumulator.put(
                data={
                    'robot0_gripper_width': last_gripper_data['gripper_position'][...,None]
                },
                timestamps=last_gripper_data['gripper_timestamp']
            )

        # return obs
        # obs_data = dict(camera_obs)
        # obs_data.update(robot_obs)
        # 抓手部分
        # obs_data.update(gripper_obs)
        # obs_data['timestamp'] = camera_obs_timestamps

        return obs_data




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
        receive_time = time.time()                  # 获取当前时间
        is_new = timestamps > receive_time          # 找到所有新时间戳
        new_actions = actions[is_new]               # 获取新动作
        new_timestamps = timestamps[is_new]         # 获取新时间戳
        # print("timestamps:{}".format(timestamps))
        # print("new_actions:{}".format(new_actions))



        # 对于每个新的动作，为每个机器人（及其夹爪）调度一个路点（waypoint），考虑了延迟（如果启用补偿延迟）。 schedule waypoints
        for i in range(len(new_actions)):
            # 表示机器人动作延迟的属性和抓手延迟
            # compensate_latency 为 False 时：r_latency 被设置为 0.0，表示不补偿延迟
            r_latency = self.robot_action_latency if compensate_latency else 0.0
            # 抓手部分
            g_latency = self.gripper_action_latency if compensate_latency else 0.0
            r_actions = new_actions[i,:6]
            g_actions = new_actions[i,6:]
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )
            # 抓手部分
            self.gripper.schedule_waypoint(
                pos=g_actions,
                target_time=new_timestamps[i]-g_latency
            )

        # 如果存在 action_accumulator，则将新的动作和时间戳记录下来 record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    


    def exec_actions_human(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # 检查每个时间戳是否晚于当前时间（receive_time），如果是，则认为这是一个新的动作。提取新的动作和时间戳。convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]



        # schedule waypoints
        for i in range(len(new_actions)):
            # 表示机器人动作延迟的属性和抓手延迟
            # compensate_latency 为 False 时：r_latency 被设置为 0.0，表示不补偿延迟
            r_latency = self.robot_action_latency if compensate_latency else 0.0
            # 抓手部分
            g_latency = self.gripper_action_latency if compensate_latency else 0.0
            r_actions = new_actions[i,:6]
            g_actions = new_actions[i,6:]
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )
            # 抓手部分
            self.gripper.schedule_waypoint(
                pos=g_actions,
                target_time=new_timestamps[i]-g_latency
            )

        # 如果存在 action_accumulator，则将新的动作和时间戳记录下来 record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def exec_actions_policy(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # 检查每个时间戳是否晚于当前时间（receive_time），如果是，则认为这是一个新的动作。提取新的动作和时间戳。convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]



        # schedule waypoints
        for i in range(len(new_actions)):
            # 表示机器人动作延迟的属性和抓手延迟
            # compensate_latency 为 False 时：r_latency 被设置为 0.0，表示不补偿延迟
            r_latency = self.robot_action_latency if compensate_latency else 0.0
            # 抓手部分
            g_latency = self.gripper_action_latency if compensate_latency else 0.0
            r_actions = new_actions[i,:6]
            g_actions = new_actions[i,6:]
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )
            # 抓手部分
            self.gripper.schedule_waypoint(
                pos=g_actions,
                target_time=new_timestamps[i]-g_latency
            )

        # 如果存在 action_accumulator，则将新的动作和时间戳记录下来 record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    






    # 包含所有机器人状态的列表。每个状态可能是一个包含机器人当前位置、姿态、速度等信息的字典。
    def get_robot_state(self):
        return self.robot.get_state()
    
    # 这部分是本来就没有
    # 包含所有夹爪状态的列表。每个状态可能是一个包含夹爪当前位置、姿态、开合状态等信息的字典。
    def get_gripper_state(self):
        # return [gripper.get_state() for gripper in self.grippers]
        return self.gripper.get_state()

    # 开始一个新的记录会话（episode）recording API
    # 它创建一个唯一的文件夹来存储该会话的视频和观测数据，并为相机和动作创建两个累积器（accumulators）。
    # 累积器用于存储观测数据和动作数据，以便在会话结束时可以重新播放和分析。
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()                                # 会话的开始时间，如果未提供，则使用当前时间。
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff 
        episode_id = self.replay_buffer.n_episodes                  # 一个对象replay_buffer，用于管理会话的视频和观测数据
        this_video_dir = self.video_dir.joinpath(str(episode_id))   # video_dir 视频文件的存储目录
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras                           # camera 是一个相机对象，用于录制视频
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
        assert self.is_ready

        # 停止视频记录 stop video recorder
        self.camera.stop_recording()

        # TODO
        if self.obs_accumulator is not None:
            # 如果它们都存在，说明有累积的观测数据和动作数据需要处理 recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
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

            # - 计算会话的结束时间——观测数据和动作数据中最后一条记录的时间戳
            # - 从动作累积器中提取动作数据和时间戳，并确定有效的步骤数量。如果动作时间戳在结束时间之前，则将其视为有效步骤
            # - 创建一个包含时间戳和动作的 `episode` 字典
            # - 对于每个机器人，它使用 `PoseInterpolator` 和 `get_interp1d` 方法来插值机器人末端执行器的姿态（位置和旋转）、关节位置和关节速度以及夹爪宽度，并将这些插值结果添加到 `episode` 字典中
            # - 将处理好的 `episode` 添加到重放缓冲区中，并压缩数据
            # - 最后，打印保存的会话ID，清除累积器，以便开始新的会话。
            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)
                episode['robot0_eef_pos'] = robot_pose[:,:3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
                # print("2,get_interp1d")
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                # print("3,get_interp1d")
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)
                # 抓手用
                # print("4,get_interp1d")
                gripper_interpolator = get_interp1d(
                    t=np.array(self.obs_accumulator.timestamps['robot0_gripper_width']),
                    x=np.array(self.obs_accumulator.data['robot0_gripper_width'])
                )
                episode['robot0_gripper_width'] = gripper_interpolator(timestamps)

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
