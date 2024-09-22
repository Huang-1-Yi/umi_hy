"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

    ================================ Human in control 人为控制 ==============================
    Robot movement:
    使用您的 SpaceMouse 来移动机器人的 EEF(在 xy 平面上锁定)
    按 SpaceMouse 右键解锁 z 轴
    按 SpaceMouse 左键启用旋转轴
    录制控制:
    点击 opencv 窗口（确保它处于焦点状态）
    按 "C" 开始评估（将控制权交给策略）
    按 "Q" 退出程序。
    ================================ Policy in control策略控制 ==============================
    确保您可以快速按下机器人硬件紧急停止按钮！
    Recording control:
    按 "S" 停止评估并夺回控制权
"""

# %%
import os
import pathlib
import time
import datetime
import aiofiles
import asyncio

import av
import click
import cv2
import yaml
import dill
import hydra
import torch
import json
import numpy as np
import scipy.spatial.transform as st
from omegaconf                                  import OmegaConf
from multiprocessing.managers                   import SharedMemoryManager
from diffusion_policy.common.replay_buffer      import ReplayBuffer
from diffusion_policy.common.cv2_util           import get_image_transform
from diffusion_policy.common.pytorch_util       import dict_apply
from diffusion_policy.workspace.base_workspace  import BaseWorkspace
from umi.common.cv_util                         import parse_fisheye_intrinsics,FisheyeRectConverter
from umi.common.precise_sleep                   import precise_wait
# from umi.real_world.bimanual_umi_env            import BimanualUmiEnv
from umi.real_world.umi_env_franka              import UmiEnv
from umi.real_world.keystroke_counter           import KeystrokeCounter, Key, KeyCode
from umi.real_world.real_inference_util         import (get_real_obs_dict,
                                                        get_real_obs_resolution,
                                                        get_real_umi_obs_dict,
                                                        get_real_umi_action)
from umi.real_world.spacemouse_shared_memory    import Spacemouse
from umi.common.pose_util                       import pose_to_mat, mat_to_pose

# 导入操作系统接口、路径操作、多进程共享内存管理器、音视频处理、命令行参数解析、OpenCV图像处理、YAML和JSON数据格式处理、深度学习框架PyTorch、科学计算库NumPy和SciPy、机器人控制相关的自定义模块等
OmegaConf.register_new_resolver("eval", eval, replace=True)

# 定义了一个解决机器人末端执行器（EEF）与桌子碰撞的算法。它计算了机器人手指与桌子之间的最小距离，并在必要时调整EEF的高度以避免碰撞
def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))


loop = asyncio.get_event_loop()
async def save_pose_to_file(data, filename):
    async with aiofiles.open(filename, 'a') as file:
        await file.write(json.dumps(data.tolist()) + '\n')

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap):
    max_gripper_width = 0.115
    gripper_speed = 0.2
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("1.创建文件存储数据:", steps_per_inference)

    # 存储函数
    current_time = datetime.datetime.now()
    # 格式化时间：例如，年_月_日_时_分
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M')
    # 修改文件名
    file_name1 = 'exec_actions_human_joint{}.json'.format(formatted_time)
    file_name2 = 'exec_actions_policy_joint{}.json'.format(formatted_time)

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with    Spacemouse(shm_manager=shm_manager) as sm, \
                KeystrokeCounter() as key_counter, \
                UmiEnv( output_dir= output,                                         # 创建双臂机器人控制环境：输出目录、机器人配置、夹爪配置、控制频率、观测图像分辨率等
                        robot_ip            =   robots_config[0]['robot_ip'],
                        # 抓手部分
                        gripper_ip          =   grippers_config[0]['gripper_ip'],   # '192.168.1.20'
                        gripper_port        =   grippers_config[0]['gripper_port'], # 1000
                        frequency           =   frequency,
                        robot_type          =   robots_config[0]['robot_type'],     # 'realman'
                        obs_image_resolution=   obs_res,
                        obs_float32         =   True,
                        camera_reorder      =   [int(x) for x in camera_reorder],
                        init_joints         =   init_joints,
                        enable_multi_cam_vis=   True,
                        camera_obs_latency  =   0.15,           # 0.17              # 相机延迟，umi默认0.17，测量为0.126-0.156
                        camera_obs_horizon  =   cfg.task.shape_meta.obs.camera0_rgb.horizon, # obs
                        robot_obs_horizon   =   cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                        gripper_obs_horizon =   cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                        no_mirror           =   no_mirror,
                        fisheye_converter   =   fisheye_converter,
                        mirror_crop         =   False,          # ？
                        mirror_swap         =   mirror_swap,
                        max_pos_speed       =   0.1,            # 2                 # 动作速度 action
                        max_rot_speed       =   0.2,            # 6
                        shm_manager         =   shm_manager) as env:
            # BimanualUmiEnv(
            #     output_dir=output,
            #     robots_config=robots_config,
            #     grippers_config=grippers_config,
            #     frequency=frequency,
            #     obs_image_resolution=obs_res,
            #     obs_float32=True,
            #     camera_reorder=[int(x) for x in camera_reorder],
            #     init_joints=init_joints,
            #     enable_multi_cam_vis=True,
            #     # latency
            #     camera_obs_latency=0.17,
            #     # obs
            #     camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
            #     robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
            #     gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
            #     no_mirror=no_mirror,
            #     fisheye_converter=fisheye_converter,
            #     mirror_swap=mirror_swap,
            #     # action
            #     max_pos_speed=2.0,
            #     max_rot_speed=6.0,
            #     shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)                                        # 设置OpenCV的线程数为2，以优化图像处理
            print("4.等待相机Waiting for camera:{}".format([int(x) for x in camera_reorder]))                       # 打印一条消息，表示正在等待摄像头
            time.sleep(1.0)                                             # 让程序暂停1秒，等待摄像头就绪

            # 加载匹配数据集，指定匹配数据集时，创建重放缓冲区，并从视频中提取每一集的第一个帧
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break
                        episode_first_frame_map[episode_idx] = img
            print(f"5.加载初始框架给XX集 Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # 创建模型 # have to be done after fork to prevent duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)               # 使用Hydra工具从配置对象cfg中获取目标类（模型类）
            workspace = cls(cfg)                                    # 实例化获取的模型类，创建一个工作空间对象workspace
            workspace: BaseWorkspace                                # 类型标注，表示workspace是BaseWorkspace类型
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)# 调用工作空间对象的load_payload方法，加载必要的payload（可能包括模型参数等），可以根据需要排除或包含特定的键

            policy = workspace.model                                # 从工作空间对象中获取模型，并将其赋值给变量policy
            if cfg.training.use_ema:                                # 如果配置中指定使用指数移动平均（EMA）模型，则从工作空间对象中获取EMA模型
                policy = workspace.ema_model
            policy.num_inference_steps = 16                         # 设置策略模型的推断步骤数为16，这通常用于DDIM（深度卷积逆扩散模型）推断迭代 DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr         # 从配置中获取观察姿势表示
            action_pose_repr = cfg.task.pose_repr.action_pose_repr  # 从配置中获取动作姿势表示
            print('6.观察姿势表示的配置信息 obs_pose_rep', obs_pose_rep)              # 打印观察姿势表示的配置信息
            print('7.动作姿势表示的配置信息 action_pose_repr', action_pose_repr)      # 打印动作姿势表示的配置信息


            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()
            pose = np.concatenate([                                 # 构建机器人的姿势数据(末端执行器位置, 末端执行器旋转轴角度)
                obs[f'robot0_eef_pos'],
                obs[f'robot0_eef_rot_axis_angle']
            ], axis=-1)[-1]                                         # 将位置和旋转轴角度沿最后一个轴拼接，并取最后一个元素
            
            print("8.0.1 robot0_eef_pos == ", obs[f'robot0_eef_pos'])
            print("8.0.2 robot0_eef_rot_axis_angle == ", obs[f'robot0_eef_rot_axis_angle'])
            episode_start_pose.append(pose)                         # 将机器人的起始姿势添加到列表中
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                # assert action.shape[-1] == 10 * len(robots_config)
                assert action.shape[-1] == 10
                # 将动作转换为适合机器人执行的形式
                # print('9.action == ', action)
                action = get_real_umi_action(action, obs, action_pose_repr)
                # assert action.shape[-1] == 7 * len(robots_config)
                assert action.shape[-1] == 7
                del result

            print('9.策略已准备好 Ready!')
            # time.sleep(20)
            # 控制循环，允许用户通过SpaceMouse进行机器人控制
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                # 获取目标姿态
                robot_states = env.get_robot_state()                    # 从环境中获取机器人的所有状态
                target_pose = np.asarray(robot_states['TargetTCPPose']) # 机器人的目标姿态
                # target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])
                # target_pose = robot_states['TargetTCPPose'] # 获取目标姿势
                # print("state['TargetTCPPose'] == ", target_pose)
                gripper_states = env.get_gripper_state()                # 从环境中获取所有机器人的夹爪状态
                gripper_target_pos = np.asarray(gripper_states['gripper_position'])
                print("target_pose:{}, gripper_states:{}".format(target_pose,gripper_states))
                # gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])
                # gripper_target_pos =gripper_states['gripper_position']
                control_robot_idx_list = [0]                            # 定义一个控制机器人索引列表，初始只包含索引0的机器人
                t_start = time.monotonic()                              # 获取当前时间，用于后续计算时间间隔
                iter_idx = 0                                            # 初始化迭代索引，用于控制循环
                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt         # 计算当前控制周期的结束时间
                    t_sample = t_cycle_end - command_latency            # 计算采样时间，即从SpaceMouse命令开始到机器人执行命令之间的时间延迟
                    t_command_target = t_cycle_end + dt                 # 计算命令目标时间，即下一个控制周期开始的时间
                    # 从环境中获取当前观测
                    obs = env.get_obs()

                    # 可视化
                    episode_id = env.replay_buffer.n_episodes           # 获取当前剧集的编号
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]      # 获取最后一个匹配摄像头的图像
                    match_episode_id = episode_id                       # 将当前剧集编号赋值给匹配剧集编号
                    if match_episode is not None:                       # 检查是否指定了匹配剧集，如果是，则使用指定的值
                        match_episode_id = match_episode    
                    if match_episode_id in episode_first_frame_map:     # 检查是否已加载匹配剧集的第一个帧，如果是，则使用它
                        match_img = episode_first_frame_map[match_episode_id]
                        # 获取匹配图像和观测图像的尺寸
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        # 获取图像转换器，用于将匹配图像转换为观测图像的尺寸
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        # 将匹配图像转换为浮点数，并归一化到0-1范围内
                        match_img = tf(match_img).astype(np.float32) / 255
                        # 将观测图像和匹配图像混合
                        vis_img = (vis_img + match_img) / 2
                    



                    # obs_left_img = obs['camera0_rgb'][-1]
                    # obs_right_img = obs['camera0_rgb'][-1]
                    # vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])        # 使用OpenCV显示拼接后的图像
                    _ = cv2.pollKey()                               # 等待用户按键，以便于在显示窗口中退出
                    press_events = key_counter.get_press_events()   # 获取按键事件
                    # print("10.0.1 press_events == ", press_events)
                    # press_events = 'c'
                    start_policy = False                            # 初始化一个标志位，用于切换到策略控制
                    for key_stroke in press_events:                 # 遍历所有按键事件
                        # 如果用户按下了退出键（q），则结束当前剧集，退出程序
                        # 用户按下了切换到策略控制键（c），则切换到策略控制
                        # 如果用户按下了下一集键（e），则移动到下一个剧集
                        # 如果用户按下了上一集键（w），则移动到上一个剧集
                        # 如果用户按下了移动机器人键（m），则为机器人夹爪设置一个路径点，夹爪将在给定时间到达指定宽度，需要位置
                        if key_stroke == KeyCode(char='q'):
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)

                            pos = ep[f'robot0_eef_pos'][0]                              # 获取机器人的末端执行器位置和旋转
                            rot = ep[f'robot0_eef_rot_axis_angle'][0]
                            grip = ep[f'robot0_gripper_width'][0]                       # 获取机器人的夹爪宽度
                            pose = np.concatenate([pos, rot])                           # 将位置和旋转合并为一个数组，表示机器人的姿态
                            env.robot[0].servoL(pose, duration=duration)                # 根据给定的姿态和持续时间，控制机器人的末端执行器移动到目标位置
                            # 为机器人夹爪设置一个路径点，夹爪将在给定时间到达指定宽度，需要位置
                            env.grippers[0].schedule_waypoint(grip, target_time=time.time() + duration)
                            
                            # ？？？？？？
                            target_pose[0] = pose
                            gripper_target_pos[0] = grip
                            time.sleep(duration)                                        # 暂停程序执行，以等待机器人完成移动
                        # 如果用户按下了退格键，则弹出一个确认框，询问用户是否确定要删除当前剧集
                        # 如果用户按下了切换控制机器人键（a），则设置控制机器人索引列表为所有机器人
                        # 如果用户按下了选择机器人1键（1），则设置控制机器人索引列表只包含机器人1
                        # 如果用户按下了选择机器人2键（2），则设置控制机器人索引列表只包含机器人2
                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()                      # 删除当前剧集
                                key_counter.clear()                     # 清除按键计数器
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = range(target_pose.shape[0])
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]
                    # 检查是否需要切换到策略控制，如果是，则退出内部循环
                    if start_policy: 
                        break                                           # 切换到策略控制

                    precise_wait(t_sample)                              # 等待直到达到采样时间
                    # 5.遥操作
                    # 5.1 获取遥操作命令
                    sm_state = sm.get_motion_state_transformed()        # 获取SpaceMouse的运动状态
                    # print(sm_state)
                    print("sm_state:{}".format(sm_state))               # 打印运动状态
                    # dpos = sm_state[:3] * (0.5 / frequency)
                    # drot_xyz = sm_state[3:] * (1.5 / frequency)
                    dpos = sm_state[:3] * (env.max_pos_speed / frequency)       # 计算位置增量
                    drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)   # 计算旋转增量
                    # print("max_pos_speed:{},max_rot_speed:{}".format(env.max_pos_speed,env.max_rot_speed))           # 打印max_speed
                    
                    # 旋转轴和Z轴解锁
                    # 如果没有按下第一个按钮，平移模式
                    drot = st.Rotation.from_euler('xyz', drot_xyz)      # 将旋转速度转换为一个旋转对象
                    # 更新目标姿态，根据SpaceMouse的输入和当前控制机器人列表
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                    dpos = 0                                            # 重置平移速度变量
                    # 如果SpaceMouse的第一个按钮被按下，则设置夹爪的平移速度为负值，表示夹爪关闭
                    if sm.is_button_pressed(0):                     # close gripper
                        dpos = -gripper_speed / frequency
                    # 如果SpaceMouse的第二个按钮被按下，则设置夹爪的平移速度为正值，表示夹爪打开
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    # 更新夹爪的目标位置，确保夹爪的位置在0到最大夹爪宽度之间
                    gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)
                    
                    # # solve collision with table
                    # for robot_idx in control_robot_idx_list:
                    #     solve_table_collision(
                    #         ee_pose=target_pose[robot_idx],
                    #         gripper_width=gripper_target_pos[robot_idx],
                    #         height_threshold=robots_config[robot_idx]['height_threshold'])
                    

                    action = np.zeros(7,)
                    action[:6] = target_pose
                    action[6] = gripper_target_pos      #  + 0.04

                    # 5.2 裁剪目标姿势，，评估的特殊操作————hy
                    # target_pose[:2] = np.clip(target_pose[:2], [0.25, -0.45], [0.77, 0.40]) # 裁剪XY轴位置
                    # execute actions
                    # print("env.exec_actions用7维数组action:{}".format(action))           # 打印action
                    print("env.exec_actions用gripper:{}".format(action[6]))           # 打印gripper
                    loop.run_until_complete(save_pose_to_file(action, file_name1))
    
                    # 5.3 执行遥操作命令
                    env.exec_actions(                               # 执行动作
                        actions=[action], 
                        # timestamps=[t_command_target-time.monotonic()+time.time()])
                        timestamps=[t_command_target-time.monotonic()+time.time()],# 设置时间戳
                        compensate_latency=False)                       # dp没有compensate_latency
                    precise_wait(t_cycle_end)                           # 等待直到控制周期的结束时间
                    iter_idx += 1                                       # 增加迭代索引，用于控制内部循环
                
                # ========== policy control loop ==============
                try:
                    policy.reset()                                      # 重置策略模型，可能是在新剧集开始时
                    start_delay = 1.0                                   # 定义了一个延迟，用于等待开始
                    eval_t_start = time.time() + start_delay            # 计算剧集开始的实际时间
                    t_start = time.monotonic() + start_delay            # 计算内部循环开始的时间
                    env.start_episode(eval_t_start)                     # 启动一个新剧集
                    obs = env.get_obs()                                 # 获取当前环境中的观测
                    episode_start_pose = list()                         # 初始化一个列表，用于存储剧集开始的姿态
                    pose = np.concatenate([                             # 合并末端执行器的位置和旋转，并获取最后一个元素
                        obs[f'robot0_eef_pos'],
                        obs[f'robot0_eef_rot_axis_angle']
                    ], axis=-1)[-1]
                    episode_start_pose.append(pose)                     # 将姿态添加到列表中

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60                                # 定义了一个延迟，用于减少总延迟 # wait for 1/30 sec to get the closest frame actually, reduces overall latency
                    precise_wait(eval_t_start - frame_latency, time_func=time.time) # 等待直到剧集开始的时间
                    print("11.策略控制开始!!!Policy control loop Started!")
                    iter_idx = 0                                        # 初始化迭代索引
                    perv_target_pose = None                             # 初始化一个变量，用于存储前一个目标姿态
                    # 开始无限循环，处理每个控制周期
                    while True:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt       # 计算当前控制周期的结束时间calculate timing
                        obs = env.get_obs()                             # 获取当前环境中的观测 
                        obs_timestamps = obs['timestamp']               # 从观测中提取时间戳
                        # print(f'12.观测Obs延迟{time.time() - obs_timestamps[-1]}')
                        with torch.no_grad():                           # 创建无梯度的上下文，执行不涉及梯度更新的操作 
                            s = time.time()                             # 记录当前时间，用于后续计算推理延迟
                            # print(f"Adjusted shape for camera0_rgb: {obs_dict_np['camera0_rgb'].shape}")
                            obs_dict_np = get_real_umi_obs_dict(        # 将环境的观测转换为一个字典，包含了机器人末端执行器的位置和旋转
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            # print(obs_dict_np)
                            # 主动调整
                            # 确保图像形状为 (3, 224, 224)
                            # for key, value in obs_dict_np.items():
                            #     if 'camera' in key and '_rgb' in key:
                            #         value = cv2.resize(value, (224, 224))
                            #         value = value.transpose(2, 0, 1)  # 将 (H, W, C) 转换为 (C, H, W)
                            #         obs_dict_np[key] = value

                            obs_dict = dict_apply(obs_dict_np,          # 将观测字典中的所有值转换为PyTorch张量，并移动到GPU上
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)    # 使用策略模型来预测一个动作
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()# 从预测结果中提取第一个动作，并将其转换为CPU上的NumPy数组
                            action = get_real_umi_action(raw_action, obs, action_pose_repr) # 将动作转换为适合机器人执行的形式
                            # print('13.接口Inference延迟:', time.time() - s)             # 打印出推理的延迟
                        
                        this_target_poses = action                      # 将策略生成的动作转换为环境中的动作 convert policy action to env actions 
                        # 有无抓手
                        assert this_target_poses.shape[1] == 7                      # 检查动作数组的维度是否正确
                        # 处理每个机器人的目标姿态
                        for target_pose in this_target_poses:
                            solve_table_collision(  # 桌子碰撞————调用solve_table_collision函数，以确保机器人的末端执行器（EEF）和夹爪不会与桌子发生碰撞
                                ee_pose=target_pose[:6],
                                gripper_width=target_pose[6],        #  + 0.04
                                height_threshold=robots_config[0]['height_threshold']
                            )
                        # for target_pose in this_target_poses:
                        #     for robot_idx in range(len(robots_config)):
                        #         solve_table_collision(
                        #             ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                        #             gripper_width=target_pose[robot_idx * 7 + 6],
                        #             height_threshold=robots_config[robot_idx]['height_threshold']
                        #         )
                            
                        #     # solve collison between two robots
                        #     solve_sphere_collision(
                        #         ee_poses=target_pose.reshape([len(robots_config), -1]),
                        #         robots_config=robots_config
                        #     )

                        # 处理动作的定时问题，确保动作与观测的时间戳对齐 deal with timing the same，step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1] # 计算每个动作的时间戳
                        print("14.dt == ",dt)                                                  # 打印出控制周期的时间间隔
                        action_exec_latency = 0.01                                          # 定义执行动作的延迟
                        curr_time = time.time()                                             # 获取当前时间
                        is_new = action_timestamps > (curr_time + action_exec_latency)      # 检查是否到了新的时间点
                        if np.sum(is_new) == 0:                                             # 如果所有动作的时间戳都在当前时间之前，则
                            this_target_poses = this_target_poses[[-1]]                     # 只保留最后一个动作    # exceeded time budget, still do something
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))   # schedule on next available step
                            action_timestamp = eval_t_start + (next_step_idx) * dt          # 计算下一个可用时间点作为动作的时间戳
                            print('Over budget', action_timestamp - curr_time)              # 打印出超出预算的时间
                            action_timestamps = np.array([action_timestamp])                # 为动作设置新的时间戳
                        else:                                                               # 如果还有新的动作时间戳，则只保留那些时间戳在当前时间之后的动作
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]
                        # execute actions
                        # print("env.exec_actions_this_target_poses.shape[1]:{}".format(this_target_poses.shape[1]))           # 打印max_speed
                        loop.run_until_complete(save_pose_to_file(action, file_name2))

                        print("env.exec_actions用gripper:{}".format(this_target_poses[6]))           # 打印gripper
                        env.exec_actions(                                                   # 执行所有机器人动作，并将动作的时间戳设置为实际的时间点 execute actions
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        
                        # print(f"15.已提交 {len(this_target_poses)} 步动作.") # 打印出已提交的步骤数

                        # print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes       # 可视化，获取当前剧集的编号
                        # obs_left_img = obs['camera0_rgb'][-1]
                        # obs_right_img = obs['camera0_rgb'][-1]
                        vis_img = obs['camera0_rgb'][-1]                # 获取最后一个左侧摄像头的图像
                        text = 'Episode: {}, Time: {:.1f}'.format( episode_id, time.monotonic() - t_start )# 创建一个文本字符串，显示当前剧集编号和运行时间
                        cv2.putText(                                    # 使用OpenCV在图像上绘制文本
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])        # 使用OpenCV显示图像

                        _ = cv2.pollKey()                               # 等待用户按键，以便在显示窗口中退出
                        press_events = key_counter.get_press_events()   # 获取按键事件
                        stop_episode = False                            # 初始化一个标志位，用于控制是否停止剧集
                        
                        for key_stroke in press_events:                 # 遍历所有按键事件
                            if key_stroke == KeyCode(char='s'):         # 如果用户按下了开始策略控制键（s），则
                                print('停止剧集,切换人工控制 Stopped.')
                                stop_episode = True                     # 设置标志位以停止剧集,切换人工控制
                        t_since_start = time.time() - eval_t_start      # 计算剧集开始以来经过的时间
                        
                        if t_since_start > max_duration:                # 如果经过的时间超过了最大持续时间，则设置标志位以停止剧集
                            print("最大持续时间到了 Max Duration reached.")
                            stop_episode = True
                        if stop_episode:                                # 如果设置了停止剧集的标志位，则退出循环，结束策略控制循环
                            env.end_episode()
                            break

                        precise_wait(t_cycle_end - frame_latency)       # 等待直到控制周期的结束时间 wait for execution
                        iter_idx += steps_per_inference                 # 增加迭代索引，用于控制内部循环

                # 开始一个异常处理块，用于处理键盘中断
                except KeyboardInterrupt:
                    print("Interrupted!") # 打印出“中断！”
                    # stop robot.
                    env.end_episode() # 结束当前剧集，并可能包含停止机器人的代码
                # finally: # huangyi——程序在执行过程中被中断，某些必要的清理工作也会完成
                #     print("Cleaning up resources...")
                #     # 在这里放置任何需要清理的代码，例如关闭文件或释放资源
                #     # ... finally block code ...
                print("Stopped.") # 打印出“已停止”



# %%
if __name__ == '__main__':
    main()