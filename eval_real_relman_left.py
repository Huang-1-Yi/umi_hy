"""
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
import os
import pathlib
import time
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
from umi.real_world.umi_env_realman             import UmiEnv
from umi.real_world.keystroke_counter           import KeystrokeCounter, Key, KeyCode
from umi.real_world.real_inference_util         import (get_real_obs_dict,
                                                        get_real_obs_resolution,
                                                        get_real_umi_obs_dict,
                                                        get_real_umi_action)
from umi.real_world.spacemouse_shared_memory    import Spacemouse
from umi.common.pose_util                       import pose_to_mat, mat_to_pose
from umi.real_world.rm_py                       import log_setting
from umi.real_world.rm_py                       import robotic_arm

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

# 使用click库定义命令行接口，允许用户通过命令行来运行程序
@click.command() 
@click.option('--input', '-i', required=True, help='Path to checkpoint')                            # 必需的，提供检查点文件的路径
@click.option('--output', '-o', required=True, help='Directory to save recording')                  # 必需，指定一个目录来保存记录
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')        # 必需的，提供机器人配置文件的路径
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')       # 可以指定一个数据集来覆盖和调整初始条件
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')   # 指定要从匹配数据集中匹配的特定数据集
@click.option('--match_camera', '-mc', default=0, type=int)             # 默认值为0，用户可以指定要匹配的摄像头
@click.option('--camera_reorder', '-cr', default='0')                   # 默认值为’0’，用于指定摄像头的重新排序
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")         # 默认值为0，用户可以指定要可视化的RealSense摄像头
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.") # 默认为False，如果设置为True，程序将在开始时初始化机器人的关节配置
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")    # 默认值为6，用于指定每次推理的动作范围
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')     # 2000000秒，用于指定每个周期的最大持续时间
@click.option('--frequency', '-f', default=100, type=float, help="Control frequency in Hz.")                 # 默认值为10Hz，用于指定控制频率
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.") # 默认值为0.01秒，用于指定接收到SpaceMouse命令到机器人执行命令之间的延迟
@click.option('-nm', '--no_mirror', is_flag=True, default=False)        # 如果设置为True，则表示不需要镜像处理
@click.option('-sf', '--sim_fov', type=float, default=None)             # 默认为None，用于指定模拟的视场角
@click.option('-ci', '--camera_intrinsics', type=str, default=None)     # 默认为None，用于指定摄像机的内在参数
@click.option('--mirror_swap', is_flag=True, default=False)             # 如果设置为True，则表示需要进行镜像交换
def main(input, output,robot_config,                                    # 输入的检查点路径、输出目录路径、机器人配置文件的路径
        match_dataset, match_episode,                                   # 用于覆盖和调整初始条件的数据集、从匹配数据集中匹配的特定剧集、
        match_camera,camera_reorder, vis_camera_idx, init_joints,       # 要匹配的摄像头、 摄像头的重新排序、要可视化的RealSense摄像头的索引、是否在开始时初始化机器人的关节配置、
        steps_per_inference, max_duration, frequency, command_latency,  # 每次推理的动作范围、 每个周期的最大持续时间、控制频率、从接收到SpaceMouse命令到机器人执行命令之间的延迟、
        no_mirror, sim_fov, camera_intrinsics, mirror_swap):            # 是否不需要镜像处理、模拟的视场角、摄像机的内在参数、是否需要进行镜像交换
    max_gripper_width = 0.10                                            # 定义最大抓握宽度和抓握速度的变量，分别赋值为0.09m和0.2m/s
    gripper_speed = 0.2
    # 加载机器人配置文件 load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))                 # 使用yaml.safe_load函数将YAML文件内容解析为Python字典，并存储在robot_config_data变量中
    # 从配置数据中提取机器人和夹爪的配置信息，分别存储在robots_config和grippers_config变量中
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # 使用PyTorch的torch.load函数来加载检查点文件
    ckpt_path = input                                                   # load checkpoint 检查点文件的路径改名
    if not ckpt_path.endswith('.ckpt'):                                 # ckpt_path变量结尾如果不是.ckpt，将’checkpoints’和’latest.ckpt’添加到路径中，构成一个新的路径
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)             # open(ckpt_path, 'rb')打开文件进行二进制读取，map_location='cpu'指定加载模型到CPU上，pickle_module=dill使用dill模块来加载对象。    
    
    # 从加载的检查点文件中提取配置信息，并打印出模型的名称和数据集的路径
    cfg = payload['cfg']
    print("1.打印模型名称——model_name:", cfg.policy.obs_encoder.model_name)
    print("2.打印数据集路径——dataset_path:", cfg.task.dataset.dataset_path)
    dt = 1/frequency                                                    # 根据frequency（控制频率）计算每个控制周期的时间间隔dt
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)              # 调用get_real_obs_resolution函数来获取真实观测的分辨率，这个分辨率可能来自配置文件中的shape_meta

    fisheye_converter = None                                            # 初始化设置为None，存储鱼眼镜头转换器
    if sim_fov is not None:                                             # 如果sim_fov(模拟视场角)被指定，那么camera_intrinsics(摄像机内在参数)也必须被指定
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(open(camera_intrinsics, 'r')))# 加载摄像机内在参数
        # 创建FisheyeRectConverter对象,将鱼眼镜头的图像转换为普通视角的图像,out_size参数指定输出图像的分辨率，out_fov参数指定输出图像的视场角
        fisheye_converter = FisheyeRectConverter(**opencv_intr_dict, out_size=obs_res, out_fov=sim_fov)
    
    print("3.推理步骤数——steps_per_inference:", steps_per_inference)     # 打印出每个推理步骤的步数
    with SharedMemoryManager() as shm_manager:                          # 创建一个共享内存管理器，用于管理多进程间的数据共享
        with    Spacemouse(shm_manager=shm_manager) as sm, \
                KeystrokeCounter() as key_counter, \
                UmiEnv( output_dir=output,                              # 创建双臂机器人控制环境：输出目录、机器人配置、夹爪配置、控制频率、观测图像分辨率等
                    robot_ip=robots_config[0]['robot_ip'],
                    # 抓手部分
                    gripper_ip = '192.168.1.20',# ？
                    gripper_port=1000,# ？
                    frequency=frequency,
                    robot_type=robots_config[0]['robot_type'],          # 'realman'
                    obs_image_resolution=obs_res,
                    obs_float32=True,
                    camera_reorder=[int(x) for x in camera_reorder],
                    init_joints=init_joints,
                    enable_multi_cam_vis=True,
                    camera_obs_latency=0.14,                            # ！！！相机延迟，umi默认0.17，测量为0.126-0.156
                    camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon, # obs
                    robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                    gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                    no_mirror=no_mirror,
                    fisheye_converter=fisheye_converter,
                    mirror_crop=False,# ？
                    mirror_swap=mirror_swap,
                    max_pos_speed=0.05,                                  # 动作速度 action
                    max_rot_speed=0.1,
                    shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)                                        # 设置OpenCV的线程数为2，以优化图像处理
            print("4.Waiting for camera")                               # 打印一条消息，表示正在等待摄像头
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

            device = torch.device('cuda')                           # 获取CUDA设备，表示模型将在GPU上运行
            policy.eval().to(device)                                # 将策略模型设置为评估模式，并将其移动到GPU上
            print("8.0 预热策略推断 Warming up policy inference")       # 打印信息，表示开始预热策略推断。
            obs = env.get_obs()                                     # 从环境env中获取观察数据
            # print('env_obs == ',obs)
            episode_start_pose = list()                             # 初始化列表，存储每个机器人的起始姿势
            pose = np.concatenate([                                 # 构建机器人的姿势数据(末端执行器位置, 末端执行器旋转轴角度)
                obs[f'robot0_eef_pos'],
                obs[f'robot0_eef_rot_axis_angle']
            ], axis=-1)[-1]                                         # 将位置和旋转轴角度沿最后一个轴拼接，并取最后一个元素
            
            print("8.0.1 robot0_eef_pos == ", obs[f'robot0_eef_pos'])
            print("8.0.2 robot0_eef_rot_axis_angle == ", obs[f'robot0_eef_rot_axis_angle'])
            episode_start_pose.append(pose)                         # 将机器人的起始姿势添加到列表中
            with torch.no_grad():                                   # 创建一个无梯度的上下文，用于执行不涉及梯度更新的操作
                policy.reset()                                      # 重置策略，可能是在准备新的推理或实验开始
                obs_dict_np = get_real_umi_obs_dict(                # 将环境的观测转换为一个字典，包含了机器人末端执行器的位置和旋转
                    env_obs=obs,                                    # 从环境env中获取观察数据
                    shape_meta=cfg.task.shape_meta,                 # 获取真实观测的分辨率
                    obs_pose_repr=obs_pose_rep,                     # 从配置中获取观察姿势表示
                    episode_start_pose=episode_start_pose)          # 存储每个机器人的起始姿势
                # 使用策略来预测一个动作,此处有修改
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(np.array(x)).unsqueeze(0).to(device))# lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                # print("8.1 预测结果result",result)
                # 从预测结果中提取第一个动作，并将其转换为CPU上的NumPy数组
                action = result['action_pred'][0].detach().to('cpu').numpy()
                # assert action.shape[-1] == 10 * len(robots_config)
                assert action.shape[-1] == 10
                # 将动作转换为适合机器人执行的形式
                # print('9.action == ', action)
                action = get_real_umi_action(action, obs, action_pose_repr)
                # assert action.shape[-1] == 7 * len(robots_config)
                assert action.shape[-1] == 7
                # 删除result变量，释放内存
                del result

            print('9.策略已准备好 Ready!')
            # time.sleep(20)
            # 控制循环，允许用户通过SpaceMouse进行机器人控制
            while True:
                # ========= human control loop ==========
                print("10.Human in control!")
                robot_states = env.get_robot_state()                    # 从环境中获取机器人的所有状态
                # target_pose = np.asarray(robot_states['TargetTCPPose']) # 机器人的目标姿态
                target_pose = np.asarray(robot_states['ActualTCPPose'])# 机器人的目标姿态
                gripper_states = env.get_gripper_state()                # 从环境中获取所有机器人的夹爪状态
                gripper_target_pos = np.asarray(gripper_states['gripper_position'])
                print("target_pose:{}, gripper_states:{}".format(target_pose,gripper_states))
                control_robot_idx_list = [0]                            # 定义一个控制机器人索引列表，初始只包含索引0的机器人
                t_start = time.monotonic()                              # 获取当前时间，用于后续计算时间间隔
                iter_idx = 0                                            # 初始化迭代索引，用于控制循环
                # 开始一个内部无限循环，用于处理每个控制周期    
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
                            duration = 3.0                                              # robot 移动持续时间为3秒
                            ep = match_replay_buffer.get_episode(match_episode_id)      # 获取匹配剧集的详细信息
                            pos = ep[f'robot0_eef_pos'][0]                              # 获取机器人的末端执行器位置和旋转
                            rot = ep[f'robot0_eef_rot_axis_angle'][0]
                            grip = ep[f'robot0_gripper_width'][0]                       # 获取机器人的夹爪宽度
                            pose = np.concatenate([pos, rot])                           # 将位置和旋转合并为一个数组，表示机器人的姿态
                            env.robot[0].servoL(pose, duration=duration)                # 根据给定的姿态和持续时间，控制机器人的末端执行器移动到目标位置
                            # 为机器人夹爪设置一个路径点，夹爪将在给定时间到达指定宽度，需要位置
                            env.grippers[0].schedule_waypoint(grip, target_time=time.time() + duration)
                            target_pose = pose
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
                        break                                       # 切换到策略控制
                    precise_wait(t_sample)                              # 等待直到达到采样时间
                    # 5.遥操作
                    # 5.1 获取遥操作命令
                    sm_state = sm.get_motion_state_transformed()    # 获取SpaceMouse的运动状态，并转换为世界坐标系get teleop command #
                    # print(sm_state)
                    print("sm_state:{}".format(sm_state))               # 打印运动状态
                    # dpos = sm_state[:3] * (0.5 / frequency)         # 计算平移速度，基于SpaceMouse的——平移——状态和控制频率
                    # drot_xyz = sm_state[3:] * (1.5 / frequency)     # 计算旋转速度，基于SpaceMouse的——旋转——状态和控制频率
                    dpos = sm_state[:3] * (env.max_pos_speed / frequency)       # 计算位置增量
                    drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)   # 计算旋转增量
                    drot = st.Rotation.from_euler('xyz', drot_xyz)  # 将旋转速度转换为一个旋转对象
                    
                    # 更新目标姿态，根据SpaceMouse的输入和当前控制机器人列表
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                    dpos = 0                                        # 重置平移速度变量

                    # # 如果SpaceMouse的第一个按钮被按下，则设置夹爪的平移速度为负值，表示夹爪关闭
                    # if sm.is_button_pressed(0):                     # close gripper
                    #     dpos = -gripper_speed / frequency
                    # # 如果SpaceMouse的第二个按钮被按下，则设置夹爪的平移速度为正值，表示夹爪打开
                    # if sm.is_button_pressed(1):
                    #     dpos = gripper_speed / frequency
                    # # 更新夹爪的目标位置，确保夹爪的位置在0到最大夹爪宽度之间
                    # for robot_idx in control_robot_idx_list:
                    #     gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width)
                    
                    # 如果SpaceMouse的第一个按钮被按下，则设置夹爪的平移速度为负值，表示夹爪关闭
                    if sm.is_button_pressed(0):                     # close gripper
                        dpos = -gripper_speed / frequency
                    # 如果SpaceMouse的第二个按钮被按下，则设置夹爪的平移速度为正值，表示夹爪打开
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    # 更新夹爪的目标位置，确保夹爪的位置在0到最大夹爪宽度之间
                    gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)

                    action = np.zeros(7,)
                    action[:6] = target_pose
                    action[6] = gripper_target_pos# gripper_target_pos
                    
                    # 执行所有机器人的动作，并将动作的时间戳设置为命令目标时间  # execute teleop command
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)                           # 等待直到控制周期的结束时间
                    iter_idx += 1                                       # 增加迭代索引，用于控制内部循环
                
                # ========== policy control loop 策略控制循环的开始==============
                # 开始一个尝试块，用于捕获可能发生的异常
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
                        print(f'12.观测的延迟 Obs latency {time.time() - obs_timestamps[-1]}')
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
                            print('13.接口延迟 Inference latency:', time.time() - s)             # 打印出推理的延迟
                        
                        this_target_poses = action                      # 将策略生成的动作转换为环境中的动作 convert policy action to env actions 
                        # 有无抓手
                        assert this_target_poses.shape[1] == 7                      # 检查动作数组的维度是否正确
                        # 处理每个机器人的目标姿态
                        for target_pose in this_target_poses:
                            solve_table_collision(  # 桌子碰撞————调用solve_table_collision函数，以确保机器人的末端执行器（EEF）和夹爪不会与桌子发生碰撞
                                ee_pose=target_pose[:6],
                                gripper_width=target_pose[6],
                                height_threshold=robots_config[0]['height_threshold']
                            )
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
                        env.exec_actions(                                                   # 执行所有机器人动作，并将动作的时间戳设置为实际的时间点 execute actions
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"15.已提交的步骤数 Submitted {len(this_target_poses)} steps of actions.") # 打印出已提交的步骤数

                        episode_id = env.replay_buffer.n_episodes       # 可视化，获取当前剧集的编号
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

if __name__ == '__main__':
    main()