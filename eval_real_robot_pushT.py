"""
Usage:  python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>
================ Human in control ==============
Robot movement:
    移动您的SpaceMouse以移动机器人末端执行器(EEF)(在xy平面上锁定)
    按SpaceMouse右键以解锁z轴
    按SpaceMouse左键以启用旋转轴
Recording control:
    点击opencv窗口(确保它是焦点)
    按“C”开始评估(将控制权交给策略)
    按“Q”退出程序。
================ Policy in control ==============
    确保您能够快速按下机器人硬件紧急停止按钮！
    Recording control:
    按“S”停止评估并取回控制权.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st

from diffusion_policy.common.cv2_util                       import get_image_transform
from diffusion_policy.common.precise_sleep                  import precise_wait
from diffusion_policy.common.pytorch_util                   import dict_apply
from diffusion_policy.policy.base_image_policy              import BaseImagePolicy
from diffusion_policy.real_world.real_env                   import RealEnv
from diffusion_policy.real_world.real_inference_util        import (get_real_obs_resolution, get_real_obs_dict)
from diffusion_policy.real_world.spacemouse_shared_memory   import Spacemouse
from diffusion_policy.workspace.base_workspace              import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True) # 在OmegaConf中注册一个新的解析器，用于执行eval函数，并替换已有的解析器

@click.command() # 定义一个Click命令行接口
@click.option('--input', '-i', required=True, help='Path to checkpoint')            # required=True，帮助信息为“检查点的路径”
@click.option('--output', '-o', required=True, help='Directory to save recording')  # required=True，帮助信息为“保存记录的目录”
@click.option('--robot_ip', '-ri', default='192.168.1.10', help="UR5's IP address e.g. 192.168.0.204")   # required=True,必需，UR5的IP地址，例如 192.168.0.204
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')       # 默认值为None，用于覆盖和调整初始条件的数据集
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')   # 默认值为None，类型为int，匹配数据集中指定的集
@click.option('--vis_camera_idx', default=1, type=int, help="Which RealSense camera to visualize.")         # --vis_camera_idx，默认值为0，类型为int，帮助信息为“要可视化的RealSense相机”
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.") # -默认值为False，是否在开始时初始化机器人关节配置
@click.option('--steps_per_inference', '-si', default=12, type=int, help="Action horizon for inference.")   # 默认值为6，类型为int，帮助信息为“推理的动作视界”
@click.option('--max_duration', '-md', default=180, help='Max duration for each epoch in seconds.')         # 180，默认值为60，每个周期的最大持续时间(秒)
@click.option('--frequency', '-f', default=5, type=float, help="Control frequency in Hz.")                  # 默认值为10，控制频率(Hz)
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.") # 从接收SpaceMouse命令到在机器人上执行的延迟(秒)”
def main(input, output, robot_ip, match_dataset, match_episode, vis_camera_idx, init_joints, steps_per_inference, max_duration, frequency, command_latency):
    match_camera_idx = 0                    # 定义变量match_camera_idx，初始值为0
    episode_first_frame_map = dict()        # 定义字典episode_first_frame_map，用于存储每集的第一个帧
    if match_dataset is not None:           # 如果match_dataset不为空
        match_dir = pathlib.Path(match_dataset)         # 将match_dataset转换为路径对象
        match_video_dir = match_dir.joinpath('videos')  # 获取视频目录的路径
        for vid_dir in match_video_dir.glob("*/"):      # 遍历视频目录中的每个子目录
            episode_idx = int(vid_dir.stem)                                 # 获取子目录名称并转换为整数，作为集索引
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')  # 获取匹配视频路径
            if match_video_path.exists():                                   # 如果匹配视频存在
                frames = skvideo.io.vread(str(match_video_path), num_frames=1)  # 读取视频的第一个帧
                episode_first_frame_map[episode_idx] = frames[0]                # 将第一个帧存入字典
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")  # 打印加载的集数量

    # 加载检查点
    ckpt_path = input                           # 获取检查点路径
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill) # 使用dill模块加载检查点
    cfg = payload['cfg']                        # 获取配置
    cls = hydra.utils.get_class(cfg._target_)   # 获取配置中的目标类
    workspace = cls(cfg)                        # 实例化工作区
    workspace: BaseWorkspace                    # 声明类型
    workspace.load_payload(payload, exclude_keys=None, include_keys=None) # 加载检查点数据

    # 方法(diffusion/robomimic/ibc)特定的设置
    action_offset = 0           # 定义动作偏移量，初始值为0
    delta_action = False        # 定义是否为增量动作，初始值为False
    if 'diffusion' in cfg.name:         # DP模型
        policy: BaseImagePolicy         # 声明策略类型
        policy = workspace.model        # 获取工作区中的模型作为策略
        if cfg.training.use_ema:        # 如果使用EMA模型
            policy = workspace.ema_model    # 获取EMA模型作为策略
        device = torch.device('cuda')   # 使用CUDA设备
        policy.eval().to(device)        # 设置策略为评估模式并移动到CUDA设备
        # 设置推理参数
        policy.num_inference_steps = 16 # DDIM推理迭代次数
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1 # 动作步数

    elif 'robomimic' in cfg.name:       # BCRNN模型
        policy: BaseImagePolicy         # 声明策略类型
        policy = workspace.model        # 获取工作区中的模型作为策略
        device = torch.device('cuda')   # 使用CUDA设备
        policy.eval().to(device)        # 设置策略为评估模式并移动到CUDA设备
        # BCRNN总是有1的动作视界
        steps_per_inference = 1         # 设置每次推理步骤为1
        action_offset = cfg.n_latency_steps             # 设置动作偏移为配置中的延迟步骤数
        delta_action = cfg.task.dataset.get('delta_action', False) # 获取是否为增量动作

    elif 'ibc' in cfg.name:             # IBC模型
        policy: BaseImagePolicy         # 声明策略类型
        policy = workspace.model        # 获取工作区中的模型作为策略
        policy.pred_n_iter = 5          # 设置预测迭代次数为5
        policy.pred_n_samples = 4096    # 设置预测样本数为4096
        device = torch.device('cuda')   # 使用CUDA设备
        policy.eval().to(device)        # 设置策略为评估模式并移动到CUDA设备
        steps_per_inference = 1         # 设置每次推理步骤为1
        action_offset = 1               # 设置动作偏移为1
        delta_action = cfg.task.dataset.get('delta_action', False) # 获取是否为增量动作

    else:                               # 如果配置名称不支持
        raise RuntimeError("Unsupported policy type: ", cfg.name) # 抛出不支持的策略类型错误

    # 设置实验
    dt = 1/frequency                                        # 计算时间步长
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)  # 获取实际观察分辨率
    n_obs_steps = cfg.n_obs_steps                           # 获取观察步数
    print("n_obs_steps: ", n_obs_steps)                     # 打印观察步数
    print("steps_per_inference:", steps_per_inference)      # 打印每次推理步骤
    print("action_offset:", action_offset)                  # 打印动作偏移

    with SharedMemoryManager() as shm_manager:              # 使用共享内存管理器、SpaceMouse和实际环境
        with Spacemouse(shm_manager=shm_manager) as sm, RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            thread_per_video=3,                 # 每个相机视图的视频录制线程数
            video_crf=21,                       # 视频录制质量，值越低质量越好(但速度较慢)
            shm_manager=shm_manager) as env:    # 初始化实际环境
            cv2.setNumThreads(1)                # 设置OpenCV线程数为1

            # 应与演示相同
            # RealSense曝光设置
            # env.realsense.set_exposure(exposure=120, gain=0) # 设置RealSense相机曝光
            # # RealSense白平衡设置
            # env.realsense.set_white_balance(white_balance=5900) # 设置RealSense相机白平衡

            print("Waiting for realsense")      # 打印等待RealSense消息
            time.sleep(1.0)                     # 等待1秒

            print("Warming up policy inference")# 打印政策推理预热消息
            obs = env.get_obs()                 # 获取观察结果
            with torch.no_grad():               # 禁用梯度计算
                policy.reset()                              # 重置策略
                obs_dict_np = get_real_obs_dict(            # 获取实际观察字典
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))      # 将观察结果转换为张量
                result = policy.predict_action(obs_dict)    # 预测动作
                action = result['action'][0].detach().to('cpu').numpy()         # 获取预测的动作
                assert action.shape[-1] == 2                # 断言动作形状正确
                del result                                  # 删除结果

            print('Ready!') # 打印准备就绪消息
            while True:
                # ========= 人类控制循环 ==========
                print("Human in control!")                  # 打印人类控制消息
                state = env.get_robot_state()               # 获取机器人状态
                target_pose = state['TargetTCPPose']        # 获取目标姿势
                print("state['TargetTCPPose'] == ", target_pose)
                t_start = time.monotonic()                  # 获取当前时间
                iter_idx = 0                                # 初始化迭代索引为0
                while True:
                    # 计算时间
                    # print("t_start ==",t_start)
                    t_cycle_end = t_start + (iter_idx + 1) * dt     # 计算循环结束时间
                    # print("t_cycle_end ==",t_cycle_end)
                    t_sample = t_cycle_end - command_latency        # 计算采样时间
                    t_command_target = t_cycle_end + dt             # 计算命令目标时间

                    # 获取观察结果
                    obs = env.get_obs()                             # 获取观察结果

                    # 可视化
                    episode_id = env.replay_buffer.n_episodes       # 获取当前集ID
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]   # 获取可视化图像
                    match_episode_id = episode_id                   # 设置匹配集ID为当前集ID
                    if match_episode is not None:                   # 如果匹配集不为空，设置匹配集ID为指定集ID
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map: # 如果匹配集ID在字典中
                        match_img = episode_first_frame_map[match_episode_id]   # 获取匹配图像
                        ih, iw, _ = match_img.shape                             # 获取匹配图像形状
                        oh, ow, _ = vis_img.shape                               # 获取可视化图像形状
                        tf = get_image_transform(                               # 获取图像变换函数
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255      # 变换匹配图像
                        vis_img = np.minimum(vis_img, match_img)                # 最小化可视化图像和匹配图像
                    text = f'Episode: {episode_id}'                 # 设置文本为当前集ID
                    cv2.putText(                                    # 在可视化图像上绘制文本
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])        # 显示可视化图像
                    # key_stroke = cv2.pollKey()                    # 获取按键
                    key_stroke = cv2.waitKey(1) & 0xFF
                    if key_stroke == ord('q'):                      # 如果按键为q，退出程序
                        env.end_episode()                           # 结束当前集
                        print("当前键盘输入 == ", key_stroke)
                        print("结束当前集")
                        exit(0)                                     # 退出程序
                    elif key_stroke == ord('c'):                    # 如果按键为c，退出人类控制循环，将控制权交给策略
                        print("当前键盘输入 == ", key_stroke)
                        print("控制权交给策略")
                        break
                    # else:
                    #     print("当前键盘无输入, key_stroke =", key_stroke)
                    precise_wait(t_sample)  # 精确等待采样时间

                    # 5.遥操作
                    # 5.1 获取遥操作命令
                    sm_state = sm.get_motion_state_transformed()    # 获取SpaceMouse的运动状态
                    # print(sm_state)                               # 打印运动状态
                    dpos = sm_state[:3] * (env.max_pos_speed / frequency)       # 计算位置增量
                    drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)   # 计算旋转增量

                    # 旋转轴和Z轴解锁
                    # 如果没有按下第一个按钮，平移模式
                    if not sm.is_button_pressed(0):
                        # print("没有按下第一个按钮, drot_xyz[:] =", drot_xyz[:])
                        drot_xyz[:] = 0                             # 旋转增量置零
                    else:
                        # print("按下第一个按钮, dpos[:] = 0", drot_xyz[:])
                        dpos[:] = 0                                 # 位置增量置零
                    # 如果没有按下第二个按钮，2D平移模式
                    if not sm.is_button_pressed(1): 
                        dpos[2] = 0                                 # Z轴位置增量置零

                    drot = st.Rotation.from_euler('xyz', drot_xyz)  # 计算旋转
                    target_pose[:3] += dpos                         # 更新目标位置
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()               # 更新目标旋转
                    
                    # 5.2 裁剪目标姿势，，评估的特殊操作————hy
                    # target_pose[:2] = np.clip(target_pose[:2], [0.25, -0.45], [0.77, 0.40]) # 裁剪XY轴位置

                    # 5.3 执行遥操作命令
                    env.exec_actions(                               # 执行动作
                        actions=[target_pose], 
                        timestamps=[t_command_target-time.monotonic()+time.time()]) # 设置时间戳
                    precise_wait(t_cycle_end)                       # 精确等待循环结束时间
                    # print("t_cycle_end ==",t_cycle_end)
                    iter_idx += 1                                   # 增加迭代索引
                
                # ========== 策略控制循环 ==============
                # 开始新集
                try:
                    policy.reset()                              # 重置策略
                    start_delay = 1.0                           # 设置开始延迟
                    eval_t_start = time.time() + start_delay    # 计算评估开始时间
                    t_start = time.monotonic() + start_delay    # 计算循环开始时间
                    env.start_episode(eval_t_start)             # 开始新集
                    # 等待1/30秒以获得最接近的帧，实际上减少了整体延迟
                    frame_latency = 1/30                        # 设置帧延迟
                    precise_wait(eval_t_start - frame_latency, time_func=time.time) # 精确等待评估开始时间
                    print("Started!")                           # 打印开始消息
                    iter_idx = 0                                # 初始化迭代索引为0
                    term_area_start_timestamp = float('inf')    # 设置终止区域开始时间戳为无穷大
                    perv_target_pose = None                     # 初始化前一个目标姿势为None
                    while True:
                        # 1.计算时间
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt   # 计算循环结束时间

                        # 2.获取观察结果
                        print('get_obs')                        # 打印获取观察结果消息
                        obs = env.get_obs()                     # 获取观察结果
                        obs_timestamps = obs['timestamp']       # 获取观察时间戳
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')        # 打印观察延迟

                        # 3.运行推理
                        with torch.no_grad():                   # 禁用梯度计算
                            s = time.time()                     # 获取当前时间
                            obs_dict_np = get_real_obs_dict(    # 获取实际观察字典
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))  # 将观察结果转换为张量
                            result = policy.predict_action(obs_dict)                    # 预测动作
                            # 这个动作从第一个观察步骤开始
                            action = result['action'][0].detach().to('cpu').numpy()     # 获取预测的动作
                            print('Inference latency:', time.time() - s)                # 打印推理延迟
                        
                        # 4.将策略动作转换为环境动作
                        if delta_action:                                # 如果是增量动作
                            assert len(action) == 1                             # 断言动作长度为1
                            if perv_target_pose is None:                        # 如果前一个目标姿势为None
                                perv_target_pose = obs['robot_eef_pose'][-1]        # 获取当前的机器人末端执行器姿势
                            this_target_pose = perv_target_pose.copy()          # 复制前一个目标姿势
                            this_target_pose[[0,1]] += action[-1]               # 更新目标姿势的XY位置
                            perv_target_pose = this_target_pose                 # 更新前一个目标姿势
                            this_target_poses = np.expand_dims(this_target_pose, axis=0) # 扩展维度以匹配批处理
                        else:
                            this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64) # 初始化目标姿势数组
                            this_target_poses[:] = target_pose                  # 设置目标姿势
                            this_target_poses[:,[0,1]] = action                 # 更新目标姿势的XY位置

                        # 5.处理时间
                        # 相同步骤的动作总是目标
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]     # 计算动作时间戳
                        action_exec_latency = 0.01          # 设置动作执行延迟
                        curr_time = time.time()             # 获取当前时间
                        is_new = action_timestamps > (curr_time + action_exec_latency)      # 检查新动作
                        if np.sum(is_new) == 0:             # 如果没有新动作
                            # 超过时间预算，仍然做一些事情
                            this_target_poses = this_target_poses[[-1]]                     # 使用最后一个目标姿势
                            # 在下一个可用步骤上安排
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))   # 计算下一个步骤索引
                            action_timestamp = eval_t_start + (next_step_idx) * dt          # 计算动作时间戳
                            print('Over budget', action_timestamp - curr_time)              # 打印超出预算时间
                            action_timestamps = np.array([action_timestamp])                # 设置动作时间戳
                        else:
                            this_target_poses = this_target_poses[is_new]                   # 获取新目标姿势
                            action_timestamps = action_timestamps[is_new]                   # 获取新动作时间戳

                        # 6.裁剪动作
                        # this_target_poses[:,:2] = np.clip(
                        #     this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])           # 裁剪XY轴位置

                        # 7.执行动作
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")      # 打印提交的动作步骤数量

                        # 8.可视化
                        episode_id = env.replay_buffer.n_episodes       # 获取当前集ID
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]   # 获取可视化图像
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start )    # 设置文本为当前集ID和时间
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )                                               # 在可视化图像上绘制文本
                        cv2.imshow('default', vis_img[...,::-1])        # 显示可视化图像

                        key_stroke = cv2.pollKey()                      # 获取按键
                        if key_stroke == ord('s'):                      # 如果按键为s，停止当前集，将控制权交还给人类
                            env.end_episode()   # 结束当前集
                            print('Stopped.')   # 打印停止消息
                            break

                        # 9.自动终止
                        terminate = False                               # 初始化终止标志为False
                        if time.monotonic() - t_start > max_duration:   # 如果超过最大持续时间
                            terminate = True                            # 设置终止标志为True
                            print('Terminated by the timeout!')         # 打印超时终止消息

                        term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04]) # 设置终止姿势
                        curr_pose = obs['robot_eef_pose'][-1]           # 获取当前的机器人末端执行器姿势
                        dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1) # 计算当前姿势和终止姿势之间的距离
                        if dist < 0.03:                                 # 如果距离小于0.03，在终止区域内
                            curr_timestamp = obs['timestamp'][-1]           # 获取当前时间戳
                            if term_area_start_timestamp > curr_timestamp:  # 如果终止区域开始时间戳大于当前时间戳
                                term_area_start_timestamp = curr_timestamp      # 更新终止区域开始时间戳
                            else:
                                term_area_time = curr_timestamp - term_area_start_timestamp # 计算终止区域时间
                                if term_area_time > 0.5:                    # 如果终止区域时间超过0.5秒
                                    terminate = True                            # 设置终止标志为True
                                    print('Terminated by the policy!')          # 打印策略终止消息
                        else:
                                                                        # 在终止区域外
                            term_area_start_timestamp = float('inf')        # 设置终止区域开始时间戳为无穷大
                        if terminate:   # 如果终止标志为True，结束当前集
                            env.end_episode()
                            break

                        # 10.等待执行
                        precise_wait(t_cycle_end - frame_latency) # 精确等待循环结束时间
                        iter_idx += steps_per_inference # 增加迭代索引

                except KeyboardInterrupt: # 如果发生键盘中断
                    print("Interrupted!") # 打印中断消息
                    # (XXXX)停止机器人
                    env.end_episode() # 结束当前集
                
                print("Stopped.") # 打印停止消息




# %%
if __name__ == '__main__':
    main()
