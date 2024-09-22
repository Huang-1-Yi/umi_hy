# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from umi.real_world.real_env import RealEnv
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import ( KeystrokeCounter, Key, KeyCode )

@click.command()
@click.option('--output', '-o', required=True)
@click.option('--robot_ip', default='172.24.95.9')
@click.option('--gripper_ip', default='172.24.95.17')
# @click.option('--robot_ip', default='172.24.95.8')
# @click.option('--gripper_ip', default='172.24.95.18')
@click.option('--vis_camera_idx', default=0, type=int)
@click.option('--init_joints', '-j', is_flag=True, default=False)
@click.option('-gs', '--gripper_speed', type=float, default=200.0)
def main(output, robot_ip, gripper_ip, vis_camera_idx, init_joints, gripper_speed):
    max_gripper_width = 90.

    frequency = 10
    command_latency = 1/100
    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                gripper_ip=gripper_ip,
                n_obs_steps=2,
                obs_image_resolution=(256,256),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(1)

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            gripper_target_pos = max_gripper_width
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            # 处理用户输入，并根据输入控制环境的行为，例如开始、结束回合或重置环境
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt # 计算当前循环结束的时间
                t_sample = t_cycle_end - command_latency    # 计算样本时间 t_sample，这是循环结束时间减去命令延迟
                t_command_target = t_cycle_end + dt         # 

                # pump obs 获取环境中的观测数据。这可能是环境的当前状态，包括位置、速度、传感器数据等
                obs = env.get_obs()

                # handle key presses 获取按键计数器中的按键事件
                press_events = key_counter.get_press_events()
                # 遍历按键事件列表
                for key_stroke in press_events:
                    # if key_stroke != None:
                    #     print(key_stroke)
                    # 如果按键是’q’，则设置stop为True，这可能会停止循环或结束程序
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    # 如果按键是’c’，则开始一个新回合（episode）。这可能包括重置环境或设置新的初始状态。
                    elif key_stroke == KeyCode(char='c'):
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    # 如果按键是’s’，则结束当前回合 
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    # 如果按键是退格键，则确认是否要丢弃当前回合。如果是，则丢弃当前回合并重置环境
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                # 获取空格键的计数，这可能表示某种阶段或状态
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                # 使用 precise_wait 函数等待 t_sample 时间，以确保足够的时间延迟
                precise_wait(t_sample)
                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                # 计算位置增量 dpos 和旋转增量 drot_xyz
                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                # 更新机械臂的目标姿态 target_pose，包括位置和旋转
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()
                
                dpos = 0
                # 检查空间鼠标按钮状态，以确定是否需要打开或关闭夹爪
                if sm.is_button_pressed(0):
                    # close gripper
                    dpos = -gripper_speed / frequency
                if sm.is_button_pressed(1):
                    dpos = gripper_speed / frequency
                # 将 gripper_target_pos 加上 dpos 的值，但结果必须限制在 0 到 max_gripper_width 之间
                gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)

                action = np.zeros((7,))
                action[:6] = target_pose
                action[-1] = gripper_target_pos

                # execute teleop command
                env.exec_actions(
                    actions=[action], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                
                # 在调度新的姿态和位置之前等待足够的时间
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
