# %%
# 处理空间鼠标的输入，并根据输入更新机械臂的目标姿态和夹爪位置，然后调度新的姿态和位置给机械臂和夹爪，以实现平滑和精确的运动控制
# 导入库和设置根目录(当前目录)
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.wsg_controller import WSGController
from umi.common.precise_sleep import precise_wait

# %%
# 使用Click库定义命令行参数，包括机械臂和夹爪的主机名、端口、频率和速度等
@click.command()
@click.option('-rh', '--robot_hostname', default='172.24.95.9')
@click.option('-gh', '--gripper_hostname', default='172.24.95.27')
@click.option('-gp', '--gripper_port', type=int, default=1000)
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-gs', '--gripper_speed', type=float, default=200.0)
def main(robot_hostname, gripper_hostname, gripper_port, frequency, gripper_speed):
    # 设置最大位置速度、最大旋转速度和最大夹爪宽度等参数
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    max_gripper_width = 90.
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.13
    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2
    # 创建一个共享内存管理器，用于在不同进程之间共享内存
    with SharedMemoryManager() as shm_manager:
        # 创建一个WSG控制器，用于控制夹爪
        # 创建一个RTDE插值控制器，用于控制机械臂
        # 创建一个空间鼠标对象，用于接收用户的输入
        with WSGController( shm_manager=shm_manager,
            hostname=gripper_hostname, port=gripper_port,
            frequency=frequency, move_max_speed=400.0, verbose=False ) as gripper,\
        RTDEInterpolationController( shm_manager=shm_manager,
            robot_ip=robot_hostname, frequency=125, lookahead_time=0.05, gain=1000,
            max_pos_speed=max_pos_speed*cube_diag, max_rot_speed=max_rot_speed*cube_diag,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0], verbose=False ) as controller,\
        Spacemouse( shm_manager=shm_manager ) as sm:
            # 设置目标姿态和夹爪目标位置，以初始化缓冲区
            print('Ready!')
            # 为了补偿接收器接口的延迟，使用目标姿态to account for recever interfance latency, use target pose
            # 初始化缓冲区to init buffer.
            state = controller.get_state()
            target_pose = state['TargetTCPPose']
            gripper_target_pos = gripper.get_state()['gripper_position']
            t_start = time.monotonic()
            gripper.restart_put(t_start-time.monotonic() + time.time())
            
            iter_idx = 0
            # 处理用户的输入并控制机械臂和夹爪
            while True:
                s = time.time()     # 未使用
                t_cycle_end = t_start + (iter_idx + 1) * dt # 计算当前循环结束的时间
                t_sample = t_cycle_end - command_latency    # 计算样本时间 t_sample，这是循环结束时间减去命令延迟
                t_command_target = t_cycle_end + dt         # 

                precise_wait(t_sample)  # 使用 precise_wait 函数等待 t_sample 时间，以确保足够的时间延迟
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)

                # 计算位置增量 dpos 和旋转增量 drot_xyz
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

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
 
                # 调度新的姿态和夹爪位置给机械臂和夹爪
                controller.schedule_waypoint(target_pose, 
                    t_command_target-time.monotonic()+time.time())
                gripper.schedule_waypoint(gripper_target_pos, 
                    t_command_target-time.monotonic()+time.time())

                # 在调度新的姿态和位置之前等待足够的时间
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()