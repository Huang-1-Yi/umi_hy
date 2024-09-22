# %%
# 这段代码是一个使用Python编写的机器人控制程序，它使用umi库来控制UR机器人。
# 程序的主要功能是通过读取SpaceMouse的输入来控制机器人的运动。
import sys
import os
# 这几行代码首先导入了一些必要的Python库，包括系统相关的库。然后，它计算了项目的根目录，并将该目录添加到Python的模块搜索路径中，以确保所有的模块都可以正确导入。最后，它将当前工作目录更改为项目的根目录。
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
"""
time 用于时间相关的操作。
numpy 是一个用于数值计算的库。
SharedMemoryManager 用于在多进程之间共享内存。
scipy.spatial.transform 用于处理空间变换。
umi.real_world.spacemouse_shared_memory 是一个与SpaceMouse输入设备交互的库。
umi.real_world.rtde_interpolation_controller 是一个用于控制UR机器人的库。
umi.common.precise_sleep 用于精确地等待一段时间
"""
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.common.precise_sleep import precise_wait

# %%首先设置了一些机器人控制相关的参数，如机器人的IP地址、最大速度、频率等。
# 然后，它创建了一个SharedMemoryManager实例，用于在多进程之间共享内存。
# 接着，它创建了一个RTDEInterpolationController实例，用于控制UR机器人，以及一个Spacemouse实例，用于读取SpaceMouse的输入。
def main():
    robot_ip = 'ur-2017356986.internal.tri.global'
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    frequency = 30
    cube_diag = np.linalg.norm([1,1,1])
    j_init = None
    j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
    tcp_offset = 0.13
    # tcp_offset = 0
    command_latency = 1/100
    dt = 1/frequency

    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            lookahead_time=0.1,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0],
            # joints_init=j_init,
            verbose=False) as controller:
            with Spacemouse(shm_manager=shm_manager) as sm:
                print('Ready!')
                # to account for recever interfance latency, use target pose
                # to init buffer.
                state = controller.get_state()
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    precise_wait(t_sample)
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                    
                    # if not sm.is_button_pressed(0):
                    #     # translation mode
                    #     drot_xyz[:] = 0
                    # else:
                    #     dpos[:] = 0
                    # if not sm.is_button_pressed(1):
                    #     # 2D translation mode
                    #     dpos[2] = 0    

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()

                    controller.schedule_waypoint(target_pose, 
                        t_command_target-time.monotonic()+time.time())
                    precise_wait(t_cycle_end)
                    iter_idx += 1


# %%
if __name__ == '__main__':
    main()
