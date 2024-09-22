"""
机器人控制,使用多个库和模块来与外部设备进行交互,并处理命令行输入
使用click库来处理命令行参数,numpy库来处理数值计算,scipy.spatial.transform库来处理空间变换
umi库来控制Franka机器人、SpaceMouse输入设备和WSG抓手
"""
import sys
import os
# 获取当前文件的根目录,并将该目录添加到Python的模块搜索路径中,然后更改当前工作目录到该根目录,确保Python能够正确地找到和加载相关的模块
# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# print(ROOT_DIR)
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)
"""
click 是一个用于创建命令行接口的库
time 用于时间相关的操作
numpy 是一个用于数值计算的库
SharedMemoryManager 用于在多进程之间共享内存
scipy.spatial.transform 用于处理空间变换
umi.real_world.spacemouse_shared_memory 是一个与SpaceMouse输入设备交互的库
umi.real_world.wsg_controller 是一个用于控制WSG抓手的库
umi.common.precise_sleep 用于精确地等待一段时间
umi.real_world.keystroke_counter 是一个用于计数按键事件的库
???
umi.real_world.realman_interpolation_controller 是一个用于控制Franka机器人的库
???
"""
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.wsg_controller import WSGController
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.realman_interpolation_controller import RealmanInterpolationController

# 接受五个参数,机器人IP、夹爪的主机名和端口,以及频率和夹爪速度
@click.command()
@click.option('-rh', '--robot_hostname', default='172.16.0.3')
@click.option('-gh', '--gripper_hostname', default='172.24.95.27')
@click.option('-gp', '--gripper_port', type=int, default=1000)
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-gs', '--gripper_speed', type=float, default=200.0)
def main(robot_hostname, gripper_hostname, gripper_port, frequency, gripper_speed):
    max_pos_speed = 0.25    # 最大位置速度
    max_rot_speed = 0.6
    max_gripper_width = 90.
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.13
    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2
    # 使用SharedMemoryManager来创建一个共享内存区域,以便在不同的进程间共享数据
    with SharedMemoryManager() as shm_manager:
        with WSGController(shm_manager=shm_manager,hostname=gripper_hostname,port=gripper_port, frequency=frequency,move_max_speed=400.0,verbose=False) as gripper, \
        KeystrokeCounter() as key_counter, \
        RealmanInterpolationController(shm_manager=shm_manager,robot_ip=robot_hostname, frequency=100,Kx_scale=5.0,Kxd_scale=2.0,verbose=False) as controller, \
        Spacemouse(shm_manager=shm_manager) as sm:
            print('Ready!')
            # 考虑接收器接口的延迟,使用目标姿态 to account for recever interfance latency, use target pose
            # to init buffer.
            state = controller.get_state()
            # target_pose = state['TargetTCPPose']
            target_pose = state['ActualTCPPose']
            # print(target_pose)
            # exit()
        
            # target_pose = np.array([ 0.40328411,  0.00620825,  0.29310859, -2.26569407,  2.12426248, -0.00934497])
            # controller.servoL(target_pose, 5)
            # time.sleep(8)
            # exit()
        
            gripper_target_pos = gripper.get_state()['gripper_position']# 从返回的夹爪字典里提取位置信息
            t_start = time.monotonic()                                  # 当前时间戳
            gripper.restart_put(t_start-time.monotonic() + time.time()) # 重新启动WSG抓手的放置动作
            # 初始化 迭代索引iter_idx 和 停止标志stop
            iter_idx = 0
            stop = False
            # 持续运行直到变量stop被设置为True循环体内部处理了一些与时间相关的计算,并更新了一些状态变量
            while not stop:
                state = controller.get_state() # 获取控制器的状态,并检查是否有按键事件 
                # print(target_pose - state['ActualTCPPose'])
                s = time.time()
                t_cycle_end = t_start + (iter_idx + 1) * dt # 当前循环结束的时间戳==循环开始的时间戳,第几次循环,时间间隔
                t_sample = t_cycle_end - command_latency    # 期望接收到命令的时间戳==command_latency是从命令发送到执行之间的延迟
                t_command_target = t_cycle_end + dt         # 下一个命令的时间戳

                # handle key presses 返回一个包含按键事件的列表,键盘or空间鼠标
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    # if key_stroke != None:
                    #     print(key_stroke)
                    if key_stroke == KeyCode(char='q'):         # 按下了’q’键,则设置stop为True
                        stop = True
                precise_wait(t_sample)                          # 精确地等待一段时间
                sm_state = sm.get_motion_state_transformed()    # 获取SpaceMouse的运动状态,并计算平移和旋转速度
                # print(sm_state)
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos                         # 将dpos添加到目标姿态target_pose的平移部分
                target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec() # 将drot应用于目标姿态的旋转部分

                # 检查SpaceMouse的按钮状态,并根据按钮按下情况调整夹爪的目标位置
                dpos = 0
                if sm.is_button_pressed(0):
                    # close gripper
                    dpos = -gripper_speed / frequency
                if sm.is_button_pressed(1):
                    dpos = gripper_speed / frequency
                gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)
                # 调度控制器到目标姿态的路径点
                controller.schedule_waypoint(target_pose, t_command_target-time.monotonic()+time.time())
                # 调度夹爪到目标位置的路径点
                gripper.schedule_waypoint(gripper_target_pos, t_command_target-time.monotonic()+time.time())
                precise_wait(t_cycle_end)# 精确等待到下一个周期结束
                iter_idx += 1
                # print(1/(time.time() -s))# 计算并打印循环时间,即当前时间减去循环开始时间的倒数

    # 可能是用于安全地停止机器人控制器的当前策略 没有这个东西，而是下面这个
    # self.realman.RM_API_UnInit()
    # self.realman.Arm_Socket_Close()
    # controller.terminate_current_policy()
# %%
if __name__ == '__main__':
    main()