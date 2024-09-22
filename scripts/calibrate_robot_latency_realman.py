# %%
import zerorpc
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
from umi.real_world.realman_interpolation_controller import RealmanInterpolationController
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.common.precise_sleep import precise_wait
from umi.common.latency_util import get_latency
from matplotlib import pyplot as plt

class RealmanInterface:
    # 初始化方法,创建一个zerorpc.Client实例,用于与Franka机器人服务器通信
    def __init__(self):
        self.server = zerorpc.Client(heartbeat=20)
        # self.server.connect(f"tcp://{ip}:{port}")
        self.server.connect("tcp://127.0.0.1:5555")
    
    # 获取机器人的末端执行器（EE）的当前姿态————从服务器获取EE的pose,使用转换矩阵tx_flange_tip将EE的pose从Franka的flange坐标系转换到机器人工具坐标系（tip坐标系）,返回转换后的EE姿态
    def get_ee_pose(self):
        # flange_pose = np.array(self.server.get_ee_pose())
        # tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        # return tip_pose
        flange_pose = np.array(self.server.get_ee_pose())
        return flange_pose
    
    # 获取机器人的所有关节的当前位置___从服务器获取关节位置,并将其转换为numpy数组,返回关节位置数组
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    # 获取机器人的所有关节的当前速度___从服务器获取关节速度,并将其转换为numpy数组,返回关节速度数组
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())
    
    # 控制机器人移动到指定的关节位置___接受一个关节位置数组和移动所需的时间,将关节位置转换为列表格式,并发送给服务器以控制机器人移动。
    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)
    
    # # 启动笛卡尔空间的阻抗控制。接受两个阻抗控制参数数组Kx和Kxd,将参数转换为列表格式,并发送给服务器以启动阻抗控制
    # def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
    #     self.server.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())
    
    # 更新机器人末端执行器的期望姿态。接受一个期望的EE姿态数组,将姿态转换为列表格式,并发送给服务器以更新期望姿态。
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())
    
    # 终止当前策略,调用服务器的方法以终止当前的策略或控制模式。
    def terminate_current_policy(self):
        self.server.terminate_current_policy()
    
    # 关闭与Franka机器人服务器的连接,调用服务器的方法以关闭连接
    def close(self):
        self.server.close()



# %%
@click.command()
@click.option('-rh', '--robot_hostname', default='192.168.1.18')
@click.option('-f', '--frequency', type=float, default=30)
def main(robot_hostname, frequency):
    # max_pos_speed = 0.5
    # max_rot_speed = 1.2
    max_pos_speed=0.3
    max_rot_speed=0.3
    cube_diag = np.linalg.norm([1,1,1])
    # tcp_offset = 0.21
    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2

    with SharedMemoryManager() as shm_manager:
        # with RTDEInterpolationController(
        #     shm_manager=shm_manager,
        #     robot_ip=robot_hostname,
        #     frequency=500,
        #     lookahead_time=0.1,
        #     gain=300,
        #     max_pos_speed=max_pos_speed*cube_diag,
        #     max_rot_speed=max_rot_speed*cube_diag,
        #     tcp_offset_pose=[0,0,tcp_offset,0,0,0],
        #     get_max_k=10000,
        #     verbose=False
        # with FrankaInterpolationController(
        #     shm_manager=shm_manager,
        #     robot_ip=robot_hostname,
        #     frequency=200,
        #     Kx_scale=np.array([0.8,0.8,1.2,3.0,3.0,3.0]),
        #     Kxd_scale=np.array([2.0,2.0,2.0,2.0,2.0,2.0]),
        #     verbose=False
        with RealmanInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            robot_port = 8080, 
            frequency=200,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            launch_timeout=3,
            joints_init_speed=0.2,
            get_max_k=10000,
            verbose=False
        ) as controller:
        # ) as controller,\
        # Spacemouse(
        #     shm_manager=shm_manager
        # ) as sm:
            print('Ready!')
            # 为了计算接收器接口延迟，使用目标姿态to account for recever interfance latency, use target pose
            # 初始化缓冲区。to init buffer.
            # 原代码
            state = controller.get_state()
            target_pose = state['ActualTCPPose']
            # hy
            print("1.当前位置ActualTCPPose==",target_pose)
            target_positions = target_pose.copy()
            i = 0.01
            # target_pose = controller.get_joint_positions()
            # print("2.当前关节角度target_pose==",target_pose)
            # target_pose = controller.get_ee_pose()
            # print("1.当前关节角度target_pose==",target_pose)
            # 获取机器人的初始状态和关节位置，并设置目标关节位置为关节0增加10度
            # robot = RealmanInterface()
            # curr_pose = robot.get_ee_pose()
            # print("1.当前关节角度ActualQ, curr_joint==",curr_pose)
            # target_joint_positions = curr_pose.copy()
            # target_joint_positions[0] += 10
            # robot.update_desired_ee_pose(target_joint_positions)
            # print("2.目标关节角度target_joint_positions==", target_joint_positions)

            t_start = time.time()
            
            t_target = list()
            x_target = list()

            iter_idx = 0
            while True:
                
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                precise_wait(t_sample, time_func=time.time)

                # # 原代码
                # sm_state = sm.get_motion_state_transformed()            # 获取 SpaceMouse 的运动状态——位置和旋转
                # # print(sm_state)                                       # sm_state[:3] 获取运动状态的前 3 个元素，表示位置增量
                # dpos = sm_state[:3] * (max_pos_speed / frequency)       # 位置增量，按最大速度和频率缩放
                # drot_xyz = sm_state[3:] * (max_rot_speed / frequency)   # 旋转增量，按最大旋转速度和频率缩放
                # # 使用 scipy.spatial.transform 模块将欧拉角旋转 drot_xyz 转换为旋转对象 drot。
                # # 更新 target_pose 的位置部分（前 3 个元素）和旋转部分（后 3 个元素）。
                # # 旋转部分的更新是通过将当前旋转向量（target_pose[3:]）与 drot 相乘得到的。
                # drot = st.Rotation.from_euler('xyz', drot_xyz)
                # target_pose[:3] += dpos
                # target_pose[3:] = (drot * st.Rotation.from_rotvec(
                #     target_pose[3:])).as_rotvec()
                
                # t_target.append(t_command_target)               # 记录计划的时间点
                # x_target.append(target_pose.copy())             # 记录当前的目标姿态

                # # 调用控制器的 schedule_waypoint 方法，传递新的目标姿态和计划时间点，控制器将根据这些信息进行运动规划
                # controller.schedule_waypoint(target_pose, 
                #     t_command_target)
                # # 如果 SpaceMouse 的任一按钮被按下，跳出循环
                # if sm.is_button_pressed(0) or sm.is_button_pressed(1):
                #     # close gripper
                #     break
                # # 等待当前循环周期结束并递增循环索引
                # precise_wait(t_cycle_end, time_func=time.time)
                # iter_idx += 1
                
                # 直接更新关节
                
                target_positions = target_pose.copy()
                target_positions[0] -= i/2
                target_positions[1] += i/2
                target_positions[2] += i/2
                target_positions[3] += i/4
                target_positions[4] += i/4
                target_positions[5] += i/4
                # target_positions[3] += i/2
                print("3.目标位置target_pose ==", target_positions)
                t_target.append(t_command_target)
                x_target.append(target_positions.copy())
                controller.schedule_waypoint(target_positions, t_command_target)

                if iter_idx > 60 or target_positions[0] < -0.4 :  # Run the loop for a short period to capture data
                    break
                precise_wait(t_cycle_end, time_func=time.time)
                iter_idx += 1
                i += 0.01
                # time.sleep(1)

            states = controller.get_all_state()
            print(f"Collected {len(t_target)} target time points.")
            print(f"Collected {len(x_target)} target positions.")
            print(f"Collected {len(states['robot_receive_timestamp'])} actual time points.")
            print(f"Collected {len(states['ActualTCPPose'])} actual positions.")


    t_target = np.array(t_target)
    x_target = np.array(x_target)
    t_actual = states['robot_receive_timestamp']
    x_actual = states['ActualTCPPose']
    if len(t_target) == 0 or len(x_target) == 0 or len(t_actual) == 0 or len(x_actual) == 0:
        print("Error: Collected data is empty. Exiting...")
        return
    
    n_dims = 6
    fig, axes = plt.subplots(n_dims, 3)
    fig.set_size_inches(15, 15, forward=True)

    for i in range(n_dims):
        try:
            latency, info = get_latency(x_target[...,i], t_target, x_actual[...,i], t_actual)
        except ValueError as e:
            print(f"Error in computing latency for dimension {i}: {e}")
            continue

        row = axes[i]
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_xlabel('lag')
        ax.set_ylabel('cross-correlation')
        ax.set_title(f"Action Dim {i} Cross Correlation")

        ax = row[1]
        ax.plot(t_target, x_target[...,i], label='target')
        ax.plot(t_actual, x_actual[...,i], label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Raw observation")

        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Aligned with latency={latency}")

    fig.tight_layout()
    plt.show()



# %%
if __name__ == '__main__':
    main()
