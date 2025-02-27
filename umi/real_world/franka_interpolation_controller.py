import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from umi.shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
import torch
from umi.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc

# 枚举类，用于定义命令类型。通常用于在多进程或多线程环境中清晰地标识和传递命令
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

# 将一个坐标系（可能是工具坐标系）相对于另一个坐标系（可能是基坐标系）旋转90度并平移
tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])
# 将坐标系旋转45度，然后再旋转90度
tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()
# 将坐标系旋转45度
tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()
# 两个相互逆的4x4转换矩阵,将坐标系从tx_flange_tip转换到tx_tip_flange，反之亦然。
tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface:
    # 初始化方法，创建一个zerorpc.Client实例，用于与Franka机器人服务器通信
    def __init__(self, ip='192.168.1.167', port=4242): # 172.16.0.3 连接到Franka机器人服务器的指定IP地址和端口号
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")
        print('server.connect(f"tcp://{ip}:{port}")')
                # self.server.connect(f"tcp://{ip}:{port}")
        # self.server.connect("tcp://127.0.0.1:5555")
        # self.server.connect("tcp://192.168.1.111:5555")
    
    # 获取机器人的末端执行器（EE）的当前姿态————从服务器获取EE的pose，使用转换矩阵tx_flange_tip将EE的pose从Franka的flange坐标系转换到机器人工具坐标系（tip坐标系），返回转换后的EE姿态
    def get_ee_pose(self):
        flange_pose = np.array(self.server.get_ee_pose())
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        return tip_pose
    
    # 获取机器人的所有关节的当前位置___从服务器获取关节位置，并将其转换为numpy数组，返回关节位置数组
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    # 获取机器人的所有关节的当前速度___从服务器获取关节速度，并将其转换为numpy数组，返回关节速度数组
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())
    
    # 控制机器人移动到指定的关节位置___接受一个关节位置数组和移动所需的时间，将关节位置转换为列表格式，并发送给服务器以控制机器人移动。
    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)
    
    # 启动笛卡尔空间的阻抗控制。接受两个阻抗控制参数数组Kx和Kxd，将参数转换为列表格式，并发送给服务器以启动阻抗控制
    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())
    
    # 更新机器人末端执行器的期望姿态。接受一个期望的EE姿态数组，将姿态转换为列表格式，并发送给服务器以更新期望姿态。
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())
    
    # 终止当前策略，调用服务器的方法以终止当前的策略或控制模式。
    def terminate_current_policy(self):
        self.server.terminate_current_policy()
    
    # 关闭与Franka机器人服务器的连接，调用服务器的方法以关闭连接
    def close(self):
        self.server.close()


class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    为了确保向机器人发送命令时具有可预测的延迟，这个控制器需要它自己的独立进程（由于Python的全局解释器锁（GIL））。
    """
    def __init__(self, shm_manager: SharedMemoryManager, robot_ip, robot_port=4242, frequency=1000,
        Kx_scale=1.0, Kxd_scale=1.0, launch_timeout=3,
        joints_init=None, joints_init_duration=None, soft_real_time=False,  # 预设启用软实时调度
        verbose=False, get_max_k=None, receive_latency=0.0):    # 是否接收日志、frequency * 5、接收延迟
        """
        robot_ip        中层控制器的IP地址（NUC）the ip of the middle-layer controller (NUC)
        frequency       对于franka来说设置为10001000 for franka
        Kx_scale        位置增益的比例the scale of position gains
        Kxd             速度增益的比例the scale of velocity gains
        soft_real_time  启用轮询调度和实时优先级enables round-robin scheduling and real-time priority
            需要在之前运行脚本/rtprio_setup.shrequires running scripts/rtprio_setup.sh before hand.
        """
        # 将joints_init转换为一个NumPy数组，该数组的形状是7
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")# 设置控制器的一个标识名称
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:   # 配置后面创建的循环缓冲区（ring buffer）的大小或行为
            get_max_k = int(frequency * 5)

        # build input queue
        # 创建一个共享内存队列，队列中的每个元素都将具有与这个示例相同的结构。
        # 用于在不同的进程间传递控制命令，其中每个命令都包含命令类型、目标位姿、持续时间、目标时间
        example = {'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0, 'target_time': 0.0
        }
        # 使用之前定义的 example 字典来创建一个共享内存队列 input_queue，这个队列可以在不同的进程之间安全地传递消息
        input_queue = SharedMemoryQueue.create_from_examples(shm_manager=shm_manager, examples=example, buffer_size=256)

        # 创建循环缓冲区 build ring buffer
        # 定义receive_keys列表，其中包含了三个元组，每个元组都定义了一个键和相应的函数名
        # 从机器人获取不同的数据，如末端执行器的实际位姿 ActualTCPPose、实际关节位置 ActualQ 和实际关节速度 ActualQd
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys: # 机器人的7个关节 or 末端执行器的6个自由度
            if 'joint' in func_name:        # 如果 func_name 中包含字符串 'ee_pose'，则表示这是与末端执行器的位姿相关的数据
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:    # 如果 func_name 中包含字符串 'ee_pose'，则表示这是与末端执行器的位姿相关的数据
                example[key] = np.zeros(6)
        # 添加两个时间戳键
        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        # 使用这个 example 字典和其它参数调用 SharedMemoryRingBuffer.create_from_examples 方法来创建循环缓冲区 ring_buffer
        # 用于在不同的进程间共享从机器人接收到的实时数据
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(shm_manager=shm_manager,
            examples=example,get_max_k=get_max_k,
            get_time_budget=0.2,put_desired_frequency=frequency
        )
        # 在类的其他方法中提供对共享内存队列、循环缓冲区和进程间同步事件对象的访问
        # 类的其他方法就可以使用这些变量来进行进程间通信和数据处理，例如发送控制命令、接收机器人状态数据等
        self.ready_event = mp.Event()       # 创建一个多进程事件对象 ready_event，并将其赋值给实例变量 self.ready_event。这个事件对象可以用于进程间的同步，例如，一个进程可以等待直到另一个进程设置了这个事件
        self.input_queue = input_queue      # 将之前创建的共享内存队列 input_queue 赋值给实例变量 self.input_queue。这个队列用于在不同进程间传递控制命令
        self.ring_buffer = ring_buffer      # 将之前创建的共享内存循环缓冲区 ring_buffer 赋值给实例变量 self.ring_buffer。这个缓冲区用于在不同进程间共享从机器人接收到的实时数据。
        self.receive_keys = receive_keys    # 将之前定义的 receive_keys 列表赋值给实例变量 self.receive_keys。这个列表包含了从机器人接收到的数据的不同类型及其对应的获取方法名。
            
    # ========= launch method ===========
    # 启动控制器进程，并提供了一个选项，以确定是否需要等待进程完全启动，以及是否需要输出启动信息
    def start(self, wait=True):
        super().start()         # 调用父类，启动控制器进程
        if wait:
            self.start_wait()   # 等待控制器进程完全启动并准备接收命令
        if self.verbose:        # 提供调试信息，帮助开发者了解进程的运行情况
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")
    # 停止机器人或其他进程，等待操作完成
    def stop(self, wait=True):
        message = {'cmd': Command.STOP.value}   # 消息的目的地是停止机器人的当前操作或进程
        self.input_queue.put(message)           # 将这个停止消息放入输入队列 self.input_queue 中
        if wait:                                # 等待机器人或其他进程安全地停止
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    # 控制机器人以特定的速度和加速度移动到目标位姿
    def servoL(self, pose, duration=0.1):
        """
        duration: 达到期望姿态所需的时间 desired time to reach pose
        """
        assert self.is_alive()                      # 检查控制器进程是否仍然存活
        assert(duration >= (1/self.frequency))      # 确保期望的持续时间 duration 至少大于或等于一个控制周期的时间
        pose = np.array(pose)
        assert pose.shape == (6,)                   # 将输入的 pose 转换为一个 NumPy 数组，并断言该数组的形状必须是 (6,)

        message = {
            'cmd': Command.SERVOL.value,    # 伺服控制命令，用于精确控制机器人的位姿
            'target_pose': pose,            # 目标位姿，设置为前面转换的 pose 数组
            'duration': duration            # 持续时间，设置为输入参数 duration，这表示机器人应该在该时间内到达指定的位姿
        }
        self.input_queue.put(message)       # 将这个消息放入输入队列 self.input_queue 中。这通常意味着机器人控制进程将会检查这个队列，并在接收到这个命令后执行相应的操作，即控制机器人移动到目标位姿

    # 安排机器人到达一个特定的位姿（waypoint）的时间
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)# 将输入的 pose 转换为一个 NumPy 数组，并断言该数组的形状必须是 (6,)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value, 
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    # ========= receive APIs 机器人当前状态的实时访问=============
    # 允许获取循环缓冲区中的最新状态，或者获取最后 k 个状态
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    # 获取循环缓冲区中的所有状态数据
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time# 启用软实时调度
        if self.soft_real_time:
            os.sched_setscheduler(                  # 设置当前进程的调度策略为 SCHED_RR（轮询调度），并且设置一个优先级参数
                0, os.SCHED_RR, os.sched_param(20)) # 进程以循环方式运行，并且尝试均匀分配 CPU 时间
            
        # start polymetis interface 与 Franka 机器人通信的接口
        robot = FrankaInterface(self.robot_ip, self.robot_port)

        # 初始化控制循环，并确保机器人按照指定的增益和策略运行
        # 不断地从机器人接收状态更新，执行控制命令，并更新机器人的位姿
        # 机器人控制器能够响应不同的控制命令，并根据命令的类型更新其行为
        # 还确保了控制循环以设定的频率运行，并提供了调试信息以帮助开发者监控控制器的运行状态
        try:
            # verbose=1，需要显示，打印一条消息，表明控制器正在连接到机器人，并显示机器人的 IP 地址
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            # init pose 将机器人移动到初始关节位置
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init), # 包含初始关节位置的数组
                    time_to_go=self.joints_init_duration    # 机器人移动到这些位置所需的时间
                )

            # main loop 
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()   # 获取当前时间，并确保控制循环不会回退
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(   # 在控制循环中生成机器人位姿的轨迹
                times=[curr_t],
                poses=[curr_pose]
            )


            # 开启franka阻抗控制策略，控制机器人末端执行器的阻抗增益start franka cartesian impedance policy
            robot.start_cartesian_impedance(
                Kx=self.Kx,
                Kxd=self.Kxd
            )

            t_start = time.monotonic()  # 记录控制循环的开始时间
            iter_idx = 0                # 跟踪控制循环的迭代次数
            keep_running = True         # 控制循环的持续时间
            # 不断地从机器人接收状态更新，执行控制命令，并更新机器人的位姿
            # 以设定的频率接收和执行控制命令，从而实现了连续的控制
            while keep_running:
                t_now = time.monotonic()            # 获取当前时间，并确保控制循环不会回退
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                tip_pose = pose_interp(t_now)       # 获取机器人末端执行器的目标位姿
                flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)# 将末端执行器的位姿转换为机器人法兰盘的位姿

                # 更新机器人末端执行器的期望位姿send command to robot 告诉机器人移动到 flange_pose 位置
                robot.update_desired_ee_pose(flange_pose)

                # update robot state 更新机器人状态，并将其存储在 state 字典中
                state = dict()  # 从机器人接收到的不同状态信息，如末端执行器的实际位姿、关节位置和速度
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()
                
                # 更新 state 字典中的时间戳，确保它们与实际接收时间一致
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state) # 将更新后的 state 字典放入循环缓冲区 self.ring_buffer 中，以便后续处理

                # 尝试从输入队列 self.input_queue 中获取控制命令 fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                # 处理从输入队列中获取的控制命令，并相应地更新机器人控制器的行为
                # 器人控制器能够响应不同的控制命令，并根据命令的类型更新其行为。它还确保了控制循环以设定的频率运行，并提供了调试信息以帮助开发者监控控制器的运行状态
                for i in range(n_cmd):
                    # 创建一个名为 command 的字典，检查命令类型 cmd
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    # 如果命令是 Command.STOP.value，则设置 keep_running 为 False，立即停止循环，忽略后续的命令
                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    # 如果命令是 Command.SERVOL.value，则更新机器人控制器的目标位姿 pose_interp，以确保机器人平滑地移动到新的位姿
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    # 如果命令是 Command.SCHEDULE_WAYPOINT.value，则将机器人控制器的目标位姿 pose_interp 安排在未来某个时间点
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    # 对于其他命令类型，设置 keep_running 为 False，立即停止循环
                    else:
                        # huangyi
                        # if self.verbose:
                        #     print(cmd)
                        keep_running = False
                        break

                # 确保控制循环以设定的频率运行 regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt # 下一个期望的循环开始时间
                precise_wait(t_wait_util, time_func=time.monotonic) # t_wait_util 是下一个期望的循环开始时间，time_func 用于获取时间戳

                # first loop successful, ready to receive command
                # 如果这是控制循环的第一次迭代，设置 self.ready_event.set()，这可能会通知其他进程或线程控制器已经准备好接收命令
                if iter_idx == 0:
                    self.ready_event.set()
                # 增加 iter_idx 变量，以跟踪控制循环的迭代次数
                iter_idx += 1
                # 如果 self.verbose 为 True，则打印当前的实际频率(通过计算控制循环的时间间隔得出的)
                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        # 在控制循环结束时执行一些清理操作
        finally:
            # manditory cleanup 内存清理
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.terminate_current_policy()    # 停止机器人当前的策略或控制模式
            # 可能是一个强制性的清理操作，用于确保机器人处于安全状态，例如，在执行任何其他操作之前，机器人可能需要返回到一个安全的位置或姿态
            del robot   # 删除 robot 对象，释放与机器人相关的资源
            self.ready_event.set()  # 设置 self.ready_event.set()，这可能会通知其他进程或线程控制器已经完成所有操作并准备好接收新的命令

            if self.verbose:# 表明控制器已经从机器人断开连接，并显示机器人的 IP 地址
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
