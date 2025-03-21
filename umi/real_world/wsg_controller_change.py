import os
import time
import enum
import multiprocessing as mp
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
# from umi.real_world.wsg_binary_driver import WSGBinaryDriver
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
# 这些导入语句包括操作系统相关功能、时间控制、枚举类型、多进程处理和共享内存管理，以及用于与机械手进行通信和控制的特定模块。
import zerorpc

# SHUTDOWN：关闭控制器。
# SCHEDULE_WAYPOINT：调度新的路径点。
# RESTART_PUT：重新启动。
class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class GripperInterface:
    # 初始化方法，创建一个zerorpc.Client实例，用于与Franka机器人服务器通信
    def __init__(self,hostname, port):
        print("机械爪success connected!!!")
        self.server = zerorpc.Client(heartbeat=20)
        # self.server.connect(f"tcp://{hostname}:{port}")
        self.server.connect("tcp://127.0.0.1:1000")
    
    # 该方法发送一个 AckFastStop 命令以确认快速停止故障。这个命令通常用于在发生快速停止故障后重置机械手。
    def ack_fault(self):
        return self.server.ack_fault()
    
    def homing(self,positive_direction, wait):
        return self.server.homing(positive_direction, wait)
    
    def script_query(self):
        info = {
            'state': 0,
            'position': 100,
            'velocity': 0,
            'force_motor': 0,
            'measure_timestamp': time.time(),
            'is_moving': False
        }
        print("script_query.info == ",info)
        return info
    
    def script_position_pd(self, position, velocity):
        info = {
            'state': 0,
            'position': 100,
            'velocity': 0,
            'force_motor': 0,
            'measure_timestamp': 0,
            'is_moving': 0.
        }
        return info
    

# class GripperInterface:
#     # 初始化方法，创建一个zerorpc.Client实例，用于与Franka机器人服务器通信
#     def __init__(self,hostname, port):
#         self.server = zerorpc.Client(heartbeat=20)
#         # self.server.connect(f"tcp://{hostname}:{port}")
#         self.server.connect("tcp://127.0.0.1:5556")
    
#     # 该方法发送一个 AckFastStop 命令以确认快速停止故障。这个命令通常用于在发生快速停止故障后重置机械手。
#     def ack_fault(self):
#         return self.server.ack_fault()
    
#     def homing(self,positive_direction, wait):
#         return self.server.homing(positive_direction, wait)
    
#     def script_query(self):
#         return self.server.script_query()
    
#     def script_position_pd(self, position, velocity):
#         info = self.server.script_position_pd(position, velocity)# 编写返回值

class WSGController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            hostname,               # 连接机械手的主机名和端口
            port=5556,
            frequency=30,           # 控制器的运行频率
            home_to_open=True,      # 是否将机械手初始化为打开状态
            move_max_speed=200.0,   # 机械手移动的最大速度 200
            get_max_k=None,
            command_queue_size=1024,# 命令队列的大小
            launch_timeout=3,       # 启动超时时间
            receive_latency=0.0,    # 接收延迟
            use_meters=False,       # 是否使用米作为单位
            verbose=False           # 是否启用详细输出
            ):
        super().__init__(name="WSGController")
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # 创建共享内存队列 input_queue 和共享内存环形缓冲区 ring_buffer，用于在进程之间传递数据。build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    # 启动进程，并等待启动完成（如果 wait 为 True）
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[WSGController] Controller process spawned at {self.pid}")

    # 发送关闭命令，并等待进程停止（如果 wait 为 True）
    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    # 用于等待进程启动和停止
    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    # 用于等待进程启动和停止    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= 上下文管理方法 context manager ===========
    #  可以用作上下文管理器，以便在 with 语句中使用
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= 向控制器发送命令 command methods ============
    # schedule_waypoint()：调度一个新的路径点
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # 重新启动命令
    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= 接收状态的方法 receive APIs =============
    # 获取当前状态或最近的 k 个状态
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    # 获取所有状态
    def get_all_state(self):
        # return self.ring_buffer.get_all()

        data = self.ring_buffer.get_all()
        print("now, ring_buffer.get_all() returned:", data)
        
        # 检查返回数据的每个字段是否为空
        if not data:
            print("Ring buffer returned an empty dictionary.")
        
        # for key in ['gripper_timestamp', 'gripper_position']:
        #     if key not in data:
        #         raise ValueError(f"Key {key} is missing in the returned data.")
        #     if len(data[key]) == 0:
        #         raise ValueError(f"Array for key {key} is empty.")
        return data
    

    # ========= main loop in process ============
    def run(self):
        # start connection
        try:
            # 使用 WSGBinaryDriver 连接到机械手。
            with GripperInterface(
                hostname=self.hostname, 
                port=self.port) as wsg:
            # with WSGBinaryDriver(
            #     hostname=self.hostname, 
            #     port=self.port) as wsg:
                
                # 初始化机械手，确保其处于已知状态 home gripper to initialize
                # wsg.ack_fault()
                # wsg.homing(positive_direction=self.home_to_open, wait=True)

                # 查询机械手的初始位置和时间，并初始化 PoseTrajectoryInterpolator 对象以管理路径点插值。get initial
                curr_info = wsg.script_query()
                # curr_info = {
                #     'state': 0,
                #     'position': 100,
                #     'velocity': 0,
                #     'force_motor': 0,
                #     'measure_timestamp': 0,
                #     'is_moving': 0.
                # }
                print("当前信息!!!!script_query.info == ",info)

                curr_pos = curr_info['position']
                # curr_pos = 100.0
                curr_t = time.monotonic()
                last_waypoint_time = curr_t
                pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[[curr_pos,0,0,0,0,0]]
                )
                # 设置一个标志位 keep_running 控制循环，记录开始时间 t_start 和迭代计数 iter_idx。
                keep_running = True
                t_start = time.monotonic()
                iter_idx = 0
                while keep_running:
                    # command gripper
                    # 计算当前时间 t_now，时间步长 dt，目标时间 t_target，目标位置 target_pos 和目标速度 target_vel，
                    # 然后发送控制命令 script_position_pd 以设定机械手的位置和速度
                    t_now = time.monotonic()
                    dt = 1 / self.frequency
                    t_target = t_now
                    target_pos = pose_interp(t_target)[0]
                    target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt
                    # print('controller（target_pos, target_vel） == ', target_pos, target_vel)
                    info = wsg.script_position_pd(
                        position=target_pos, velocity=target_vel)
                    # info = curr_info


                    # time.sleep(1e-3)

                    # get state from robot
                    # 将机械手的状态信息（位置、速度、力等）存储在共享内存环形缓冲区中
                    state = {
                        'gripper_state': info['state'],
                        'gripper_position': info['position'] / self.scale,
                        'gripper_velocity': info['velocity'] / self.scale,
                        'gripper_force': info['force_motor'],
                        'gripper_measure_timestamp': info['measure_timestamp'],
                        'gripper_receive_timestamp': time.time(),
                        'gripper_timestamp': time.time() - self.receive_latency
                    }
                    # 添加调试信息，检查 state 的内容
                    print("State to be written to ring buffer:", state)
                    # 确保所有字段都不为空
                    print("State to be written to ring buffer:", state)
                    for key, value in state.items():
                        if value is None or (isinstance(value, np.ndarray) and len(value) == 0):
                            raise ValueError(f"Value for key {key} is invalid: {value}")
                    
                    self.ring_buffer.put(state)
                    # 在写入数据后再次检查共享内存中的数据
                    all_data = self.ring_buffer.get_all()
                    print("Data in ring buffer after write:", all_data)



                    # 从命令队列中获取所有命令，如果队列为空，设置命令数量 n_cmd 为 0 fetch command from queue
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0
                    
                    # execute commands
                    for i in range(n_cmd):
                        # command = dict()
                        # for key, value in commands.items():
                        #     command[key] = value[i]
                        command = {key: value[i] for key, value in commands.items()}
                        cmd = command['cmd']
                        # 如果是 SHUTDOWN 命令，设置 keep_running 为 False，退出循环。
                        # 如果是 SCHEDULE_WAYPOINT 命令，调度新的路径点，更新路径插值器 pose_interp。
                        # 如果是 RESTART_PUT 命令，重置起始时间和迭代索引
                        if cmd == Command.SHUTDOWN.value:
                            keep_running = False
                            # stop immediately, ignore later commands
                            break
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_pos = command['target_pos'] * self.scale
                            target_time = command['target_time']
                            # translate global time to monotonic time
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[target_pos, 0, 0, 0, 0, 0],
                                time=target_time,
                                max_pos_speed=self.move_max_speed,
                                max_rot_speed=self.move_max_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = target_time
                        elif cmd == Command.RESTART_PUT.value:
                            t_start = command['target_time'] - time.time() + time.monotonic()
                            iter_idx = 1
                        else:
                            keep_running = False
                            break
                        
                    # 在第一次循环成功后，设置 ready_event 表示控制器准备就绪，并增加迭代计数 first loop successful, ready to receive command
                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx += 1
                    
                    # 使用 precise_wait 调节循环频率，确保控制器以设定的频率运行 regulate frequency
                    dt = 1 / self.frequency
                    t_end = t_start + dt * iter_idx
                    precise_wait(t_end=t_end, time_func=time.monotonic)
        # 在主循环结束后，进入 finally 块，确保在任何情况下都能设置 ready_event 并断开连接        
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[WSGController] Disconnected from robot: {self.hostname}")