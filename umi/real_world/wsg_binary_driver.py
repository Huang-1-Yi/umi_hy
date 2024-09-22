# 导入所需的模块
from typing import Union, Optional
import socket
import enum
import struct




# test_num = 0

# CRC16校验表
CRC_TABLE_CCITT16 = [   
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
    0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
    0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
    0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
    0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
    0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
    0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
    0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
    0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
    0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
    0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
    0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
    0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
    0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
    0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
    0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
    0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
    0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0
]

# 计算CRC-16校验和
def checksum_update_crc16(data: bytes, crc: int=0xFFFF):
    # 初始化CRC值为0xFFFF（通常的CRC初始值）
    # crc参数是可选的，如果未提供，则默认为0xFFFF
    # data参数是一个字节序列，表示需要计算CRC的数据
    for b in data:
        # 遍历data中的每个字节
        # b是当前遍历到的字节
        crc = CRC_TABLE_CCITT16[(crc ^ b) & 0x00FF] ^ ( crc >> 8 )
        # 计算CRC值，使用XOR操作和CRC表
        # 首先，将当前CRC值与数据字节b进行XOR操作
        # 然后，使用CRC表计算新的CRC值
        # XOR操作可以理解为异或，即相同为0，不同为1
        # CRC表是一个预先计算好的表，用于加速CRC计算过程
        # CRC表中的每个值对应于CRC值的一个特定状态
        # 这里使用的是CCITT16标准的CRC表
        # 最后，使用右移操作将CRC值的高位与低位进行交换
        # 这样可以保持CRC值的正确性，并准备下一个字节的计算
    return crc      # 返回计算后的CRC值

# 定义状态码
class StatusCode(enum.IntEnum):
    E_SUCCESS = 0
    E_NOT_AVAILABLE = 1
    E_NO_SENSOR = 2
    E_NOT_INITIALIZED = 3
    E_ALREADY_RUNNING = 4
    E_FEATURE_NOT_SUPPORTED = 5
    E_INCONSISTENT_DATA = 6
    E_TIMEOUT = 7
    E_READ_ERROR = 8
    E_WRITE_ERROR = 9
    E_INSUFFICIENT_RESOURCES = 10
    E_CHECKSUM_ERROR = 11
    E_NO_PARAM_EXPECTED = 12
    E_NOT_ENOUGH_PARAMS = 13
    E_CMD_UNKNOWN = 14
    E_CMD_FORMAT_ERROR = 15
    E_ACCESS_DENIED = 16
    E_ALREADY_OPEN = 17
    E_CMD_FAILED = 18
    E_CMD_ABORTED = 19
    E_INVALID_HANDLE = 20
    E_NOT_FOUND = 21
    E_NOT_OPEN = 22
    E_IO_ERROR = 23
    E_INVALID_PARAMETER = 24
    E_INDEX_OUT_OF_BOUNDS = 25
    E_CMD_PENDING = 26
    E_OVERRUN = 27
    RANGE_ERROR = 28
    E_AXIS_BLOCKED = 29
    E_FILE_EXIST = 30

# 定义命令ID
class CommandId(enum.IntEnum):
    Disconnect = 0x07
    Homing = 0x20
    PrePosition = 0x21
    Stop = 0x22
    FastStop = 0x23
    AckFastStop = 0x24

# 用于将参数转换为字节
def args_to_bytes(*args, int_bytes=1):
    buf = list()
    for arg in args:
        if isinstance(arg, float):
            # little endian 32bit float
            buf.append(struct.pack('<f', arg))
        elif isinstance(arg, int):
            buf.append(arg.to_bytes(length=int_bytes, byteorder='little'))
        elif isinstance(arg, str):
            buf.append(arg.encode('ascii'))
        else:
            raise RuntimeError(f'Unsupported type {type(arg)}')
    result = b''.join(buf)
    return result


# WSGBinaryDriver类
class WSGBinaryDriver:
    # 初始化
    def __init__(self, hostname='192.168.1.20', port=1000):
        self.hostname = hostname
        self.port = port
        self.tcp_sock = None

    # 启动TCP连接
    def start(self):
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((self.hostname, self.port))
        # self.ack_fast_stop()
    
    # 停止TCP连接和脚本
    def stop(self):
        self.stop_cmd()
        self.disconnect()
        self.tcp_sock.close()
        return
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    # ================= low level API ================
    # 发送消息
    def msg_send(self, cmd_id: int, payload: bytes):
        preamble_b = 0xAA.to_bytes(1, 'little') * 3
        cmd_b = int(cmd_id).to_bytes(1, 'little')
        size_b = len(payload).to_bytes(2, 'little')
        msg_b = preamble_b + cmd_b + size_b + payload
        checksum_b = checksum_update_crc16(msg_b).to_bytes(2, 'little')
        msg_b += checksum_b
        return self.tcp_sock.send(msg_b)
    
    # 接收消息
    def msg_receive(self) -> dict:
        # syncing
        sync = 0
        while sync != 3:
            res = self.tcp_sock.recv(1)
            if res == 0xAA.to_bytes(1, 'little'):
                sync += 1
        
        # read header
        cmd_id_b = self.tcp_sock.recv(1)
        cmd_id = int.from_bytes(cmd_id_b, 'little')

        # read size
        size_b = self.tcp_sock.recv(2)
        size = int.from_bytes(size_b, 'little')
        
        # read payload
        payload_b = self.tcp_sock.recv(size)# cmd.send(id, etob(E_SUCCESS), state % 256, ntob(pos), ntob(speed), ntob(force), ntob(time));
        status_code = int.from_bytes(payload_b[:2], 'little')

        parameters_b = payload_b[2:]

        # read checksum
        checksum_b = self.tcp_sock.recv(2)
        
        # correct checksum ends in zero
        header_checksum = 0x50f5
        msg_checksum = checksum_update_crc16(
            cmd_id_b + size_b + payload_b + checksum_b, crc=header_checksum)
        if msg_checksum != 0:
            raise RuntimeError('Corrupted packet received from WSG')
        
        result = {
            'command_id': cmd_id,
            'status_code': status_code,
            'payload_bytes': parameters_b
        }
        return result
    
    # 提交命令
    def cmd_submit(self, cmd_id: int, payload: bytes=b'', pending: bool=True, ignore_other=False):
        res = self.msg_send(cmd_id, payload)
        if res < 0:
            raise RuntimeError("Message send failed.")

        # receive response, repeat if pending
        msg = None
        keep_running = True
        while keep_running:
            msg = self.msg_receive()
            if ignore_other and msg['command_id'] != cmd_id:
                continue

            if msg['command_id'] != cmd_id:
                raise RuntimeError(
                    "Response ID ({:02X}) does not match submitted command ID ({:02X})\n".format(
                    msg['command_id'], cmd_id))
            if pending:
                status = msg['status_code']
            keep_running = pending and status == StatusCode.E_CMD_PENDING.value
        return msg

    # ============== mid level API ================
    # 封装了发送命令和处理响应的逻辑
    def act(self, cmd: CommandId, *args, wait=True, ignore_other=False):
        # 定义一个方法 `act`，它接受以下参数：
        # self: 指向当前WSGBinaryDriver对象的引用
        # cmd: 期望执行的命令，类型为CommandId枚举的实例
        # *args: 可变参数列表，用于传递给命令的参数
        # wait: 布尔值，表示是否等待命令执行完毕（默认值为True）
        # ignore_other: 布尔值，表示是否忽略非期望的响应（默认值为False）
        msg = self.cmd_submit(
            cmd_id=cmd.value,
            payload=args_to_bytes(*args),
            pending=wait,
            ignore_other=ignore_other)
        # 调用 `cmd_submit` 方法，该方法用于发送命令和处理响应
        # 传入的参数包括命令ID、参数序列（转换为字节形式）和等待/忽略其他响应的标志
        # 返回的msg是一个包含响应信息的字典
        msg['command_id'] = CommandId(msg['command_id'])
        msg['status_code'] = StatusCode(msg['status_code'])
        # 将消息中的命令ID和状态码转换为相应的枚举类型
        # 这有助于简化后续的代码处理
        status = msg['status_code']
        if status != StatusCode.E_SUCCESS:
            raise RuntimeError(f'Command {cmd} not successful: {status}')
        # 检查状态码是否为成功（E_SUCCESS）
        # 如果不为成功，则抛出运行时错误
        return msg      # 如果状态码为成功，则返回包含响应信息的msg字典

    # =============== high level API ===============
    # 断开连接
    def disconnect(self):
        # use msg_send to no wait for response
        return self.msg_send(CommandId.Disconnect.value, b'')

    # 回原点
    def homing(self, positive_direction=True, wait=True):
        arg = 0
        if positive_direction is None:
            arg = 0
        elif positive_direction:
            arg = 1
        else:
            arg = 2
        return self.act(CommandId.Homing, arg, wait=wait)
    
    # 预定位
    def pre_position(self, 
                     width: float, speed: float, 
                     clamp_on_block: bool=True, wait=True):
        flag = 0
        if clamp_on_block:
            flag = 0
        else:
            flag = 1
        
        return self.act(CommandId.PrePosition, 
                        flag, float(width), float(speed),
                        wait=wait)
    # 故障确认
    def ack_fault(self):
        return self.act(CommandId.AckFastStop, 'ack', wait=False, ignore_other=True)
    
    # 停止命令
    def stop_cmd(self):
        return self.act(CommandId.Stop, wait=False, ignore_other=True)

    # 执行自定义脚本
    def custom_script(self, cmd_id: int, *args):
        # 定义一个方法 `custom_script`，它接受以下参数：
        # self: 指向当前WSGBinaryDriver对象的引用
        # cmd_id: 期望执行的自定义命令ID，类型为整数
        # *args: 可变参数列表，用于传递给自定义命令的参数

        # 自定义负载格式：
        # Custom payload format:
        # 0:	Unused
        # 1..4	float
        # .... one float each
        payload_args = [0]
        for arg in args:
            payload_args.append(float(arg))
        payload = args_to_bytes(*payload_args, int_bytes=1)
    
        # 发送消息
        # send message
        msg = self.cmd_submit(cmd_id=cmd_id, payload=payload, pending=False)
        # 调用 `cmd_submit` 方法，该方法用于发送命令和处理响应
        # 传入的参数包括命令ID、参数序列（转换为字节形式）和等待/忽略其他响应的标志
        # 返回的msg是一个包含响应信息的字典
        status = StatusCode(msg['status_code'])
        response_payload = msg['payload_bytes']
        # global test_num #全局变量 测试状态值
        # if test_num == 0:
        #     print("status = StatusCode(msg['status_code']) == ", status)
        #     # print("状态值已知")
        #     test_num = 1
        # 如果状态码为未知命令，抛出运行时错误
        if status == StatusCode.E_CMD_UNKNOWN:
            raise RuntimeError('Command unknown - make sure script (cmd_measure.lua) is running')
        # 如果状态码不是成功，抛出运行时错误
        if status != StatusCode.E_SUCCESS:
            raise RuntimeError('Command failed')
        # 如果响应负载长度不正确，抛出运行时错误
        if len(response_payload) != 17:
            raise RuntimeError("Response payload incorrect (", 
                               "".join("{:02X}".format(b) for b in response_payload),
                               ")")
        # 解析负载
        # parse payload
        state = response_payload[0]
        values = list()
        for i in range(4):
            start = i * 4 + 1
            end = start + 4
            values.append(struct.unpack('<f', response_payload[start:end])[0])



        # 
        info = {
            'state': state,
            'position': values[0],#  - 0.01
            'velocity': values[1],
            'force_motor': values[2],
            'measure_timestamp': values[3],
            'is_moving': (state & 0x02) != 0
        }
        # info = {
        #     'state': 0,
        #     'position': 100.,
        #     'velocity': 0.,
        #     'force_motor': 0.,
        #     'is_moving': 0.
        # }
        # 如果需要，可以在此处添加更多的信息解析和处理逻辑
        return info

    # 查询脚本状态
    def script_query(self):
        return self.custom_script(0xB0)
    
    # 位置比例微分控制
    def script_position_pd(self, 
                           position: float, velocity: float,
                           dt: float = 0.1,
                           kp: float=15.0, kd: float=1e-3,
                           travel_force_limit: float=80.0, 
                           blocked_force_limit: float=None):
        if blocked_force_limit is None:
            blocked_force_limit = travel_force_limit
        assert kp > 0
        assert kd >= 0
        return self.custom_script(0xB1, position, velocity, kp, kd, travel_force_limit, blocked_force_limit)

# 测试函数
def test():
    # 导入所需的模块
    import numpy as np
    import time

    # 创建一个WSGBinaryDriver实例
    # with WSGBinaryDriver(hostname='wsg50-00004544.internal.tri.global', port=1000) as wsg:
    with WSGBinaryDriver(hostname='192.168.1.20', port=1000) as wsg:    
        # 进入上下文管理器，确保在操作完成后正确关闭连接
        # ACK（确认故障）
        # msg = wsg.cmd_submit(0x24, bytearray([0x61, 0x63, 0x6B]))
        msg = wsg.ack_fault()
        # print(msg)

        # HOME（回原点）
        # msg = wsg.cmd_submit(0x20, bytearray([0x01]))
        msg = wsg.homing()
        # print(msg)
        # time.sleep(1.0)

        # 预定位
        msg = wsg.pre_position(9, 150)
        # print(msg)
        time.sleep(1.0)

        # 设置时间参数
        T = 2
        dt = 1/30
        # 生成一系列位置点
        pos = np.linspace(0., 110., int(T/dt))[::-1]
        vel = np.diff(pos) / dt
        vel = np.append(vel, vel[-1])

        # 开始时间
        t_start = time.time()
        # 循环执行位置点
        for i in range(len(pos)):
            p = pos[i]
            v = vel[i]
            # print(p, v)
            # 发送位置命令
            info = wsg.script_position_pd(position=p, velocity=v, dt=dt)
            # info = wsg.script_position(position=p, dt=dt)
            # print(info)
            # 计算时间差
            t_end = t_start + i * dt
            t_sleep = t_end - time.time()
            # print(t_sleep)
            # 如果时间差大于0，则等待
            if t_sleep > 0:
                time.sleep(t_sleep)
        # 打印总时间
        print("第一次循环", time.time() - t_start)
        # cmd_id_b, payload_b, checksum_b = wsg.msg_receive()
        # cmd_id_b, payload_b, checksum_b = wsg.msg_receive()
        time.sleep(3.0)

        # 再次生成位置点
        T = 2
        dt = 1/30
        pos = np.linspace(0., 110., int(T/dt))
        vel = np.diff(pos) / dt
        vel = np.append(vel, vel[-1])
        # 再次开始时间
        t_start = time.time()
        for i in range(len(pos)):
            p = pos[i]
            v = vel[i]
            # print(p, v)
            info = wsg.script_position_pd(position=p, velocity=v, dt=dt)
            # info = wsg.script_position(position=p, dt=dt)
            # print(info)
            # 再次循环执行位置点
            t_end = t_start + i * dt
            t_sleep = t_end - time.time()
            # print(t_sleep)
            if t_sleep > 0:
                time.sleep(t_sleep)
        print("第二次循环", time.time() - t_start)
        # 打印总时间

        # 发送自定义消息
        # wsg.msg_send(0x30, bytearray([0x00, 0x00, 0x00, 0x00, 0x16, 0x43]))
        # cmd_id_b, payload_b, checksum_b = wsg.msg_receive()
        # time.sleep(1.0)
if __name__ == '__main__':
    test()