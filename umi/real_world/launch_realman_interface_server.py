import zerorpc
from rm_py import log_setting
from rm_py import robotic_arm
import scipy.spatial.transform as st
import numpy as np
import torch
import json
import asyncio
import aiofiles
import datetime
import math
import time

class RealmanInterface:
    # 存储目标位姿——movej和插值函数的目标值
    target_pose = None
    # 是否获取到新的位姿
    target_new_flag = 0
    # 全局计数器，排除初始错误姿态
    realman_num = 0
    # 获取当前时间
    current_time = datetime.datetime.now()
    # 格式化时间：例如，年_月_日_时_分
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M')
    # 修改文件名
    file_name1 = 'target_pose_umi_{}.json'.format(formatted_time)
    file_name2 = 'target_pose_insert_{}.json'.format(formatted_time)
    def __init__(self):
        self.robot = robotic_arm.Arm(632, "192.168.1.18")#左侧19，右侧18
        # j_init = [0.0, 0.0, 90.0, 0.0, 0.0, -30.0]
        # j_init = [-50.0, 45.0, 75.0, 0.0, -30.0, -31.0]
        # 关节角速度限制
        # for i in range(6):
        #     self.robot.Set_Joint_Speed(i,30)
        # j_init = [44.24, -29.08, -101.09, 94.03, -30.27, -81.343]# 桌子上
        # j_init = [27.23, -41.04, -120.44, 1.29, 68.55, -93.27]# 腰上
        # j_init = [-1.4195*180/np.pi, 0.726*180/np.pi, -2.7232*180/np.pi, 1.469*180/np.pi, 1.2879*180/np.pi, 4.9387*180/np.pi]# 腰上
        j_init = [80.5, -50.3, 106.5, -26.4, 46.0, -0.2]# 腰上
        self.robot.Movej_Cmd(j_init, 20, 0)
        curr_joint = self.robot.Get_Joint_Degree()[1][:6]
        print("curr_joint == ",curr_joint)
        
        # 初始化标志符
        RealmanInterface.target_new_flag = 0
        self.loop = asyncio.get_event_loop()

    # 存储函数
    async def save_pose_to_file(self, data, filename):
        async with aiofiles.open(filename, 'a') as file:
            await file.write(json.dumps(data) + '\n')

    # 获取当前末端位姿
    def get_ee_pose(self):
        curr_joint = self.robot.Get_Joint_Degree()[1][:6]
        curr_pose = self.robot.Algo_Forward_Kinematics(curr_joint)
        return curr_pose
    
    # 获取当前关节角
    def get_joint_positions(self):
        curr_joint = self.robot.Get_Joint_Degree()[1][:6]
        return curr_joint
    
    # 获取当前关节角速度
    def get_joint_velocities(self):
        curr_joint_vel = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        return curr_joint_vel

    # 获取目标末端位姿
    def get_target_ee_pose(self):
        return RealmanInterface.target_pose

    # 获取目标关节角
    def get_target_joint(self):
        target_joint = self.robot.Algo_Forward_Kinematics(RealmanInterface.target_pose)
        return target_joint
    
    # 移动到目标位置——关节运动
    def move_to_joint_positions(self, positions, vel, time_to_go):
        self.robot.Movej_Cmd(positions, 20, 0)

    # 移动到目标位置——透传：末端位置逆解关节
    def update_desired_ee_pose(self, pose):
        # print("5.透传——末端目标位姿，target_pose == ",target_pose)# target_pose[0:3]
        # 接收到新位姿的标志符
        RealmanInterface.target_new_flag = 1
        # 接收到新目标位姿
        RealmanInterface.target_pose = pose
        # 数据存储
        # self.loop.run_until_complete(self.save_pose_to_file(pose, RealmanInterface.file_name1))
        # 自编写关节透传
        tag = self.insert(pose)
        return tag
    
    # 策略关闭后，清空机械臂连接
    def terminate_current_policy(self):
        # 为了调试方便，注释掉
        # self.robot.RM_API_UnInit()
        # self.robot.Arm_Socket_Close()
        return 1

    # 逆解和插值
    def insert(self, pose_target):
        # 将新位姿标志置0
        RealmanInterface.target_new_flag = 0
        
        # 拒绝接收前6个干扰位姿
        # RealmanInterface.realman_num +=1
        # print("RealmanInterface.realman_num",RealmanInterface.realman_num)
        # if RealmanInterface.realman_num <6:
        #     return
        
        # 获取当前位姿
        curr_joint = self.robot.Get_Joint_Degree()[1][:6]
        curr_array = np.array(curr_joint)
        # print("curr_joint[:6]:",curr_joint)
        # print("curr_joint[:6]:",curr_array)

        # 相对位姿的预处理
        # curr_pose = self.robot.Algo_Forward_Kinematics(curr_joint)[3:]
        # print("curr_pose[3:]:",curr_pose)
        # pose_target[3] = curr_pose[0] + pose_target[3]/360
        # pose_target[4] = curr_pose[1] + pose_target[4]/360
        # pose_target[4] = curr_pose[2] + pose_target[5]/360
        # pose_target[3] = curr_pose[0]
        # pose_target[4] = curr_pose[1]
        # pose_target[4] = curr_pose[2]
        # print("target_pose[3:]:",pose_target[3:])
        
        # 逆解出目标位姿态
        joint_inv= self.robot.Algo_Inverse_Kinematics(curr_joint, pose_target,1)[1][:6]   # print("规划的joint_inv:",joint_inv) 
        inv_array = np.array(joint_inv)                                             # print("规划的joint_inv:",inv_array) 

        # 插值-数量计算
        num0 = abs((inv_array[0] - curr_array[0]) / 0.1)
        num1 = abs((inv_array[1] - curr_array[1]) / 0.1)
        num2 = abs((inv_array[2] - curr_array[2]) / 0.1)
        num3 = abs((inv_array[3] - curr_array[3]) / 0.1)
        num4 = abs((inv_array[4] - curr_array[4]) / 0.1)
        num5 = abs((inv_array[5] - curr_array[5]) / 0.1)
        # print("6.0.0",num0,num1,num2,num3,num4,num5)
        max_num = max(num0, num1, num2, num3, num4, num5)
        num_int = math.ceil(max_num)
        print("num_int ==", num_int)
        if num_int == 0 :
            print("6.0.1 num_int==", num_int)
            num_int = 1
            return 1
        elif num_int >100:
            print("6.0.2 num_int >40, 20度, num_int==", num_int)
            num_int = 1
            return 1
        
        # 计算渐进量，并依次执行
        joint_int = (inv_array - curr_array) / num_int
        for i in range(num_int):
            pose_target = curr_joint + joint_int * i
            pose_target_list = pose_target.tolist()
            tag = self.robot.Movej_CANFD(pose_target_list, 0)
            time.sleep(0.001)
            if RealmanInterface.target_new_flag == 1:
                print("计划运行", num_int, "次，实际运行", i, "次，收到新位姿，重新规划与透传")
                break
            else:
                # 执行出错
                if tag :
                    print("出错, tag ==", tag)
                    # print("break,break,break,break")
                    # print("6.0.0",num0,num1,num2,num3,num4,num5)
                    # print("6.0.1 pose0_array==",pose0_array)
                    # print("6.0.2 pose_target_array==",pose_target_array)
                    # print("6.0.3 num==",num_int,"joint_int == ",joint_int)
                    break
                # 每运行100次，存储1次结果
                if i % 100 == 0: 
                    # print("6.2.1 透传——i== ",i)
                    # print("6.2.2 末端期望位姿pose == ",pose_target_list)
                    curr_joint = self.robot.Get_Joint_Degree()[1][:6]
                    # print("6.2.3 当前关节角度curr_joint == ",curr_joint)
                    # self.loop.run_until_complete(self.save_pose_to_file(pose_target_list, RealmanInterface.file_name2))
        return 1
    

s = zerorpc.Server(RealmanInterface())
s.bind("tcp://127.0.0.1:5555")
s.run()