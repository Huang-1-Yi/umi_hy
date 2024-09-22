# This is a sample Python script.

# 按 Shift + F10 来执行当前选中的代码，或者用他们自己的代码替换它   Press Shift+F10 to execute it or replace it with your code.
# Double Shift，用于在 Visual Studio Code 中进行全局搜索        Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import ctypes
import os
import time


import rm_py.log_setting as rm_log
import rm_py.robotic_arm as rm
import sys
import numpy as np
import math

def demo1(robot):
    ret = robot.Movej_Cmd([18.44, -10.677, -124.158, -15, -71.455, 0], 30, 0)
    if ret != 0:
        print("设置初始位置失败:" + str(ret))
        # sys.exit()

    for num in range(0, 3):
        po1 = [0.3, 0.3, 0.3, 3.141, 0, 1.569]

        ret = robot.Movel_Cmd(po1, 30, 0)
        if ret != 0:
            print("Movel_Cmd 1 失败:" + str(ret))
            sys.exit()

        po2 = [0.41674, 0.0925, 0.3, 3.141, 0, 1.569]
        po3 = [0.40785, 0.114, 0.3, 3.141, 0, 1.569]

        ret = robot.Movec_Cmd(po2, po3, 30, 0, 0)
        if ret != 0:
            print("Movec_Cmd 1 失败:" + str(ret))
            sys.exit()

        po4 = [0.364850, 0.157, 0.3, 3.141, 0, 1.569]

        ret = robot.Movel_Cmd(po4, 30, 0)
        if ret != 0:
            print("Movel_Cmd 2 失败:" + str(ret))
            sys.exit()

        po5 = [0.386350, 0.208889, 0.3, 3.141, 0, 1.569]
        po6 = [0.40785, 0.157, 0.3, 3.141, 0, 1.569]

        ret = robot.Movec_Cmd(po5, po6, 30, 0, 0)
        if ret != 0:
            print("Movel_Cmd 2 失败:" + str(ret))
            sys.exit()

    print("cycle draw 8 demo success")

def demo2(robot):
    po0 = [18.44, -10.677, -124.158, -15, -71.455, 0]
    ret = robot.Movej_Cmd(po0, 30, 0)
    if ret != 0:
        print("设置初始位置失败:" + str(ret))
        # sys.exit()

    for num in range(0, 3):
        po1 = [0.3, 0.3, 0.3, 3.141, 0, 1.569]

        ret = robot.Movel_Cmd(po1, 30, 0)
        if ret != 0:
            print("Movel_Cmd 1 失败:" + str(ret))
            sys.exit()

        po2 = [0.25, 0.35, 0.3, 3.141, 0, 1.569]
        po3 = [0.2, 0.4, 0.3, 3.141, 0, 1.569]

        ret = robot.Movec_Cmd(po2, po3, 30, 0, 0)
        if ret != 0:
            print("Movec_Cmd 2 失败:" + str(ret))
            sys.exit()

        po4 = [0.3, 0.25, 0.3, 3.141, 0, 1.569]

        ret = robot.Movel_Cmd(po4, 30, 0)
        if ret != 0:
            print("Movel_Cmd 3 失败:" + str(ret))
            sys.exit()

        po5 = [0.386350, 0.208889, 0.3, 3.141, 0, 1.569]
        po6 = [0.40785, 0.157, 0.3, 3.141, 0, 1.569]

        ret = robot.Movec_Cmd(po5, po6, 30, 0, 0)
        if ret != 0:
            print("Movel_Cmd 4 失败:" + str(ret))
            sys.exit()

    print("cycle draw 8 demo success")

def demo3(robot):
    po0 = [18.44, -10.677, -124.158, -15, -71.455, 0]
    ret = robot.Movej_Cmd(po0, 30, 0)
    if ret != 0:
        print("设置初始位置失败:" + str(ret))
        # sys.exit()
    print(robot.Get_Arm_All_State())
    for num in range(0, 3):
        po1 = [0.3, 0.3, 0.3, 3.141, 0, 1.569]

        ret = robot.Movel_Cmd(po1, 30, 0)
        if ret != 0:
            print("Movel_Cmd 1 失败:" + str(ret))
            sys.exit()
        # for i in range(1,1001):
        #     p11=0.30-i*0.0001
        #     p12=0.30+i*0.0001
        #     pp1 = [p11, p12, 0.3, 3.141, 0, 1.569]
        #     ret = robot.Movep_CANFD(pp1, 0)
        #     time.sleep(0.005)
        #     if ret != 0:
        #         print("Movec_Cmd 2 失败,p11==" + str(p11) + ",p12==" + str(p12))
        #         sys.exit()
        for i in range(1,1001):
            p11=0.30-i*0.0001
            p12=0.30+i*0.0001
            pp1 = [p11, p12, 0.3, 3.141, 0, 1.569]
            ret = robot.Movep_CANFD(pp1, 0)
            time.sleep(0.005)
            if ret != 0:
                print("Movec_Cmd 2 失败,p11==" + str(p11) + ",p12==" + str(p12))
                sys.exit()
            # print(robot.Get_Arm_All_State())
            print("robot.Get_Joint_Degree()==",robot.Get_Joint_Degree())
        print("透传结束")
        time.sleep(1)

    #     po4 = [0.3, 0.25, 0.3, 3.141, 0, 1.569]
    #     ret = robot.Movel_Cmd(po4, 30, 0)
    #     if ret != 0:
    #         print("Movel_Cmd 4 失败:" + str(ret))
    #         sys.exit()
        
    #     po5 = [0.25, 0.20, 0.3, 3.141, 0, 1.569]
    #     ret = robot.Movej_P_Cmd(po5, 30, 0, 0)
    #     if ret != 0:
    #         print("Movec_Cmd 3 失败:" + str(ret))
    #         sys.exit()

    #     po6 = [0.2, 0.15, 0.3, 3.141, 0, 1.569]
    #     ret = robot.Movej_P_Cmd(po6, 30, 0, 0)
    #     if ret != 0:
    #         print("Movec_Cmd 3 失败:" + str(ret))
    #         sys.exit()
            
    print("cycle draw 8 demo success")


def demo4(robot):
    """
    轨迹是不连续的，运行完毕之后，也只会一个个到达目标地点
    """
    po0 = [18.44, -10.677, -124.158, -15, -71.455, 0]
    ret = robot.Movej_Cmd(po0, 30, 0)
    if ret != 0:
        print("设置初始位置失败:" + str(ret))
        # sys.exit()
    print(robot.Get_Arm_All_State())
    for num in range(0, 3):
        po1 = [0.3, 0.3, 0.3, 3.141, 0, 1.569]
        ret = robot.Movel_Cmd(po1, 30, 1, 0.2, 0)
        if ret != 0:
            print("Movel_Cmd 1 失败:" + str(ret))
            sys.exit()
        # time.sleep(1)
        # time.sleep(0.5)
        # time.sleep(0.2)

        po2 = [0.25, 0.35, 0.3, 3.141, 0, 1.569]
        ret = robot.Movej_P_Cmd(po2, 30, 1, 0.2, 0)
        if ret != 0:
            print("Movec_Cmd 2 失败:" + str(ret))
            sys.exit()
        # time.sleep(1)
        # time.sleep(0.5)
        # time.sleep(0.2)

        po3 = [0.2, 0.4, 0.3, 3.141, 0, 1.569]
        ret = robot.Movej_P_Cmd(po3, 30, 0, 0, 0)
        if ret != 0:
            print("Movec_Cmd 3 失败:" + str(ret))
            sys.exit()
        # time.sleep(1)
        # time.sleep(0.5)
        # time.sleep(0.2)

        po4 = [0.3, 0.25, 0.3, 3.141, 0, 1.569]
        ret = robot.Movej_P_Cmd(po4, 30, 0, 0, 0)
        if ret != 0:
            print("Movel_Cmd 4 失败:" + str(ret))
            sys.exit()
        # time.sleep(1)
        # time.sleep(0.5)
        # time.sleep(0.2)

        po5 = [0.25, 0.20, 0.3, 3.141, 0, 1.569]
        ret = robot.Movej_P_Cmd(po5, 30, 0, 0, 0)
        if ret != 0:
            print("Movec_Cmd 5 失败:" + str(ret))
            sys.exit()
        # time.sleep(1)
        # time.sleep(0.5)
        # time.sleep(0.2)

        po6 = [0.2, 0.15, 0.3, 3.141, 0, 1.569]
        ret = robot.Movej_P_Cmd(po6, 30, 0, 0, 0)
        if ret != 0:
            print("Movec_Cmd 6 失败:" + str(ret))
            sys.exit()
        # time.sleep(1)
        # time.sleep(0.5)
        # time.sleep(0.2)
            
    print("cycle draw 8 demo success")
# Press the green button in the gutter to run the script.



if __name__ == '__main__':
    
    # 连接机械臂，注册回调函数
    # callback = CANFD_Callback(mcallback)
    robot = rm.Arm(632, "192.168.1.18")#, pCallback=None
    # print("robot.Get_Joint_Degree()==",robot.Get_Joint_Degree())
    # curr_pose = robot.Get_Joint_Degree()[1][:6]# 只保留前6个有效的关节角度
    # print("curr_pose[:6]==",curr_pose)
    # API版本信息
    # print(robot.API_Version())
    # po0 = [18.44, -10.677, -124.158, -15, -71.455, 0]
    # ret = robot.Movej_Cmd(po0, 30, 0)

    pose1=[0.23913317456610342, -0.28501480696515336, 0.18071728459660596, 1.4742504230910858, -0.6867836907553878, 2.038267437431767]
    # ret = robot.Movej_P_Cmd(pose1, 20, 0)
    ret = robot.Movej_Cmd(pose1, 20, 0)
    
    
    # Algo_Set_ToolFrame()

    ret = robot.Movej_P_Cmd(pose1, 20, 0)





    curr_joint = robot.Get_Joint_Degree()[1][:6]
    print("curr_joint[:6]:",curr_joint)   
    curr_array = np.array(curr_joint)
    print("curr_joint[:6]:",curr_array)

    pose0=[0.23914280533790588, -0.28501081466674805, 0.18070131540298462, -2.07869291305542, 0.9683555960655212, -2.8734805583953857]
    # ret = robot.Movej_Cmd(pose0, 30, 0)
    joint_inv= robot.Algo_Inverse_Kinematics(curr_joint, pose0,1)[1][:6]
    print("规划的joint_inv:",joint_inv) 
    inv_array = np.array(joint_inv)
    print("规划的joint_inv:",inv_array) 

    num0 = abs((inv_array[0] - curr_array[0]) / 1)
    num1 = abs((inv_array[1] - curr_array[1]) / 1)
    num2 = abs((inv_array[2] - curr_array[2]) / 1)
    num3 = abs((inv_array[3] - curr_array[3]) / 1)
    num4 = abs((inv_array[4] - curr_array[4]) / 1)
    num5 = abs((inv_array[5] - curr_array[5]) / 1)
    print("6.0.0",num0,num1,num2,num3,num4,num5)

    max_num = max(num0, num1, num2, num3, num4, num5)
    num_int = math.ceil(max_num)# *10
    joint_int = (inv_array - curr_array) / num_int
    print("6.0.3 num_int==", num_int, "joint_int == ",joint_int)

    i = 0
    for i in range(num_int):
        pose_target = curr_joint + joint_int * i
        pose_target_list = pose_target.tolist()
        # tag = robot.Movep_CANFD(pose_target_list, 0)  # 透传
        tag = robot.Movej_CANFD(pose_target_list, 0)
        time.sleep(0.01)
        if i % 25 == 0: 
            print("6.透传——i== ",i,",末端实际位姿pose == ",pose_target_list)


    # pose0=[0.23914280533790588, -0.28501081466674805, 0.18070131540298462, -2.07869291305542, 0.9683555960655212, -2.8734805583953857]
    # pose1=[0.23913317456610342, -0.28501480696515336, 0.18071728459660596, 1.4742504230910858, -0.6867836907553878, 2.038267437431767]
    
    # pose0_array = np.array(pose0)
    # pose1_array = np.array(pose1)
    # num0 = abs((pose1_array[0] - pose0_array[0]) / 0.001)
    # num1 = abs((pose1_array[1] - pose0_array[1]) / 0.001)
    # num2 = abs((pose1_array[2] - pose0_array[2]) / 0.001)
    # num3 = abs((pose1_array[3] - pose0_array[3]) / 0.001)
    # num4 = abs((pose1_array[4] - pose0_array[4]) / 0.001)
    # num5 = abs((pose1_array[5] - pose0_array[5]) / 0.001)
    # print(num0,num1,num2,num3,num4,num5)
    # max_num = max(num0, num1, num2, num3, num4, num5)
    # num_int = math.ceil(max_num)



    # if num_int == 0 :
    #     num_int = 1
    # elif num_int >1000:
    #     num_int = 1000
    # print("建议拆分num==",num_int)
    # x0 = (pose1_array - pose0_array) / num_int
    # i = 0
    # for i in range(num_int):
    #     pose_target = pose0_array + x0 * i
    #     pose_target_list = pose_target.tolist()
    #     # tag = robot.Movep_CANFD(pose_target_list, 0)  # 透传
    #     if i % 100 == 0: 
    #         print("6.透传——i== ",i,",末端实际位姿pose == ",pose_target)
    
    # curr_joint = robot.Get_Joint_Degree()[1][:6]
    # print("规划的curr_joint:",curr_joint)   
    # joint_inv= robot.Algo_Inverse_Kinematics(curr_joint, pose0,1)[1][:6]
    # print("规划的joint_inv:",joint_inv)   
    # robot.Movej_Cmd(joint_inv, 20, 0)


    # curr_joint = robot.Get_Joint_Degree()[1][:6]
    # ret = robot.Movej_Cmd(pose0, 20, 0)
    # = robot.Algo_Inverse_Kinematics(pose1,)
    # print("规划的YXZ Movej_Cmd ret:" + str(ret))   
 
    # 测试插值算法
    # print("运动到初始位置：, [0, 0, -90, 0, -90, 0]")
    # joint = [0, 0, -90, 0, -90, 0]
    # ret = robot.Movej_Cmd(joint, 20, 0)
    # print("Movej_Cmd ret:" + str(ret))
    # time.sleep(1)
    # curr_joint = robot.Get_Joint_Degree()[1][:6]
    # pose_target_list0 = robot.Algo_Forward_Kinematics(curr_joint)
    # pose0_array = np.array(pose_target_list0)
    # print("pose_target_list0==",pose_target_list0)

    # pose_target_list1 = [-0.49, 0, 0.505, 1.4739833347876368, -0.6866638440295363, 2.0375513689847384]
    # pose1_array = np.array(pose_target_list1)
    # print("pose_target_list1==",pose_target_list1)

    # num0 = abs((pose1_array[0] - pose0_array[0]) / 0.001)
    # num1 = abs((pose1_array[1] - pose0_array[1]) / 0.001)
    # num2 = abs((pose1_array[2] - pose0_array[2]) / 0.001)
    # num3 = abs((pose1_array[3] - pose0_array[3]) / 0.001)
    # num4 = abs((pose1_array[4] - pose0_array[4]) / 0.001)
    # num5 = abs((pose1_array[5] - pose0_array[5]) / 0.001)
    # print(num0,num1,num2,num3,num4,num5)
    # max_num = max(num0, num1, num2, num3, num4, num5)
    # num_int = math.ceil(max_num)
    # print("建议拆分num==",num_int)
    # x0 = (pose1_array - pose0_array) / num_int
    # i = 0
    # for i in range(num_int):
    #     pose_target = pose0_array + x0 * i
    #     pose_target_list = pose_target.tolist()
    #     tag = robot.Movep_CANFD(pose_target_list, 0)  # 透传
    #     if i % 25 == 0: 
    #         print("6.透传——i== ",i,",末端实际位姿pose == ",pose_target[0:3])






    #   初始位置
    # print("运动到初始位置：, [0, 0, -90, 0, -90, 0]")
    # joint = [0, 0, -90, 0, -90, 0]
    # ret = robot.Movej_Cmd(joint, 20, 0)
    # print("Movej_Cmd ret:" + str(ret))
    # time.sleep(1)

    # pose_target_list0 = [-0.49, 0, 0.505, -2.078723430633545, 0.9683522582054138, -2.8734986782073975]
    # # ret = robot.Movep_CANFD(pose_target_list, 0)  # 透传
    # ret = robot.Movej_P_Cmd(pose_target_list0, 30)  # 透传
    # print("读取的当前位姿XYZ Movej_Cmd ret:" + str(ret))
    # time.sleep(1)  
    
    # pose_target_list1 = [-0.49, 0, 0.505, 1.4739833347876368, -0.6866638440295363, 2.0375513689847384]
    # ret = robot.Movej_P_Cmd(pose_target_list1, 30)  # 透传
    # print("规划的XYZ Movej_Cmd ret:" + str(ret))
    # time.sleep(1)

    # pose_target_list = [-0.49, 0, 0.505, 1.4739833347876368, 2.0375513689847384, -0.6866638440295363]
    # ret = robot.Movej_P_Cmd(pose_target_list, 30)  # 透传
    # print("规划的XZY Movej_Cmd ret:" + str(ret))
    # time.sleep(1)
    
    # pose_target_list = [-0.49, 0, 0.505, -0.6866638440295363, 1.4739833347876368, 2.0375513689847384]
    # ret = robot.Movej_P_Cmd(pose_target_list, 30)  # 透传
    # print("规划的YXZ Movej_Cmd ret:" + str(ret))    
    # time.sleep(1)



    # pose_target_list = [-0.49, 0, 0.505, -0.6866638440295363, 2.0375513689847384, 1.4739833347876368]
    # ret = robot.Movej_P_Cmd(pose_target_list, 30)  # 透传
    # print("规划的YZX Movej_Cmd ret:" + str(ret))   
    # time.sleep(3)

    # pose_target_list = [-0.49, 0, 0.505, 2.0375513689847384, -0.6866638440295363, 1.4739833347876368]
    # ret = robot.Movej_P_Cmd(pose_target_list, 30)  # 透传
    # print("规划的ZYX Movej_Cmd ret:" + str(ret))
    # time.sleep(1)

    # pose_target_list = [-0.49, 0, 0.505, 2.0375513689847384, 1.4739833347876368, -0.6866638440295363]
    # ret = robot.Movej_P_Cmd(pose_target_list, 30)  # 透传
    # print("规划的ZXY Movej_Cmd ret:" + str(ret))
    # time.sleep(1)

    # 断开连接
    robot.RM_API_UnInit()
    robot.Arm_Socket_Close()

# if __name__ == '__main__':
    
#     # 连接机械臂，注册回调函数
#     # callback = CANFD_Callback(mcallback)
#     robot = rm.Arm(632, "192.168.1.18")#, pCallback=None
#     # print("robot.Get_Joint_Degree()==",robot.Get_Joint_Degree())
#     curr_pose = robot.Get_Joint_Degree()[1][:6]# 只保留前6个有效的关节角度
#     print("curr_pose[:6]==",curr_pose)
#     # API版本信息
#     # print(robot.API_Version())
#     # po0 = [18.44, -10.677, -124.158, -15, -71.455, 0]
#     # ret = robot.Movej_Cmd(po0, 30, 0)

#     #   初始位置
#     print("运动到初始位置：, [0, 0, -90, 0, -90, 0]")
#     joint = [0, 0, -90, 0, -90, 0]
#     ret = robot.Movej_Cmd(joint, 20, 0)
#     print("Movej_Cmd ret:" + str(ret))
#     time.sleep(1)

#     # 获取机械臂当前状态
#     ret = robot.Get_Current_Arm_State(retry=1)
#     if ret[0] != 0:
#         print("获取机械臂当前状态失败：" + str(ret))
#         sys.exit()
#     print("获取机械臂当前状态成功，当前关节角度：" + str(ret[1]))
#     print("错误码: " + str(ret[3]) + str(ret[4]))

#     #   正逆解
#     compute_pose = robot.Algo_Forward_Kinematics(joint)
#     print(f'正解结果：{compute_pose}')
#     # compute_pose[0] += 0.
#     res = robot.Algo_Inverse_Kinematics(joint, compute_pose, 1)
#     print(f'逆解：{res}')

#     # 获取当前坐标系
#     retval, frame = robot.Get_Current_Work_Frame(retry=1)
#     if retval == 0:
#         print('当前工作坐标系：', frame.frame_name.name)
#         print('当前工作坐标系位置：', frame.pose.position.x, frame.pose.position.y, frame.pose.position.z)
#     else:
#         print(f'获取当前坐标系失败:{retval}')
#         sys.exit()


#     robot.Change_Work_Frame()
    
#     print("获取当前工作坐标系")
#     retval, frame = robot.Get_Current_Work_Frame()
#     if retval == 0:
#         print('当前工作坐标系：', frame.frame_name.name)
#         print('当前工作坐标系位置：', frame.pose.position.x, frame.pose.position.y, frame.pose.position.z)
#     else:
#         print(f'获取当前坐标系失败:{retval}')
#         sys.exit()
    
#     print("获取当前工具坐标系")
#     retval, frame = robot.Get_Current_Tool_Frame()
#     if retval == 0:
#         print('当前工具坐标系：', frame.frame_name.name)
#         print('当前工具坐标系位置：', frame.pose.position.x, frame.pose.position.y, frame.pose.position.z)
#     else:
#         print(f'获取当前坐标系失败:{retval}')
#         sys.exit()
    
#     #   正逆解
#     compute_pose = robot.Algo_Forward_Kinematics(joint)
#     print(f'正解结果：{compute_pose}')
#     # compute_pose[0] += 0.
#     res = robot.Algo_Inverse_Kinematics(joint, compute_pose, 1)
#     print(f'逆解：{res}')

    
#     print("Get_All_Work_Frame==",robot.Get_All_Work_Frame())# Get_All_Work_Frame== (0, ['World', 'Base', '63bazi'], 3)
#     print("Get_Given_Work_Frame('World')==",robot.Get_Given_Work_Frame('World'))
#     print("Get_Given_Work_Frame('Base')==",robot.Get_Given_Work_Frame('Base'))
#     print("Get_Given_Work_Frame==('63bazi')",robot.Get_Given_Work_Frame('63bazi'))
#     print("Get_Given_Work_Frame==('Arm_Tip')",robot.Get_Given_Work_Frame('Arm_Tip'))
#     # print("获取主动上报接口配置Get_Realtime_Push==",robot.Get_Realtime_Push())
#     # print("设置主动上报接口配置Set_Realtime_Push==",robot.Set_Realtime_Push())
#     # print("机械臂状态主动上报Realtime_Arm_Joint_State==",robot.Realtime_Arm_Joint_State())
#     # # demo3(robot)

#     # 断开连接
#     robot.RM_API_UnInit()
#     robot.Arm_Socket_Close()




    # #  设置工作坐标系
    # retval = robot.Manual_Set_Work_Frame('new', [0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    # if retval != 0:
    #     print(f'设置坐标系失败:{retval}')

    # # 切换当前工作坐标系
    # robot.Change_Work_Frame('new')
    # # 获取当前工作坐标系
    # retval, frame = robot.Get_Current_Work_Frame(retry=1)
    # if retval == 0:
    #     print('当前工作坐标系：', frame.frame_name.name)
    #     print('当前工作坐标系位置：', frame.pose.position.x, frame.pose.position.y, frame.pose.position.z)
    # else:
    #     print(f'获取当前坐标系失败:{retval}')
    #     sys.exit()

    # robot.Change_Work_Frame('World')
    
    # 获取指定坐标系
    # retval, frame = robot.Get_Given_Work_Frame('new', retry=1)
    # if retval == 0:
    #     print('指定工作坐标系(new):', frame)
    # else:
    #     print(f'获取指定坐标系失败:{retval}')
    #     sys.exit()







#     CUR_PATH=os.path.dirname(os.path.realpath(__file__))
#     dllPath=os.path.join(CUR_PATH,"libRM_Base.so")
#     pDll=ctypes.cdll.LoadLibrary(dllPath)

#     pDll.RM_API_Init(65,0)

# #   连接机械臂
#     byteIP = bytes("192.168.1.18","gbk")# 将字符串 “192.168.1.18” 转换为 bytes 类型，使用 gbk 编码。
#     nSocket = pDll.Arm_Socket_Start(byteIP, 8080, 200)# 调用共享库中的 Arm_Socket_Start 函数，传递三个参数：byteIP（IP 地址）、8080（端口号）和 200（超时时间），并返回一个整数 nSocket，可能是用于网络通信的 socket 句柄。
#     print (nSocket)

# #   查询机械臂连接状态
#     nRet = pDll.Arm_Socket_State(nSocket)
#     print(nRet)

# #   设置机械臂末端参数为初始值
#     nRet = pDll.Set_Arm_Tip_Init(nSocket, 1)
#     print(nRet)

# #   设置机械臂动力学碰撞检测等级
#     nRet = pDll.Set_Collision_Stage(nSocket, 0, 1)
#     print(nRet)

# #   机械臂Move_J运动 0-180
#     #   初始位置
#     float_joint = ctypes.c_float*6
#     joint1 = float_joint()
#     joint1[0] = 50
#     joint1[1] = 0
#     joint1[2] = 0
#     joint1[3] = 0
#     joint1[4] = 0
#     joint1[5] = 0
#     pDll.Movej_Cmd.argtypes = (ctypes.c_int, ctypes.c_float * 6, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
#     pDll.Movej_Cmd.restype = ctypes.c_int
#     ret = pDll.Movej_Cmd(nSocket, joint1, 20, 0, 1)

#     i = 1
#     while i < 3:
#         time.sleep(1)
#         i += 1

# #   关闭连接
#     pDll.Arm_Socket_Close(nSocket)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

