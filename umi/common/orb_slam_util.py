import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# 加载不同格式（TUM和CSV）的轨迹数据

# 加载TUM格式的轨迹数据，该格式通常用于评估单目SLAM算法的性能
def load_tum_trajectory(tum_txt_path):
    # 从指定路径加载数据，假设数据以空格分隔，并且是浮点数类型
    tum_traj_raw = np.loadtxt(tum_txt_path, delimiter=' ', dtype=np.float32)
    # 数据为空（长度为0），则返回一个空字典
    if len(tum_traj_raw) == 0:
        return {
            'timestamp': np.array([]),
            'pose': np.array([]),
        }
    # 从加载的数据中提取时间戳（timestamp_sec）、相机位置（cam_pos）和相机旋转（cam_rot_quat_xyzw，以四元数形式）
    timestamp_sec = tum_traj_raw[:,0]
    cam_pos = tum_traj_raw[:,1:4]
    cam_rot_quat_xyzw = tum_traj_raw[:,4:8]
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw) # 将四元数旋转转换为旋转矩阵
    # 位置和旋转信息
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()

    result = {
        'timestamp': timestamp_sec,
        'pose': cam_pose
    }
    return result   # 返回一个字典，其中包含时间戳和相机姿态矩阵

# 加载CSV格式的轨迹数据，存储表格数据
def load_csv_trajectory(csv_path):
    # 从指定路径加载CSV数据，并将数据加载到一个DataFrame（df）中
    df = pd.read_csv(csv_path)
    # 如果所有数据点都被标记为丢失（通过df.is_lost列），则返回一个包含原始DataFrame的字典
    if (~df.is_lost).sum() == 0:
        return {
            'raw_data': df
        }
    # 否则，它从DataFrame中提取有效的数据点，即那些没有被标记为丢失的数据点
    valid_df = df.loc[~df.is_lost]
    
    # 提取时间戳、相机位置和相机旋转，并将它们转换为NumPy数组
    timestamp_sec = valid_df['timestamp'].to_numpy()
    cam_pos = valid_df[['x', 'y', 'z']].to_numpy()
    cam_rot_quat_xyzw = valid_df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)

    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()

    result = {
        'timestamp': timestamp_sec,
        'pose': cam_pose,
        'raw_data': df
    }
    return result   # 返回一个字典，其中包含时间戳、相机姿态矩阵和原始DataFrame
