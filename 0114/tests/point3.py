# 保存点云

import pyrealsense2 as rs
import open3d as o3d
import numpy as np

def save_point_cloud_from_realsense(output_path):
    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    try:
        print("正在捕获点云数据，请稍候...")
        
        # 获取一帧点云数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("未捕获到深度帧！")

        # 将深度帧转换为点云
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
        
        # 保存点云为 PCD 文件
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"点云数据已保存到：{output_path}")

    finally:
        pipeline.stop()
        print("RealSense 管道已关闭。")

# 设置输出路径
output_path = "output_pointcloud.pcd"  # 替换为你希望保存的文件路径

# 保存点云数据
save_point_cloud_from_realsense(output_path)



