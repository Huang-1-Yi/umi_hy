import pyrealsense2 as rs
import open3d as o3d
import numpy as np

def extract_corner_points(cloud, curvature_threshold=0.5):
    # 计算点云的法线和曲率
    pcd = cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
    
    # 使用法线和曲率信息进行角点检测
    curvatures = []
    for i in range(len(pcd.points)):
        normal = np.asarray(pcd.normals)[i]
        curvature = np.linalg.norm(normal)  # 这里只是一个简单的示例，实际中可以通过计算法线变化来获得曲率
        curvatures.append(curvature)
    
    curvatures = np.array(curvatures)
    
    # 提取曲率高于阈值的点作为角点
    corner_points = np.asarray(pcd.points)[curvatures > curvature_threshold]
    
    # 将角点转换为PointCloud对象
    corner_pcd = o3d.geometry.PointCloud()
    corner_pcd.points = o3d.utility.Vector3dVector(corner_points)
    
    return corner_pcd

# 配置 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        # 获取一帧点云数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        
        # 将深度帧转换为点云
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
        
        # 提取角点
        corner_pcd = extract_corner_points(pcd)
        
        # 实时可视化
        o3d.visualization.draw_geometries([pcd, corner_pcd], window_name="Real-time Point Cloud", width=800, height=600)

        # 询问用户是否保存点云或退出
        user_input = input("输入 'y' 保存当前点云，输入 'n' 退出: ")
        if user_input.lower() == 'y':
            frame_id = depth_frame.get_frame_number()
            o3d.io.write_point_cloud(f"frame_{frame_id}.ply", pcd)
            print(f"已保存点云为 frame_{frame_id}.ply")
        elif user_input.lower() == 'n':
            break

except KeyboardInterrupt:
    print("结束实时点云读取")

finally:
    pipeline.stop()