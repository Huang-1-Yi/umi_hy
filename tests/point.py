# v1
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

# 读取点云
cloud = o3d.io.read_point_cloud("path_to_your_pointcloud.pcd")  # 替换为你的PCD文件路径

# 提取角点
corner_pcd = extract_corner_points(cloud)

# 可视化结果
o3d.visualization.draw_geometries([cloud, corner_pcd])
