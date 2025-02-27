# 对应的加载方法
import open3d as o3d

pcd = o3d.io.read_point_cloud("output_pointcloud.pcd")
o3d.visualization.draw_geometries([pcd])