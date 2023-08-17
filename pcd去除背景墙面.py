import time
import numpy as np
import open3d as o3d  # 导入open3d库进行点云操作
def remove_background_plane(pcd, plane_distance_threshold=0.00005, point_distance_threshold=0.005):
    """
    使用RANSAC从点云中分割背景平面，并根据给定的平面模型过滤掉靠近该平面的点。
    返回去除背景后的点云。
    """
    # 使用RANSAC分割点云中的平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=plane_distance_threshold,
                                             ransac_n=3,
                                             num_iterations=10000)
    non_plane = pcd.select_by_index(inliers, invert=True)
    # 使用平面模型过滤掉靠近平面的点
    a, b, c, d = plane_model
    distances = np.abs((np.dot(np.asarray(non_plane.points),
                               np.array([a, b, c])) + d) / np.linalg.norm([a, b, c]))
    mask = distances > point_distance_threshold
    filtered_pcd = non_plane.select_by_index(np.where(mask)[0])
    return filtered_pcd
pc = o3d.io.read_point_cloud(f"target0.pcd")  # 读取源点云文件
pc = remove_background_plane(pc)  # 去除背景平面
o3d.io.write_point_cloud(f"target0.pcd", pc)  # 将组合的点云保存到文件
# t0=time.time()
# for i in range(1,14):
#     pc = o3d.io.read_point_cloud(f"pointcloud{i}.pcd")  # 读取源点云文件
#     pc = remove_background_plane(pc)  # 去除背景平面
#     o3d.io.write_point_cloud(f"pointcloud{i}.pcd", pc)  # 将组合的点云保存到文件
# t1=time.time()
# print(t1-t0)

