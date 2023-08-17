# import time
#
# import open3d as o3d  # 导入open3d库进行点云操作
# t0=time.time()
#
# for i in range(1,14):
#     pc = o3d.io.read_point_cloud(f"pointcloud{i}.pcd")
#     voxel_size = 0.0015
#     pc = pc.voxel_down_sample(voxel_size)
#     o3d.io.write_point_cloud(f"pointcloud{i}.pcd", pc)
# t1=time.time()
#
#
# print(t1-t0)


import cupoch as cph
import time
t0=time.time()
for i in range(1,14):
    pc = cph.io.read_point_cloud(f"pointcloud{i}.pcd")
    voxel_size = 0.0015
    pc = pc.voxel_down_sample(voxel_size)
    cph.io.write_point_cloud(f"pointcloud{i}.pcd", pc)
t1=time.time()
print(t1-t0)