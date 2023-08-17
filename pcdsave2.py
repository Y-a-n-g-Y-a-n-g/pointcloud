import requests
import rospy
from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import struct
import time
import cupoch as cph
doingflag=False
fisttime=True
cbpointcloud=None
header = Header()
header.frame_id = 'camera_depth_optical_frame'
targetflag=0
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(cph.geometry.KDTreeSearchParamRadius(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_feature = cph.registration.compute_shot_feature(pcd_down, radius_feature, cph.geometry.KDTreeSearchParamRadius(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_feature
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = cph.registration.registration_fast_based_on_feature_matching(source_down,target_down,source_fpfh,target_fpfh,
                                                                          cph.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold),)
    return result
def fusiontwopointcloud():
    global  targetflag
    source = cph.io.read_point_cloud("source.pcd").voxel_down_sample(voxel_size=0.0015)
    target = cph.io.read_point_cloud("target.pcd").voxel_down_sample(voxel_size=0.0015)
    source = source.voxel_down_sample(voxel_size=0.0015)
    target = target.voxel_down_sample(voxel_size=0.0015)
    # 定义ICP参数
    threshold = 0.01  # 两个点之间的最大对应距离
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size=0.015)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=0.015)
    result_fast=execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size=0.0015)
    trans_init = result_fast.transformation
    reg_p2p = cph.registration.registration_icp(source,
                                                target,
                                                threshold,
                                                trans_init,
                                                cph.registration.TransformationEstimationPointToPoint(),
                                                cph.registration.ICPConvergenceCriteria(max_iteration=10000))
    if reg_p2p.fitness < 0.8:
        print(f"ICP registration failed, fitness score : {reg_p2p.fitness:.3f}")
        targetflag += 1
        cph.io.write_point_cloud(f"target{targetflag}.pcd", target)
        return target
    else:
        print(f"ICP registration successful, fitness score : {reg_p2p.fitness:.3f}")
        source = source.transform(reg_p2p.transformation)
        cb = source + target
        cb = cb.voxel_down_sample(voxel_size=0.0015)
        targetflag+=1
        cph.io.write_point_cloud(f"target{targetflag}.pcd", cb)
        return cb
def rgb_to_float_color(r, g, b):
    """Converts 8-bit RGB values to the packed float format used by PointCloud2."""
    rgb = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack('f', struct.pack('I', rgb))[0]
def callback(msg):
    print("handling pointcloud")
    global doingflag,fisttime,cbpointcloud,targetflag
    if doingflag:
        return
    doingflag=True
    # 读取点云数据
    cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
    # 解析点云的坐标和颜色
    xyz = [(x, y, z) for x, y, z, rgb in cloud_points]
    # 解析 RGB 颜色
    rgb = [struct.unpack('I', struct.pack('f', rgb))[0] for x, y, z, rgb in cloud_points]
    rgb_colors = [(rgb >> 16 & 0x0000ff, rgb >> 8 & 0x0000ff, rgb & 0x0000ff) for rgb in rgb]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    # Create a color array from the rgb_colors and normalize to [0,1]
    colors = np.array(rgb_colors) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Save the point cloud in PCD format
    o3d.io.write_point_cloud("source.pcd", pcd)

    if fisttime:
        pcd.voxel_down_sample(0.0015)
        o3d.io.write_point_cloud(f"target.pcd", pcd)
        fisttime=False
    else:
        pcd.voxel_down_sample(0.0015)
        cbpointcloud = fusiontwopointcloud()
        cph.io.write_point_cloud(f"target.pcd", cbpointcloud)
    # pcd = o3d.io.read_point_cloud("target.pcd")
    # # 提取点的位置和颜色信息
    # points = np.asarray(pcd.points)
    # colors = np.asarray(pcd.colors)
    # # 确保颜色在 [0, 1] 范围内
    # colors = colors / 255.0
    # rgb_floats = [rgb_to_float_color(r, g, b) for r, g, b in colors]
    # header.stamp = rospy.Time.now()
    # fields = [PointField('x', 0, PointField.FLOAT32, 1),
    #           PointField('y', 4, PointField.FLOAT32, 1),
    #           PointField('z', 8, PointField.FLOAT32, 1),
    #           PointField('rgb', 12, PointField.FLOAT32, 1)]
    # combined_points = [(x, y, z, rgb) for (x, y, z), rgb in zip(points, rgb_floats)]
    #
    #
    # pointcloud2_msg = pc2.create_cloud(header,
    #                                    fields=fields,
    #                                    points=combined_points)
    # pointcloud_publisher.publish(pointcloud2_msg)
    doingflag=False
def main():
    rospy.init_node('pointcloud_listener', anonymous=True)
    global pointcloud_publisher
    pointcloud_publisher = rospy.Publisher("/combine_pointcloud", PointCloud2, queue_size=10)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, callback)
    rospy.spin()
if __name__ == '__main__':
    main()