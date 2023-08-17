import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import struct
def callback(msg):
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
    o3d.io.write_point_cloud("./pointcloud.pcd", pcd)
def main():
    rospy.init_node('pointcloud_listener', anonymous=True)
    rospy.Subscriber("/camera/depth/color/points2", PointCloud2, callback)
    rospy.spin()
if __name__ == '__main__':
    main()