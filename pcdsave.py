import requests
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import struct
doingflag=False
fisttime=True
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
class PointCloudFusion():
    def __init__(self,url="http://192.168.31.122:6006/merge-pointclouds"):
        self.url=url
    def gettransformation(self,source_file_path,target_file_path):
        source_file_path = source_file_path
        target_file_path = target_file_path
        with open(source_file_path, 'rb') as source_file, open(target_file_path, 'rb') as target_file:
            files = {'source': source_file,'target': target_file}
            response = requests.post(self.url, files=files, timeout=100)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print("Error:", response.status_code, response.text)
pointcloudfusion=PointCloudFusion()
def callback(msg):
    global doingflag,fisttime
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

    if fisttime:
        pcd.voxel_down_sample(0.0015)
        #pcd = remove_background_plane(pcd)
        o3d.io.write_point_cloud("./combine.pcd", pcd)
        fisttime=False
    else:
        pcd.voxel_down_sample(0.0015)
        #pcd = remove_background_plane(pcd)
        o3d.io.write_point_cloud("./new.pcd", pcd)
        result = pointcloudfusion.gettransformation(f'new.pcd', 'combine.pcd')
        print(result)
        if result['fitness'] > 0.9:
            print("融合成功")
            transformation = result['transformation']
            source = o3d.io.read_point_cloud(f'new.pcd').voxel_down_sample(voxel_size=0.0015)
            target = o3d.io.read_point_cloud('combine.pcd').voxel_down_sample(voxel_size=0.0015)
            source = source.transform(transformation)
            cp = target + source
            cp = cp.voxel_down_sample(voxel_size=0.0015)
            o3d.io.write_point_cloud("combine.pcd", cp)
        else:
            print("融合失败")
    doingflag=False
def main():
    rospy.init_node('pointcloud_listener', anonymous=True)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, callback)
    rospy.spin()
if __name__ == '__main__':
    main()