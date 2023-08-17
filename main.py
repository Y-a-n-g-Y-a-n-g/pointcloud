import time  # 导入时间模块用于计算执行时间
import numpy as np  # 导入numpy库进行数值操作
import open3d as o3d  # 导入open3d库进行点云操作
device = o3d.core.Device('CUDA:0' if o3d.core.cuda.is_available() else 'CPU:0')
print("Device:", device)

# 函数用于深度复制点云
def copy_point_cloud(pcd):
    copied_pcd = o3d.geometry.PointCloud()  # 创建一个空的点云对象
    # 从输入点云复制点到新的点云
    copied_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points))
    # 从输入点云复制颜色到新的点云
    copied_pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors))
    return copied_pcd  # 返回深度复制的点云




# 函数用于可视化两个点云的对齐
def draw_registration_result(source, target, transformation):
    source_temp = copy_point_cloud(source)  # 深度复制源点云
    target_temp = copy_point_cloud(target)  # 深度复制目标点云
    source_temp.transform(transformation)  # 使用提供的转换矩阵转换源点云
    # 一起可视化转换后的源点云和目标点云
    o3d.visualization.draw_geometries([source_temp, target_temp])
def preprocess_point_cloud(pcd, voxel_size):  # 函数用于预处理点云
    pcd_down = pcd.voxel_down_sample(voxel_size)  # 使用体素网格下采样点云
    radius_normal = voxel_size * 2  # 定义法线估计的搜索半径
    # 为下采样的点云估计法线
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5  # 定义FPFH特征计算的搜索半径
    # 计算下采样点云的FPFH特征
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh  # 返回下采样的点云和其FPFH特征
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):  # 函数执行全局注册
    distance_threshold = voxel_size * 1.5  # 定义对应的距离阈值
    # 使用FPFH特征执行基于RANSAC的注册
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=True,  # 使用相互过滤器进行对应过滤
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        # 为RANSAC算法定义对应检查器
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result  # 返回注册结果

# 函数使用ICP精细调整
def refine_registration(source, target, voxel_size, global_transformation):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_icp(source,
                                                         target,
                                                         distance_threshold,
                                                         global_transformation,
                                                         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                         o3d.pipelines.registration.ICPConvergenceCriteria(
                                                             max_iteration=5000))
    return result

def save_registration_result(source, target, transformation, filename):  # 函数将注册结果保存为点云文件
    # 使用提供的转换矩阵转换源点云
    transformed_source = source.transform(transformation)
    combined_pcd = target + transformed_source  # 合并转换后的源和目标点云
    voxel_size = 0.0015  # 体素大小为1厘米
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(filename, combined_pcd)  # 将组合的点云保存到文件
    print(f"保存已注册的点云到 {filename}")  # 打印保存位置
# 主函数将两个点云融合并保存结果
def fusiontwopointcloud(source, target, save, initial_transformation=None):
    t0 = time.time()  # 记录开始时间
    source = o3d.io.read_point_cloud(source)  # 读取源点云文件
    target = o3d.io.read_point_cloud(target)  # 读取目标点云文件
    #print("加载点云文件所需时间:", time.time() - t0, "秒")

    voxel_size = 0.0015  # 定义下采样的体素大小
    # 预处理源和目标点云
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    #print("预处理文件所需时间:", time.time() - t0, "秒")

    # 在源和目标点云之间执行全局注册
    global_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    #print("全局注册转换:",time.time()-t0)  # 打印来自全局注册的转换矩阵

    #print(global_result.transformation)
    #print("global_result所需时间:", time.time() - t0, "秒")
    # 使用ICP精细调整注册
    icp_result = refine_registration(source_down, target_down, voxel_size, global_result.transformation)
    #print("ICP细化转换:")  # 打印来自ICP细化的转换矩阵
   # print(icp_result.transformation)
    #print("ICP_result所需时间:", time.time() - t0, "秒")

    transformation = icp_result.transformation  # 从ICP结果中提取转换矩阵
    save_registration_result(source_down, target_down, transformation, save)  # 保存注册结果
    t1 = time.time()  # 记录结束时间
    return round(t1 - t0, 2)  # 返回经过的时间


t1 = fusiontwopointcloud("combine.pcd", "pointcloud2.pcd", "p12.pcd")
t2=fusiontwopointcloud("p12.pcd","pointcloud3.pcd","p123.pcd",)
t3=fusiontwopointcloud("p123.pcd","pointcloud4.pcd","p1234.pcd",)
t4=fusiontwopointcloud("p1234.pcd","pointcloud5.pcd","p12345.pcd",)
t5=fusiontwopointcloud("p12345.pcd","pointcloud6.pcd","p123456.pcd")
t6=fusiontwopointcloud("p123456.pcd","pointcloud7.pcd","p1234567.pcd")
t7=fusiontwopointcloud("p1234567.pcd","pointcloud8.pcd","p12345678.pcd")
t8=fusiontwopointcloud("p12345678.pcd","pointcloud9.pcd","p123456789.pcd")
t9=fusiontwopointcloud("p123456789.pcd","pointcloud10.pcd","p12345678910.pcd")
t10=fusiontwopointcloud("p12345678910.pcd","pointcloud11.pcd","p1234567891011.pcd")
t11=fusiontwopointcloud("p1234567891011.pcd","pointcloud12.pcd","p123456789101112.pcd")
t12=fusiontwopointcloud("p123456789101112.pcd","pointcloud13.pcd","p12345678910111213.pcd")
print('pointcloud1: ',t1)
print('pointcloud2: ',t2)
print('pointcloud3: ',t3)
print('pointcloud4: ',t4)
print('pointcloud5: ',t5)
print('pointcloud6: ',t6)
print('pointcloud7: ',t7)
print('pointcloud8: ',t8)
print('pointcloud9: ',t9)
print('pointcloud10: ',t10)
print('pointcloud11: ',t11)
print('pointcloud12: ',t12)
print('total: ',t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12)
