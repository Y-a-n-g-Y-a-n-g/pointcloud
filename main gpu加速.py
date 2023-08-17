import time
import cupoch as cph
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
def fusiontwopointcloud(source, target, save,):
    t0=time.time()
    # 加载PCD文件
    source = cph.io.read_point_cloud(source)
    target = cph.io.read_point_cloud(target)
    source2 = source.voxel_down_sample(voxel_size=0.0015)
    target2 = target.voxel_down_sample(voxel_size=0.0015)
    # 定义ICP参数
    threshold = 0.01  # 两个点之间的最大对应距离
    source_down, source_fpfh = preprocess_point_cloud(source2, voxel_size=0.015)
    target_down, target_fpfh = preprocess_point_cloud(target2, voxel_size=0.015)
    result_fast=execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size=0.0015)
    trans_init = result_fast.transformation
    reg_p2p = cph.registration.registration_icp(source2,
                                                target2,
                                                threshold,
                                                trans_init,
                                                cph.registration.TransformationEstimationPointToPoint(),
                                                cph.registration.ICPConvergenceCriteria(max_iteration=10000))
    print(reg_p2p)
    source2=source.transform(reg_p2p.transformation)
    cb=source2+target2
    cb = cb.voxel_down_sample(voxel_size=0.0015)
    cph.io.write_point_cloud(save, cb)
    return time.time()-t0
t1 = fusiontwopointcloud("./pointdata/pointcloud1.pcd", "./pointdata/pointcloud2.pcd", "p12.pcd")
t2=fusiontwopointcloud("p12.pcd","./pointdata/pointcloud3.pcd","p123.pcd",)
t3=fusiontwopointcloud("p123.pcd","./pointdata/pointcloud4.pcd","p1234.pcd",)
t4=fusiontwopointcloud("p1234.pcd","./pointdata/pointcloud5.pcd","p12345.pcd",)
t5=fusiontwopointcloud("p12345.pcd","./pointdata/pointcloud6.pcd","p123456.pcd")
t6=fusiontwopointcloud("p123456.pcd","./pointdata/pointcloud7.pcd","p1234567.pcd")
t7=fusiontwopointcloud("p1234567.pcd","./pointdata/pointcloud8.pcd","p12345678.pcd")
t8=fusiontwopointcloud("p12345678.pcd","./pointdata/pointcloud9.pcd","p123456789.pcd")
t9=fusiontwopointcloud("p123456789.pcd","./pointdata/pointcloud10.pcd","p12345678910.pcd")
t10=fusiontwopointcloud("p12345678910.pcd","./pointdata/pointcloud11.pcd","p1234567891011.pcd")
t11=fusiontwopointcloud("p1234567891011.pcd","./pointdata/pointcloud12.pcd","p123456789101112.pcd")
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
print('total: ',t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11)