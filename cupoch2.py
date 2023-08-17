# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os
import time
import cupoch as cph
firsttime=True
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
def fusiontwopointcloud(source, target,):
    t0=time.time()
    # 加载PCD文件
    source = cph.io.read_point_cloud(source).voxel_down_sample(voxel_size=0.0015)
    target = cph.io.read_point_cloud(target).voxel_down_sample(voxel_size=0.0015)
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
                                                cph.registration.ICPConvergenceCriteria(max_iteration=5000))
    t1 = time.time()
    source = source.transform(reg_p2p.transformation)
    cp= target + source
    cp = cp.voxel_down_sample(voxel_size=0.0015)
    cph.io.write_point_cloud("history.pcd", cp)
    return {"fitness":reg_p2p.fitness,"inlier_rmse":reg_p2p.inlier_rmse,"transformation":reg_p2p.transformation.tolist(),"time":round(t1-t0,2)} # 返回4x4的矩阵

@app.route('/merge-pointclouds', methods=['POST'])
def merge_pointclouds():
    # Check if the post request has the file part
    if 'source' not in request.files :
        return 'No file part', 400

    source_file = request.files['source']

    # Check if user does not select file
    if source_file.filename == '' :
        return 'No selected file', 400

    # Save the uploaded files
    source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_file.filename)


    global firsttime
    if firsttime:
        firsttime=False
        source_file.save("history.pcd")
        return jsonify({"fitness":0,"inlier_rmse":1,"transformation":[],"time":0})
    source_file.save(source_path)
    # Process and merge the point clouds
    result = fusiontwopointcloud(source_path,'history.pcd')
    # 返回4x4的矩阵作为响应
    return jsonify(result)

if __name__ == "__main__":
    while True:  # 死循环
        try:
            app.run(debug=True, host="192.168.31.122", port=6006)
        except Exception as e:
            print(f"Server crashed due to {e}. Restarting...")
            time.sleep(2)  # 等待5秒钟再次启动，以避免立即重新启动