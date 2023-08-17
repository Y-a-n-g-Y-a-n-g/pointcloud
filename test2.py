# -*- coding: utf-8 -*-
import requests
import time
import cupoch as cph

class PointCloudFusion():
    def __init__(self,url="http://192.168.31.122:6006/merge-pointclouds"):
        self.url=url
    def gettransformation(self,source_file_path):
        source_file_path = source_file_path
        with open(source_file_path, 'rb') as source_file:
            files = {'source': source_file}
            response = requests.post(self.url, files=files, timeout=500)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print("Error:", response.status_code, response.text)
pointcloudfusion=PointCloudFusion()



for i in range(1,13):
    t0 = time.time()
    result = pointcloudfusion.gettransformation(f'pointdata/pointcloud{i}.pcd')
    print(result)
    if result['fitness']>0.8:
        print("融合成功")
        transformation=result['transformation']
        source = cph.io.read_point_cloud(f'pointdata/pointcloud{i}.pcd').voxel_down_sample(voxel_size=0.0015)
        target = cph.io.read_point_cloud('combine.pcd').voxel_down_sample(voxel_size=0.0015)
        source=source.transform(transformation)
        cp=target+source
        cp = cp.voxel_down_sample(voxel_size=0.0015)
        cph.io.write_point_cloud("combine.pcd", cp)
    else:
        print("融合失败")
    t1 = time.time()
    print(t1 - t0)
