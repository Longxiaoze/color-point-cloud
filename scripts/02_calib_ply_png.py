#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_color_ply.py

流程：
 1. 从 YAML(TXT) 文件读取相机内参 K；
 2. 加载一张图像和一个 PLY 点云；
 3. 在图像上交互式点击 N 个对应点（鼠标左键点击，回车结束）；
 4. 在 Open3D 窗口中交互式挑选 N 个对应点（Shift+左键点击，窗口关闭结束）；
 5. 用 solvePnP 计算 LiDAR→Camera 的外参 (R, t)；
 6. 使用算出的外参对整个点云着色并可视化。

依赖：
  pip install open3d pyyaml pillow numpy opencv-python matplotlib
"""

import yaml
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

def load_camera_info(yaml_path):
    with open(yaml_path, 'r') as f:
        cam = list(yaml.safe_load_all(f))[0]
    K = np.array(cam['k'], dtype=float).reshape(3, 3)
    return K

def pick_image_points(img_path, num_points):
    img = np.array(Image.open(img_path))
    plt.figure("Image - pick {} points".format(num_points))
    plt.imshow(img)
    plt.axis('off')
    print(f"请在图像窗口中依次点击 {num_points} 个点，然后回车。")
    pts = plt.ginput(num_points, timeout=0)
    plt.close()
    return np.array(pts, dtype=float)  # shape (N,2)

def pick_pointcloud_points(ply_path, num_points):
    pcd = o3d.io.read_point_cloud(ply_path)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"PointCloud - pick {num_points} points")
    vis.add_geometry(pcd)
    print(f"请在点云窗口中按 Shift+左键 点击 {num_points} 个点，然后关闭窗口。")
    vis.run()  # 关闭窗口后继续
    idx = vis.get_picked_points()
    vis.destroy_window()
    if len(idx) != num_points:
        print(f"[WARN] 您选中了 {len(idx)} 个点，但需要 {num_points} 个。")
    return np.array(idx, dtype=int), np.asarray(pcd.points)

def solve_extrinsic(object_pts, image_pts, K):
    """
    object_pts: (N,3) 3D lidar 点
    image_pts:  (N,2) 像素坐标
    """
    dist_coeffs = np.zeros((4,1))  # 无畸变
    success, rvec, tvec = cv2.solvePnP(
        object_pts, image_pts, K, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP 失败")
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)
    print("[INFO] 计算得到外参：")
    print("R =\n", R)
    print("t =\n", t.flatten())
    return R, t

def colorize_point_cloud(ply_path, img_path, K, R_ex, t_ex):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    # LiDAR -> Camera
    pts_cam = (R_ex @ pts.T + t_ex).T
    proj = (K @ pts_cam.T).T
    u = proj[:,0] / proj[:,2]
    v = proj[:,1] / proj[:,2]
    img = np.array(Image.open(img_path))
    colors = np.zeros_like(pts, dtype=float)
    mask = (proj[:,2] > 1e-6) & (u>=0)&(u<img.shape[1])&(v>=0)&(v<img.shape[0])
    ui = u[mask].astype(int)
    vi = v[mask].astype(int)
    colors[mask] = img[vi, ui] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    if len(sys.argv)!=5:
        print("用法: python3 interactive_color_ply.py <camera_info.txt> <pointcloud.ply> <image.png> <num_points>")
        sys.exit(1)

    yaml_path, ply_path, img_path, n_str = sys.argv[1:]
    if not all(os.path.isfile(p) for p in (yaml_path, ply_path, img_path)):
        print("错误：文件不存在")
        sys.exit(1)
    N = int(n_str)

    # 1. 读取内参
    K = load_camera_info(yaml_path)

    # 2. 图像交互点击
    img_pts = pick_image_points(img_path, N)

    # 3. 点云交互点击
    idxs, pts_world = pick_pointcloud_points(ply_path, N)
    obj_pts = pts_world[idxs]

    # 4. 计算外参
    R_ex, t_ex = solve_extrinsic(obj_pts, img_pts, K)

    # 5. 上色并可视化
    colored_pcd = colorize_point_cloud(ply_path, img_path, K, R_ex, t_ex)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cam_frame   = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    T = np.eye(4)
    T[:3,:3], T[:3,3] = R_ex, t_ex.flatten()
    cam_frame.transform(T)
    o3d.visualization.draw_geometries([colored_pcd, world_frame, cam_frame],
                                      window_name="Colored Point Cloud",
                                      width=1024, height=768)

if __name__ == "__main__":
    main()
