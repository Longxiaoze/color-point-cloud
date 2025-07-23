#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calib_equi_lidar.py

Workflow:
 1. Load image and point cloud;
 2. Interactively click N corresponding points on the image (Shift + left mouse button);
 3. Interactively pick N corresponding points in the Open3D window (Shift + left mouse button);
 4. Convert LiDAR points and image points to spherical coordinates;
 5. Use nonlinear optimization (least_squares) to compute LiDAR→Camera extrinsics (R, t);
 6. Colorize the entire point cloud using the computed extrinsics and visualize;
 7. Save extrinsics as a YAML file;
 8. Visualize point cloud projection onto the image and depth map.

Dependencies:
    pip install numpy opencv-python open3d scipy matplotlib pillow pyyaml
"""

import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
import yaml

# Global variables
image_points = []
point_cloud_points = []
image = None
point_index = 0
def pick_image_points(img_path, num_points):
    global image_points, point_index, image
    image_points = []
    point_index = 0
    image = cv2.imread(img_path)
    orig_image = image.copy()
    scale = 1.0
    win_w, win_h = 1920, 1080
    img_h, img_w = image.shape[:2]
    offset_x, offset_y = 0, 0
    done = [False]

    def show_image():
            nonlocal scale, offset_x, offset_y
            disp_w = min(win_w, int(img_w * scale))
            disp_h = min(win_h, int(img_h * scale))
            x1 = int(offset_x)
            y1 = int(offset_y)
            x2 = min(x1 + disp_w, int(img_w * scale))
            y2 = min(y1 + disp_h, int(img_h * scale))
            img_scaled = cv2.resize(orig_image, (int(img_w * scale), int(img_h * scale)))
            img_disp = img_scaled[y1:y2, x1:x2].copy()
            for (px, py) in image_points:
                    px_s = int(px * scale) - x1
                    py_s = int(py * scale) - y1
                    if 0 <= px_s < img_disp.shape[1] and 0 <= py_s < img_disp.shape[0]:
                            cv2.circle(img_disp, (px_s, py_s), 5, (0, 0, 255), -1)
            cv2.imshow("Image", img_disp)

    def mouse_callback(event, x, y, flags, param):
            nonlocal scale, offset_x, offset_y
            if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
                    px = (x + offset_x) / scale
                    py = (y + offset_y) / scale
                    image_points.append((px, py))
                    print(f"Image point {len(image_points)-1}: ({int(px)}, {int(py)})")
                    show_image()
                    if len(image_points) == num_points:
                            print("All points selected. Press Enter or Space to continue.")
            elif event == cv2.EVENT_MOUSEWHEEL:
                    # Zoom in/out
                    if flags > 0:
                            scale_new = min(scale * 1.2, 10.0)
                    else:
                            scale_new = max(scale / 1.2, 0.1)
                    # Keep mouse position fixed
                    mx, my = x, y
                    ox = (mx + offset_x) / scale
                    oy = (my + offset_y) / scale
                    scale = scale_new
                    offset_x = int(ox * scale - mx)
                    offset_y = int(oy * scale - my)
                    offset_x = max(0, min(offset_x, int(img_w * scale) - win_w))
                    offset_y = max(0, min(offset_y, int(img_h * scale) - win_h))
                    show_image()
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
                    # Pan
                    dx = -param.get('last_x', x) + x
                    dy = -param.get('last_y', y) + y
                    param['last_x'] = x
                    param['last_y'] = y
                    offset_x = min(max(offset_x - dx, 0), max(int(img_w * scale) - win_w, 0))
                    offset_y = min(max(offset_y - dy, 0), max(int(img_h * scale) - win_h, 0))
                    show_image()
            elif event == cv2.EVENT_RBUTTONDOWN:
                    param['last_x'] = x
                    param['last_y'] = y

    print(f"Hold Shift and click {num_points} points on the image. Use mouse wheel to zoom, right mouse button to pan. After selecting, press Enter or Space to continue.")
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", win_w, win_h)
    mouse_param = {}
    cv2.setMouseCallback("Image", mouse_callback, mouse_param)
    show_image()

    while True:
            key = cv2.waitKey(20)
            if len(image_points) == num_points and (key == 13 or key == 32):  # Enter or Space
                    break
            elif key == 27:  # ESC
                    break

    cv2.destroyAllWindows()

    if len(image_points) != num_points:
            raise ValueError(f"You selected {len(image_points)} points, but {num_points} are required.")
    return np.array(image_points, dtype=float)

def pick_pointcloud_points(ply_path, num_points):
    pcd = o3d.io.read_point_cloud(ply_path)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"PointCloud - pick {num_points} points")
    vis.add_geometry(pcd)
    print(f"Please Shift+Left Click {num_points} points in the point cloud window, then close the window.")
    vis.run()  # Continue after window is closed
    idx = vis.get_picked_points()
    vis.destroy_window()
    if len(idx) != num_points:
            raise ValueError(f"You selected {len(idx)} points, but {num_points} are required.")
    return np.array(idx, dtype=int), np.asarray(pcd.points)

def point_to_spherical(point):
    x, y, z = point
    r = np.linalg.norm([x, y, z])
    theta = np.arctan2(y, x)  # Azimuth
    phi = np.arcsin(z / r)    # Elevation
    return theta, phi

def spherical_to_pixel(theta, phi, width, height):
    u = (theta + np.pi) / (2 * np.pi) * width
    v = (phi + np.pi / 2) / np.pi * height
    return int(u), int(v)

def pixel_to_spherical(u, v, width, height):
    theta = (u / width) * 2 * np.pi - np.pi
    phi = (v / height) * np.pi - np.pi / 2
    return theta, phi

def cost_function(params, lidar_points, image_points, width, height):
    # params: [tx, ty, tz, rx, ry, rz] → extrinsics
    T = np.eye(4)
    T[:3, 3] = params[:3]
    T[:3, :3] = R_scipy.from_rotvec(params[3:]).as_matrix()

    residuals = []
    for lidar_point, (u, v) in zip(lidar_points, image_points):
            # LiDAR → Camera coordinates
            point_cam = T @ np.append(lidar_point, 1)
            x, y, z = point_cam[:3]
            r = np.linalg.norm([x, y, z])
            theta = np.arctan2(y, x)
            phi = np.arcsin(z / r)

            # Image coordinates
            u_proj = (theta + np.pi) / (2 * np.pi) * width
            v_proj = (phi + np.pi / 2) / np.pi * height

            residuals.append(u - u_proj)
            residuals.append(v - v_proj)

    return np.array(residuals)

def save_calibration_yaml(R, t, output_path="calibration.yaml"):
    quat = R_scipy.from_matrix(R).as_quat()

    data = {
            "transform": {
                    "translation": {
                            "x": float(t[0]),
                            "y": float(t[1]),
                            "z": float(t[2]),
                    },
                    "rotation": {
                            "x": float(quat[0]),
                            "y": float(quat[1]),
                            "z": float(quat[2]),
                            "w": float(quat[3]),
                    }
            }
    }

    with open(output_path, 'w') as f:
            yaml.dump(data, f)
    print(f"Extrinsics saved to {output_path}")

def colorize_point_cloud(pcd, image_path, T):
    pts = np.asarray(pcd.points)
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_cam = (T[:3, :3] @ pts.T + T[:3, 3:]).T

    img = np.array(Image.open(image_path))
    width, height = img.shape[1], img.shape[0]
    colors = np.zeros_like(pts, dtype=np.float64)

    for i, (x, y, z) in enumerate(pts_cam):
            r = np.linalg.norm([x, y, z])
            theta = np.arctan2(y, x)
            phi = np.arcsin(z / r)
            u = (theta + np.pi) / (2 * np.pi) * width
            v = (phi + np.pi / 2) / np.pi * height

            if 0 <= u < width and 0 <= v < height:
                    colors[i] = img[int(v), int(u)] / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_projection(image_path, pcd, T):
    img = cv2.imread(image_path)
    win_w, win_h = 1920, 1080
    width, height = img.shape[1], img.shape[0]
    disp_img = img.copy()
    pts = np.asarray(pcd.points)
    pts_cam = (T[:3, :3] @ pts.T + T[:3, 3:]).T

    for x, y, z in pts_cam:
            r = np.linalg.norm([x, y, z])
            if r < 1e-6:
                    continue
            theta = np.arctan2(y, x)
            phi = np.arcsin(z / r)
            u = (theta + np.pi) / (2 * np.pi) * width
            v = (phi + np.pi / 2) / np.pi * height

            if 0 <= u < width and 0 <= v < height:
                    cv2.circle(disp_img, (int(u), int(v)), 1, (0, 255, 0), -1)
    cv2.namedWindow("Projected Points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Projected Points", win_w, win_h)
    cv2.imshow("Projected Points", disp_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Projected Points")

def visualize_depth_map(pcd, T):
    win_w, win_h = 1920, 1080
    pts = np.asarray(pcd.points)
    pts_cam = (T[:3, :3] @ pts.T + T[:3, 3:]).T

    # 设定全景图尺寸
    pano_w, pano_h = win_w, win_h
    depth_map = np.zeros((pano_h, pano_w), dtype=np.float32)
    count_map = np.zeros((pano_h, pano_w), dtype=np.int32)

    for x, y, z in pts_cam:
        r = np.linalg.norm([x, y, z])
        if r < 1e-6:
            continue
        theta = np.arctan2(y, x)
        phi = np.arcsin(z / r)
        u = int((theta + np.pi) / (2 * np.pi) * pano_w)
        v = int((phi + np.pi / 2) / np.pi * pano_h)
        if 0 <= u < pano_w and 0 <= v < pano_h:
            if depth_map[v, u] == 0 or r < depth_map[v, u]:
                depth_map[v, u] = r
            count_map[v, u] += 1

    # 填补未赋值像素
    mask = (depth_map == 0)
    if np.any(~mask):
        min_depth = np.min(depth_map[~mask])
        depth_map[mask] = min_depth

    # 归一化到0-255
    depth_norm = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8) * 255
    depth_img = depth_norm.astype(np.uint8)
    depth_img_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

    cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Map", pano_w, pano_h)
    cv2.imshow("Depth Map", depth_img_color)
    cv2.waitKey(0)
    cv2.destroyWindow("Depth Map")

def load_calibration_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    t = data['transform']['translation']
    q = data['transform']['rotation']
    trans = np.array([t['x'], t['y'], t['z']])
    quat = np.array([q['x'], q['y'], q['z'], q['w']])
    R_mat = R_scipy.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = trans
    return T

def main():
    if len(sys.argv) != 5:
            print("Usage: python3 calib_equi_lidar.py <pointcloud.ply> <image.png> <num_points> <output_yaml>")
            sys.exit(1)

    ply_path, img_path, n_str, output_yaml = sys.argv[1:]
    if not all(os.path.isfile(p) for p in (ply_path, img_path)):
            print("Error: File does not exist")
            sys.exit(1)
    N = int(n_str)

    if os.path.isfile(output_yaml):
            print(f"Detected existing extrinsics file {output_yaml}, loading and rendering point cloud directly.")
            T = load_calibration_yaml(output_yaml)
    else:
            # 1. Interactive image click
            img_pts = pick_image_points(img_path, N)

            # 2. Interactive point cloud click
            idxs, pts_world = pick_pointcloud_points(ply_path, N)
            obj_pts = pts_world[idxs]

            # 3. Convert image points to spherical coordinates
            width, height = Image.open(img_path).size
            image_spherical = [pixel_to_spherical(u, v, width, height) for u, v in img_pts]
            lidar_spherical = [point_to_spherical(p) for p in obj_pts]

            # 4. Nonlinear optimization for extrinsics
            initial_guess = [0.5, 0.0, 1.5, 0, 0, 0]  # tx, ty, tz, rx, ry, rz
            result = least_squares(lambda x: cost_function(x, obj_pts, img_pts, width, height), initial_guess)
            T = np.eye(4)
            T[:3, 3] = result.x[:3]
            T[:3, :3] = R_scipy.from_rotvec(result.x[3:]).as_matrix()

            # 5. Save extrinsics
            save_calibration_yaml(T[:3, :3], T[:3, 3], output_yaml)

    # 6. Colorize point cloud
    print("Colorizing point cloud...")
    pcd = o3d.io.read_point_cloud(ply_path)
    colored_pcd = colorize_point_cloud(pcd, img_path, T)

    # 7. Visualize point cloud
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    cam_frame.transform(T)
    o3d.visualization.draw_geometries([colored_pcd, world_frame, cam_frame],
                                                                        window_name="Colored Point Cloud",
                                                                        width=1024, height=768)

    # 8. Visualize point cloud projection onto image
    print("Visualizing point cloud projection onto image...")
    print("Press Esc to close the image window.")
    visualize_projection(img_path, colored_pcd, T)

    # 9. Visualize depth map
    print("Visualizing point cloud depth map...")
    print("Press Esc to close the image window.")
    visualize_depth_map(colored_pcd, T)


if __name__ == "__main__":
    main()