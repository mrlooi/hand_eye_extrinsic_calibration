import open3d
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
# import apriltag
import cv2 
import json
import os.path as osp

from ext_calib_optimizer import ping_pong_optimize

from utils_3d import filter_cloud_nan, estimate_tag_pose, extract_image_and_points
from detect_pattern import detect_circle_pattern_corners

o3d = open3d


def read_ee_poses(file_name):
    with open(file_name, 'r') as f:
        wf_to_ee = json.load(f)

    poses_data = wf_to_ee["world_frame_to_ee_tip_0_tfs"]
    ee_poses = {}
    for key, value in poses_data.items():
        data = poses_data[key]
        t = np.array([data['x_pos'], data['y_pos'], data['z_pos']], dtype=float)
        q = np.array([data['w_rot'], data['x_rot'], data['y_rot'], data['z_rot']], dtype=float)
        T = np.eye(4)
        T[:3,:3] = quat2mat(q)
        T[:3, 3] = t
        ee_poses[key] = T

    return ee_poses


def read_point_cloud(pcd_file, point_multiplier=1.0):
    if not osp.exists(pcd_file):
        return False, None

    cloud = o3d.read_point_cloud_with_nan(pcd_file)
    if cloud.is_empty():
        return False, None

    if point_multiplier is not None and point_multiplier != 1.0:
        cloud.points = o3d.Vector3dVector(np.asarray(cloud.points) * point_multiplier)

    return True, cloud


if __name__ == '__main__':
    import os, os.path as osp

    inv = np.linalg.inv

    """
    INPUT START
    """
    COLOR_HEIGHT = 480
    COLOR_WIDTH = 640
    point_multiplier = 1. / 1000 # mm to m # IMPORTANT!! Points MUST BE NORMALIZED TO METRES. IF ALREADY IN METRES (e.g. Kinect), set to 1.0

    # COLOR_HEIGHT = 540
    # COLOR_WIDTH = 960
    ee_pose_file = "./data/asymm_circle/world_frame_to_ee_tip_0_tfs.json"

    file_ids = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    pcd_files = ['./data/asymm_circle/pcd_files/%s.pcd'%(id) for id in file_ids]

    VIS_BOARD_DETECTION = False
    CALIB_BOARD_DIMS = (4, 11)
    """
    INPUT END
    """

    ee_poses = read_ee_poses(ee_pose_file)
    COLOR_DIMS = COLOR_HEIGHT * COLOR_WIDTH

    tag_poses = {}
    for i, id in enumerate(file_ids):
        if id not in ee_poses:
            print("File '%s' not in ee_poses, skipping..."%(id))
            continue

        pcd_file = pcd_files[i]

        rt, cloud = read_point_cloud(pcd_file, point_multiplier=point_multiplier)  # cm to m
        if not rt:
            print("File '%s': Could not read pcd '%s', skipping..."%(id, pcd_file))
            continue

        img, points = extract_image_and_points(cloud, COLOR_HEIGHT, COLOR_WIDTH)
        img = (img * 255).astype(np.uint8) 

        rt, points_3d = detect_circle_pattern_corners(img, points, CALIB_BOARD_DIMS, vis=VIS_BOARD_DETECTION)
        if not rt:
            print("File '%s': calib pattern not found or 3d points contain nan values, skipping..."%(id))
            continue

        tag_pose = estimate_tag_pose(points_3d)
        tag_poses[id] = tag_pose

    if len(tag_poses) == 0:
        import sys
        print("Tag poses is empty, exiting.")
        sys.exit(0)

    T_we = np.empty((len(tag_poses), 4, 4), dtype=float)
    T_cp = np.empty((len(tag_poses), 4, 4), dtype=float)

    for i, id in enumerate(tag_poses):
        T_we[i, :, :] = ee_poses[id]
        T_cp[i, :, :] = tag_poses[id]

    T_wc, T_ep, residual = ping_pong_optimize(T_we, T_cp, 1000, 1e-6)

    print('T_wc:\n', T_wc)

    T_cw = inv(T_wc)


    # # VISUALIZE FINAL POSE
    mesh_frame_est = o3d.create_mesh_coordinate_frame(size = 0.3, origin = [0,0,0]) #original camera frame
    mesh_frame_est.transform(T_cw)
    mesh_frame_est2 = o3d.create_mesh_coordinate_frame(size = 0.4, origin = [0,0,0]) #original camera frame
    
    filter_cloud_nan(cloud)  # filter nans so that cloud can be visualized in open3d

    o3d.draw_geometries([cloud, mesh_frame_est, mesh_frame_est2])

