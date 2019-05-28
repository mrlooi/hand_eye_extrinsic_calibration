import open3d as o3d
import numpy as np
import apriltag
import cv2 as cv
import json
import os, os.path as osp

from transforms3d.quaternions import quat2mat, mat2quat

from ext_calib_optimizer import ping_pong_optimize
from utils_3d import filter_cloud_nan, estimate_tag_pose, extract_image_and_points
from detect_pattern import detect_april_tag_corners


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
    """
    INPUT START
    """

    KINECT_COLOR_HEIGHT = 540
    KINECT_COLOR_WIDTH = 960
    point_multiplier = 1.0  # IMPORTANT!!

    file_ids = ['0', '1', '2', '3', '4', '5', '6', '7']

    data_path = './data/april_tag'
    ee_pose_file = osp.join(data_path, 'world_frame_to_ee_tip_0_tfs.json')
    pcd_files = [osp.join(data_path, 'pcd_files/cloud_xyzrgba_%s.pcd'%(id)) for id in file_ids]
    
    VIS_BOARD_DETECTION = False
    """
    INPUT END
    """

    ee_poses = read_ee_poses(ee_pose_file)

    tag_poses = {}
    for i, id in enumerate(file_ids):
        pcd_file = pcd_files[i]
        print("Reading from %s"%(pcd_file))
        rt, pcd = read_point_cloud(pcd_file, point_multiplier)
        if not rt:
            continue

        img, cloud_points = extract_image_and_points(pcd, KINECT_COLOR_HEIGHT, KINECT_COLOR_WIDTH)
        img = (img * 255).astype(np.uint8) 

        rt, points_3d = detect_april_tag_corners(img, cloud_points, vis=VIS_BOARD_DETECTION)
        if not rt:
            print("File '%s': calib pattern not found or 3d points contain nan values, skipping..."%(id))
            continue

        assert len(points_3d) == 1  # ASSUMES JUST 1 TAG IS USED
        tag_pose = estimate_tag_pose(points_3d[0]) 
        tag_poses[id] = tag_pose


    if len(tag_poses) == 0:
        import sys
        print("Tag poses is empty, exiting.")
        sys.exit(0)

    T_we = np.empty((len(file_ids), 4, 4), dtype=float)
    T_cp = np.empty((len(file_ids), 4, 4), dtype=float)
    for i, id in enumerate(file_ids):
        T_we[i, :, :] = ee_poses[id]
        T_cp[i, :, :] = tag_poses[id]

    T_wc, T_ep, residual = ping_pong_optimize(T_we, T_cp, 1000, 1e-6)

    print('T_wc:\n', T_wc)
    # print('T_ep:\n', T_ep)

    # out_T_wc_file = "%s_T_wc.npy"%(key)
    # np.save(out_T_wc_file, T_wc)
    # print("Saved T_wc to %s"%(out_T_wc_file))

    T_cw = np.linalg.inv(T_wc)

    # # VISUALIZE FINAL POSE
    mesh_frame_est = o3d.create_mesh_coordinate_frame(size = 0.3, origin = [0,0,0]) #original camera frame
    mesh_frame_est.transform(T_cw)
    # mesh_frame_est2.transform(M_cam2base)
    
    filter_cloud_nan(pcd)  # filter nans so that cloud can be visualized in open3d

    o3d.draw_geometries([pcd, mesh_frame_est])

