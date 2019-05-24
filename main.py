import open3d as o3d
import numpy as np
import apriltag
import cv2 as cv
import json
import os, os.path as osp

from transforms3d.quaternions import quat2mat, mat2quat
from ext_calib_optimizer import ping_pong_optimize

KINECT_COLOR_HEIGHT = 540
KINECT_COLOR_WIDTH = 960

key = "NeveSetup"
file_ids = ['0', '1', '2', '3', '4']

# key = "RadianSetup1"
# file_ids = ['0', '1', '2', '3', '4']#, '5', '6']#, '7']

# key = "RadianSetup2"
# file_ids = ['0', '1', '2']

data_path = './data/one_shot_calib_data_%s/extrinsic/data/'%(key)
ee_pose_file = osp.join(data_path, 'world_frame_to_ee_tip_0_tfs.json')
pcd_files = [osp.join(data_path, 'cloud_xyzrgba/cloud_xyzrgba_%s.pcd'%(id)) for id in file_ids]

def main():
    tag_poses = {}
    for i, id in enumerate(file_ids):
        pcd_file = pcd_files[i]
        print("Reading from %s"%(pcd_file))
        pcd = o3d.read_point_cloud_with_nan(pcd_file)
        img, cloud_points = extract_image_and_points(pcd)
        points_3d_for_all = detect_tag_corners(cloud_points, img)
        # we assume there's only one tag in a image.
        for points_3d in points_3d_for_all:
            tag_pose = estimate_tag_pose(points_3d)
            tag_poses[id] = tag_pose

    ee_poses = read_ee_poses(ee_pose_file)

    T_we = np.empty((len(file_ids), 4, 4), dtype=float)
    T_cp = np.empty((len(file_ids), 4, 4), dtype=float)
    for i, id in enumerate(file_ids):
        T_we[i, :, :] = ee_poses[id]
        T_cp[i, :, :] = tag_poses[id]

    T_wc, T_ep, residual = ping_pong_optimize(T_we, T_cp, 1000, 1e-6)

    print('T_wc:\n', T_wc)
    print('T_ep:\n', T_ep)

    T_wp_1 = np.matmul(T_we, T_ep)
    T_wp_2 = np.matmul(T_wc, T_cp)

    out_T_wc_file = "%s_T_wc.npy"%(key)
    np.save(out_T_wc_file, T_wc)
    print("Saved T_wc to %s"%(out_T_wc_file))


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


def estimate_tag_pose(tag_corners_3d):
    # we assume each tag has 4 corners
    t = tag_corners_3d[0]
    rx = tag_corners_3d[3] - t
    ry = tag_corners_3d[1] - t

    sx = np.linalg.norm(rx)
    sy = np.linalg.norm(ry)

    rx /= sx
    ry /= sy

    rz = np.cross(rx, ry)
    rz /= np.linalg.norm(rz)

    T = np.eye(4, dtype=float)
    T[:3, 0] = rx
    T[:3, 1] = ry
    T[:3, 2] = rz
    T[:3, 3] = t

    return T


def extract_image_and_points(cloud):
    points = np.asarray(cloud.points)
    color = np.asarray(cloud.colors)
    img = color.reshape((KINECT_COLOR_HEIGHT, KINECT_COLOR_WIDTH, 3)) * 255
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    return img, points


def detect_tag_corners(cloud_points, img):
    detector = apriltag.Detector()
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    detections, dimg = detector.detect(gray, True)
    img_to_show = img[...].copy()

    points_3d_for_all = []
    for detection in detections:
        points_3d = []
        points_2d = np.round(detection.corners).astype(int)

        c = 30
        try:
            for point_2d in points_2d:
                x, y = point_2d
                # TODO: use bilinear interpolation to improve precision.
                point_3d = cloud_points[x + y*KINECT_COLOR_WIDTH]
                if np.any(np.isnan(point_3d)):
                    raise ValueError
                img_to_show = cv.circle(img_to_show, (x, y), 4, (0, 0, c), 2)
                c *= 2
                points_3d.append(point_3d)
            points_3d_for_all.append(points_3d)
        except ValueError:
            print("Corner point in image is not registered by camera. It has 'nan' value, Please change the view")

        # cv.imshow('Detected Apriltag Corners', img_to_show)
        # while cv.waitKey(5) < 0:
        #     pass

    return points_3d_for_all


if __name__ == '__main__':
    main()
