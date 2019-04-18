import open3d as o3d
import numpy as np
import apriltag
import cv2 as cv
import json
import transformations as tr


KINECT_COLOR_HEIGHT = 540
KINECT_COLOR_WIDTH = 960

# NeveSetup
file_ids = ['0', '1', '2', '3', '4']
data_path = './data/one_shot_calib_data_NeveSetup/'

# RadianSetup1
# file_ids = ['0', '1', '2', '3', '4', '5', '6', '7']
# data_path = './data/one_shot_calib_data_RadianSetup1/extrinsic/data/'

# RadianSetup2
# file_ids = ['0', '1', '2']
# data_path = './data/one_shot_calib_data_RadianSetup2/extrinsic/data/'


def main():
    tag_poses = {}
    for id in file_ids:
        pcd = o3d.read_point_cloud_with_nan(data_path + 'cloud_xyzrgba/cloud_xyzrgba_%s.pcd' % id)
        img, cloud_points = extract_image_and_points(pcd)
        points_3d_for_all = detect_tag_corners(cloud_points, img)
        # we assume there's only one tag in a image.
        for points_3d in points_3d_for_all:
            tag_pose = estimate_tag_pose(points_3d)
            tag_poses[id] = tag_pose

    ee_poses = read_ee_poses(data_path + 'world_frame_to_ee_tip_0_tfs.json')

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

    output_traj_for_evo('./data/evo/T_wp1.txt', T_wp_1)
    output_traj_for_evo('./data/evo/T_wp2.txt', T_wp_2)

    output_poses_for_evo(tag_poses, ee_poses)


def ping_pong_optimize(T_we, T_cp, max_iteration=1000, min_residual_change=1e-6):
    """
    T_we * T_ep = T_wc * T_cp. Optimize T_wc and T_ep alternatively.
    :param T_we: multiple 4x4 transform matrix stacked by column
    :param T_cp: multiple 4x4 transform matrix stacked by column
    :param max_iteration:
    :param min_residual_change:
    :return: T_wc, T_ep, residual
    """
    T_ep = np.eye(4, dtype=float)
    T_wc = np.eye(4, dtype=float)

    last_residual = 1e9
    converged = False
    for i in range(max_iteration):
        T_wc, residual = optimize_T_wc(T_we, T_cp, T_ep)
        # T_ep, _ = optimize_T_ep_umeyama(T_we, T_cp, T_wc)
        T_ep = optimize_T_ep_linear(T_we, T_cp, T_wc)  # The linear method is much better than the umeyama method for solving T_ep

        print('Residual:', residual)

        if abs(last_residual - residual) < min_residual_change:
            converged = True
            last_residual = residual
            break
        last_residual = residual

    if converged:
        print('optimize finished. min_residual_change condition satisfied.')
    else:
        print('optimize finished. max_iteration condition satisfied.')

    return T_wc, T_ep, last_residual


def optimize_T_wc(T_we, T_cp, T_ep):
    """
    T_we * T_ep = T_wc * T_cp. Optimize T_wc by fixing T_ep
    w = world frame
    e = ee tip frame
    c = camera frame
    p = tag plane frame
    :param T_we: multiple 4x4 transform matrix stacked by column
    :param T_cp: multiple 4x4 transform matrix stacked by column
    :param T_ep:
    :return: T_wc, residual
    """
    T_wp = np.matmul(T_we, T_ep)

    t_wp = T_wp[:, 0:3, 3].T
    t_cp = T_cp[:, 0:3, 3].T

    R_wc, t_wc, s = umeyama_alignment(t_cp, t_wp)

    T_wc = np.eye(4, dtype=float)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc

    delta = R_wc @ t_cp + t_wc.reshape(3, 1) - t_wp
    residual = np.sum(np.linalg.norm(delta, axis=0)) / t_wp.shape[1]

    return T_wc, residual


def optimize_T_ep_umeyama(T_we, T_cp, T_wc):
    """
    T_we * T_ep = T_wc * T_cp. Optimize T_ep by fixing T_wc
    w = world frame
    e = ee tip frame
    c = camera frame
    p = tag plane frame
    :param T_we: multiple 4x4 transform matrix stacked by column
    :param T_cp: multiple 4x4 transform matrix stacked by column
    :param T_wc:
    :return: T_ep, residual
    """
    T_wp = np.matmul(T_wc, T_cp)

    T_pw = np.linalg.inv(T_wp)
    T_ew = np.linalg.inv(T_we)

    t_pw = T_pw[:, 0:3, 3].T
    t_ew = T_ew[:, 0:3, 3].T

    R_pe, t_pe, s = umeyama_alignment(t_ew, t_pw)

    T_pe = np.eye(4, dtype=float)
    T_pe[:3, :3] = R_pe
    T_pe[:3, 3] = t_pe

    delta = R_pe @ t_ew + t_pe.reshape(3, 1) - t_pw
    residual = np.sum(np.linalg.norm(delta, axis=0)) / t_pw.shape[1]

    T_ep = np.eye(4, dtype=float)
    T_ep[:3, :3] = R_pe.T
    T_ep[:3, 3] = -R_pe.T @ t_pe

    return T_ep, residual


def optimize_T_ep_linear(T_we, T_cp, T_wc):
    """
    T_we * T_ep = T_wc * T_cp. Optimize T_ep by fixing T_wc
    w = world frame
    e = ee tip frame
    c = camera frame
    p = tag plane frame
    :param T_we: multiple 4x4 transform matrix stacked by column
    :param T_cp: multiple 4x4 transform matrix stacked by column
    :param T_wc:
    :return: T_ep, residual
    """
    T_wp = np.matmul(T_wc, T_cp)

    t_wp = T_wp[:, :3, 3].reshape(-1, 3, 1)
    R_wp = T_wp[:, :3, :3].reshape(-1, 3, 3)

    t_we = T_we[:, :3, 3].reshape(-1, 3, 1)
    R_we = T_we[:, :3, :3].reshape(-1, 3, 3)
    R_ew = np.transpose(R_we, axes=(0, 2, 1))

    t_ep_all = np.matmul(R_ew, t_wp - t_we)
    t_ep = np.sum(t_ep_all, axis=0) / t_ep_all.shape[0]

    R_ep_all = np.matmul(R_ew, R_wp)
    q_ep_all = np.zeros((R_ep_all.shape[0], 4), dtype=float)
    for i in range(R_ep_all.shape[0]):
        q_ep_all[i, :] = tr.quaternion_from_matrix(R_ep_all[i])

    cov_q_ep = q_ep_all.T @ q_ep_all

    ws, vs = np.linalg.eigh(cov_q_ep)
    q_ep = vs[:, -1]
    T_ep = tr.quaternion_matrix(q_ep)
    T_ep[:3, 3] = t_ep.reshape(-1)

    return T_ep


def output_poses_for_evo(tag_poses, ee_poses):
    # tag_poses_file = open('./data/evo/tag_poses.txt', 'w')
    # ee_poses_file = open('./data/evo/ee_poses.txt', 'w')

    tag_traj = np.zeros((len(file_ids), 8), dtype=np.float32)
    ee_traj = np.zeros((len(file_ids), 8), dtype=np.float32)
    fake_timestamp = 0
    for id in file_ids:
        T = tag_poses[id]
        t = T[:3, 3]
        q = tr.quaternion_from_matrix(T)
        tag_traj[fake_timestamp, :] = np.array([fake_timestamp, t[0], t[1], t[2], q[1], q[2], q[3], q[0]])

        T = ee_poses[id]
        t = T[:3, 3]
        q = tr.quaternion_from_matrix(T)
        ee_traj[fake_timestamp, :] = np.array([fake_timestamp, t[0], t[1], t[2], q[1], q[2], q[3], q[0]])

        fake_timestamp += 1

    np.savetxt('./data/evo/tag_poses.txt', tag_traj)
    np.savetxt('./data/evo/ee_poses.txt', ee_traj)


def output_traj_for_evo(file_name, traj):
    traj_tum = np.zeros((traj.shape[0], 8), dtype=np.float32)
    for i in range(traj.shape[0]):
        T = traj[i, :, :]
        t = T[:3, 3]
        q = tr.quaternion_from_matrix(T)
        traj_tum[i, :] = np.array([i, t[0], t[1], t[2], q[1], q[2], q[3], q[0]])

    np.savetxt(file_name, traj_tum)


def read_ee_poses(file_name):
    with open(file_name, 'r') as f:
        wf_to_ee = json.load(f)

    poses_data = wf_to_ee["world_frame_to_ee_tip_0_tfs"]
    ee_poses = {}
    for key, value in poses_data.items():
        data = poses_data[key]
        t = np.array([data['x_pos'], data['y_pos'], data['z_pos']], dtype=float)
        q = np.array([data['w_rot'], data['x_rot'], data['y_rot'], data['z_rot']], dtype=float)
        T = tr.quaternion_matrix(q)
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
    img_to_show = img[...]

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


def umeyama_alignment(x, y, with_scale=False):
    """
    R*x + t = y
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        print("data matrices must have the same shape")
        sys.exit(0)

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


if __name__ == '__main__':
    main()
