"""
Written by Vincent Looi and Jerry Yu Zhao
"""
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

def ping_pong_optimize(T_we, T_cp, max_iteration=1000, min_residual_change=1e-6):
    """
    T_we * T_ep = T_wc * T_cp. Optimize T_wc and T_ep alternatively.
    :param T_we: 4x4 transformation matrix, row-major
    :param T_cp: 4x4 transformation matrix, row-major
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
    :param T_we: 4x4 transformation matrix, row-major
    :param T_cp: 4x4 transformation matrix, row-major
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

    delta = np.dot(R_wc, t_cp) + t_wc.reshape(3, 1) - t_wp
    residual = np.sum(np.linalg.norm(delta, axis=0)) / t_wp.shape[1]

    return T_wc, residual


def optimize_T_ep_umeyama(T_we, T_cp, T_wc):
    """
    T_we * T_ep = T_wc * T_cp. Optimize T_ep by fixing T_wc
    w = world frame
    e = ee tip frame
    c = camera frame
    p = tag plane frame
    :param T_we: 4x4 transformation matrix, row-major
    :param T_cp: 4x4 transformation matrix, row-major
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

    delta = np.dot(R_pe, t_ew) + t_pe.reshape(3, 1) - t_pw
    residual = np.sum(np.linalg.norm(delta, axis=0)) / t_pw.shape[1]

    T_ep = np.eye(4, dtype=float)
    T_ep[:3, :3] = R_pe.T
    T_ep[:3, 3] = np.dot(-R_pe.T, t_pe)

    return T_ep, residual


def optimize_T_ep_linear(T_we, T_cp, T_wc):
    """
    T_we * T_ep = T_wc * T_cp. Optimize T_ep by fixing T_wc
    w = world frame
    e = ee tip frame
    c = camera frame
    p = tag plane frame
    :param T_we: 4x4 transformation matrix, row-major
    :param T_cp: 4x4 transformation matrix, row-major
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
        q_ep_all[i, :] = mat2quat(R_ep_all[i])

    cov_q_ep = np.dot(q_ep_all.T, q_ep_all)

    ws, vs = np.linalg.eigh(cov_q_ep)
    q_ep = vs[:, -1]
    T_ep = np.eye(4)
    T_ep[:3,:3] = quat2mat(q_ep)
    T_ep[:3, 3] = t_ep.reshape(-1)

    return T_ep


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