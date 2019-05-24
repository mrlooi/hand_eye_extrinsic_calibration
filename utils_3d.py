import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import open3d

def filter_cloud_nan(cloud):
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    valid = ~np.isnan(points[:, 0])
    cloud.points = open3d.Vector3dVector(points[valid])
    cloud.colors = open3d.Vector3dVector(colors[valid])

def create_cloud(points, colors=[], normals=[]):
    cloud = open3d.PointCloud()
    cloud.points = open3d.Vector3dVector(points)
    if len(colors) > 0:
        assert len(colors) == len(points)
        cloud.colors = open3d.Vector3dVector(colors)
    if len(normals) > 0:
        assert len(normals) == len(points)
        cloud.normals = open3d.Vector3dVector(normals)

    return cloud

def extract_image_and_points(cloud, cloud_height, cloud_width, flip_rgb=False):
    points = np.asarray(cloud.points)
    color = np.asarray(cloud.colors)

    img = color.reshape((cloud_height, cloud_width, 3))
    if flip_rgb:
        img = img[:, :, ::-1].copy()
    points = points.reshape((cloud_height, cloud_width, 3))
    return img, points

def estimate_tag_pose(tag_corners_3d):
    assert len(tag_corners_3d) == 4, "tag should have 4 corners"

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

