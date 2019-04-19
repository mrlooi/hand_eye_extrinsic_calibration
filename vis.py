import open3d #as o3d
import numpy as np
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import quat2mat

def get_data(key="RadianSetup1"):

    if key == "RadianSetup2":
        camera_extrinsic_gt = {
                "position": [
                1.5026177886697314, 
                -0.43833321316199203, 
                2.5773974481543376
                ], 
                "orientation": [
                0.15361315314913665, 
                -0.7109802579304441, 
                -0.6750412709789273, 
                0.12340727080643848
                ]
            }

        est_T_wc =  np.array([
             [ 0.05129271,  0.9173576 , -0.3947456 ,  1.50492931],
             [ 0.99654564, -0.07286512, -0.03984299, -0.40877499],
             [-0.06531345, -0.39133835, -0.91792617,  2.58123   ],
             [ 0.        ,  0.        ,  0.      ,    1.        ]
        ])
    elif key == "RadianSetup1":
        camera_extrinsic_gt = {
            "position": [
            1.5026177886697314, 
            -0.43833321316199203, 
            2.5773974481543376
            ], 
            "orientation": [
            0.15361315314913665, 
            -0.7109802579304441, 
            -0.6750412709789273, 
            0.12340727080643848
            ]
        }
    elif key == "NeveSetup":
        camera_extrinsic_gt = {
            "position": [
            1.6729682311702465, 
            0.08451025077097321, 
            2.014965414389718
            ], 
            "orientation": [
            0.05256023904098993, 
            -0.7142395129244176, 
            -0.69346038049903, 
            0.07881649654472865
            ]
        }
    else:
        raise ValueError

    est_T_wc = np.load("%s_T_wc.npy"%(key))

    T_wc = np.eye(4)
    T_wc[:3,:3] = quat2mat(camera_extrinsic_gt["orientation"])
    T_wc[:3,-1] = camera_extrinsic_gt["position"]
    T_cw = np.linalg.inv(T_wc)  # cam to world_frame

    est_T_cw = np.linalg.inv(est_T_wc)

    return T_cw, est_T_cw

if __name__ == '__main__':
    # file_ids = ['0', '1', '2']
    # file_1 = file_ids[0]
    key = "NeveSetup"
    data_path = './data/one_shot_calib_data_%s/extrinsic/data/'%(key)
    pcd_file = data_path + "cloud_xyzrgba/cloud_xyzrgba_0.pcd"
    cloud_1 = open3d.read_point_cloud(pcd_file)
    T_cw, est_T_cw = get_data(key)


    # gt mesh frame
    mesh_frame_gt = open3d.create_mesh_coordinate_frame(size = 0.5, origin = [0,0,0]) #original camera frame
    mesh_frame_gt.transform(T_cw)

    # estimated mesh frame
    mesh_frame_est = open3d.create_mesh_coordinate_frame(size = 0.3, origin = [0,0,0]) #original camera frame
    mesh_frame_est.transform(est_T_cw)

    open3d.draw_geometries([cloud_1, mesh_frame_gt, mesh_frame_est])
