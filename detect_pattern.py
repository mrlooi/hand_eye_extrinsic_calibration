"""
File to detect calibration board patterns
Currently supports asymmetric circles and apriltags
"""
import numpy as np
import cv2
cv = cv2

def detect_circle_pattern_corners(img, points, board_dims=(4,11), vis=True):
    BOARD_W, BOARD_H = board_dims

    # board_corners_index = np.array([0,BOARD_W-1,2*BOARD_W*(BOARD_W-1)+BOARD_W-1,2*BOARD_W*(BOARD_W-1)]) # 4 corners to form square
    board_corners_index = np.array([0,BOARD_W-1,-1,-BOARD_W]) # 4 board corners, ordered

    points_3d = np.zeros((4, 3), dtype=np.float32)

    img_copy = img.copy()
    rt, pix = cv2.findCirclesGrid(img_copy, board_dims, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if rt:
        pix = pix.squeeze()

        # board_corners_index = np.array([16,19,-4,-1])
        # board_corners_index = np.array([0,BOARD_W-1,-1,-(BOARD_W)])
        pix = pix[board_corners_index]
        for ix,px in enumerate(pix):
            px = np.round(px).astype(np.int32)
            x,y = px
            cv2.circle(img_copy, tuple(px), 2, (0,0,255))
            point_3d = points[y, x]
            if np.any(np.isnan(point_3d)):
                print("Nan value found in point (%d,%d)"%(x,y))
                rt = False

            points_3d[ix] = point_3d

    else:
        print("[WARN] Could not detect calib pattern (w,h): (%d,%d)"%(BOARD_W,BOARD_H))

    if vis:
        cv2.imshow("detected_pattern", img_copy)
        cv2.waitKey(0)

    return rt, points_3d

def detect_april_tag_corners(img, points, vis=False):
    """
    points: numpy array of shape (H,W,3)  # x y z
    img_points: numpy array of shape (H,W,3)  # b g r
    """
    pts_shape = points.shape
    img_shape = img.shape
    assert len(pts_shape) == 3 and pts_shape == img_shape and pts_shape[-1] == 3

    import apriltag

    detector = apriltag.Detector()
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    detections, dimg = detector.detect(gray, True)

    if len(detections) == 0:
        print("[WARN]: No detections")
        return False, None

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
                point_3d = points[y,x]
                if np.any(np.isnan(point_3d)):
                    raise ValueError
                img_to_show = cv.circle(img_to_show, (x, y), 4, (255, 0, 0), 2)
                # c *= 2
                points_3d.append(point_3d)
            points_3d_for_all.append(points_3d)
        except ValueError:
            print("[WARN]: Corner point in image is not registered by camera. It has 'nan' value, Please change the view")
            return False, None

        if vis:
            cv2.imshow('Detected Apriltag Corners', img_to_show)
            cv2.waitKey(0)
        # while cv.waitKey(5) < 0:
        #     pass

    return True, points_3d_for_all


