import numpy as np
import cv2
import dv


def aedat_generator(
        data_path, cam_calibration_path,
        cam_size=(260, 346), size=(260, 346),
        interpolation=cv2.INTER_NEAREST,
        skip_usec=0, interval_usec=5000):
    K_L = np.load(f"{cam_calibration_path}/K0.npy")
    K_R = np.load(f"{cam_calibration_path}/K1.npy")
    D_L = np.load(f"{cam_calibration_path}/D0.npy")
    D_R = np.load(f"{cam_calibration_path}/D1.npy")
    R_L = np.load(f"{cam_calibration_path}/R0.npy")
    R_R = np.load(f"{cam_calibration_path}/R1.npy")
    P_L = np.load(f"{cam_calibration_path}/P0.npy")
    P_R = np.load(f"{cam_calibration_path}/P1.npy")

    with dv.AedatFile(data_path) as f:
        events_L = np.hstack([packet for packet in f["events"].numpy()])
        events_R = np.hstack([packet for packet in f["events_1"].numpy()])
        pos_L = np.empty(cam_size, dtype=np.uint8)
        neg_L = np.empty(cam_size, dtype=np.uint8)
        pos_R = np.empty(cam_size, dtype=np.uint8)
        neg_R = np.empty(cam_size, dtype=np.uint8)

        i_L = i_R = 0
        time_next = max(events_L[i_L]["timestamp"], events_R[i_R]["timestamp"]) + skip_usec
        while i_L < len(events_L) and events_L[i_L]["timestamp"] < time_next:
            i_L += 1
        while i_R < len(events_R) and events_R[i_R]["timestamp"] < time_next:
            i_R += 1

        while i_L < len(events_L) or i_R < len(events_R):
            time = time_next
            time_next = time + interval_usec

            pos_L.fill(0)
            neg_L.fill(0)
            while i_L < len(events_L) and events_L[i_L]["timestamp"] < time_next:
                xy = np.array([events_L[i_L]["x"], events_L[i_L]["y"]], dtype=np.float64)
                xy_undistorted = cv2.undistortPoints(xy, K_L, D_L, None, R_L, P_L)[0, 0]
                xy_int = np.rint(xy_undistorted).astype("int32")
                xy_bounded = all(xy_int >= 0) and all(xy_int < np.flip(cam_size))
                if xy_bounded:
                    if events_L[i_L]["polarity"]:
                        pos_L[xy_int[1], xy_int[0]] += 1
                    else:
                        neg_L[xy_int[1], xy_int[0]] += 1
                i_L += 1
            pos_L_new = cv2.resize(
                pos_L, dsize=np.flip(size),
                interpolation=interpolation)
            neg_L_new = cv2.resize(
                neg_L, dsize=np.flip(size),
                interpolation=interpolation)

            pos_R.fill(0)
            neg_R.fill(0)
            while i_R < len(events_R) and events_R[i_R]["timestamp"] < time_next:
                xy = np.array([events_R[i_R]["x"], events_R[i_R]["y"]], dtype=np.float64)
                xy_undistorted = cv2.undistortPoints(xy, K_R, D_R, None, R_R, P_R)[0, 0]
                xy_int = np.rint(xy_undistorted).astype("int32")
                xy_bounded = all(xy_int >= 0) and all(xy_int < np.flip(cam_size))
                if xy_bounded:
                    if events_R[i_R]["polarity"]:
                        pos_R[xy_int[1], xy_int[0]] += 1
                    else:
                        neg_R[xy_int[1], xy_int[0]] += 1
                i_R += 1
            pos_R_new = cv2.resize(
                pos_R, dsize=np.flip(size),
                interpolation=interpolation)
            neg_R_new = cv2.resize(
                neg_R, dsize=np.flip(size),
                interpolation=interpolation)

            yield pos_L_new, neg_L_new, pos_R_new, neg_R_new
