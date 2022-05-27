import numpy as np
import cv2
import dv


def aedat_view_frames(
        f, cam_height, cam_width):
    frame = np.zeros((cam_height, cam_width * 2, 3), dtype="uint8")
    for pos_L, neg_L, pos_R, neg_R in f:
        frame[:, :cam_width, 0] = pos_L
        frame[:, :cam_width, 2] = neg_L
        frame[:, cam_width:, 0] = pos_R
        frame[:, cam_width:, 2] = neg_R
        frame[frame > 0] = 255
        cv2.imshow("frame", frame)
        k = cv2.waitKey(0)
        if k == ord("q"):
            cv2.destroyWindow("frame")
            break


def aedat_read_frames(
        data_path, calib_path,
        cam_height=260, cam_width=346,
        resize_height=260, resize_width=346,
        resize_interpolation=cv2.INTER_NEAREST,
        skip_usec=0, interval_usec=5000):
    K_L = np.load(calib_path + '/K0.npy')
    K_R = np.load(calib_path + '/K1.npy')
    D_L = np.load(calib_path + '/D0.npy')
    D_R = np.load(calib_path + '/D1.npy')
    R_L = np.load(calib_path + '/R0.npy')
    R_R = np.load(calib_path + '/R1.npy')
    P_L = np.load(calib_path + '/P0.npy')
    P_R = np.load(calib_path + '/P1.npy')

    with dv.AedatFile(data_path) as f:
        events_L = np.hstack([packet for packet in f['events'].numpy()])
        events_R = np.hstack([packet for packet in f['events_1'].numpy()])

        i_L = i_R = 0
        time_next = max(events_L[i_L]['timestamp'], events_R[i_R]['timestamp']) + skip_usec
        while i_L < len(events_L) and events_L[i_L]['timestamp'] < time_next:
            i_L += 1
        while i_R < len(events_R) and events_R[i_R]['timestamp'] < time_next:
            i_R += 1

        while i_L < len(events_L) or i_R < len(events_R):
            time = time_next
            time_next = time + interval_usec

            pos_L = np.zeros((cam_height, cam_width), dtype='uint8')
            neg_L = np.zeros((cam_height, cam_width), dtype='uint8')
            while i_L < len(events_L) and events_L[i_L]['timestamp'] < time_next:
                xy = np.array([events_L[i_L]['x'], events_L[i_L]['y']], dtype='float64')
                xy_undistorted = cv2.undistortPoints(xy, K_L, D_L, None, R_L, P_L)[0, 0]
                xy_int = np.rint(xy_undistorted).astype('int32')
                xy_bounded = all(xy_int >= 0) and all(xy_int < [cam_width, cam_height])
                if xy_bounded:
                    if events_L[i_L]['polarity']:
                        pos_L[xy_int[1], xy_int[0]] += 1
                    else:
                        neg_L[xy_int[1], xy_int[0]] += 1
                i_L += 1
            pos_L = cv2.resize(pos_L, dsize=(resize_width, resize_height), interpolation=resize_interpolation)
            neg_L = cv2.resize(neg_L, dsize=(resize_width, resize_height), interpolation=resize_interpolation)

            pos_R = np.zeros((cam_height, cam_width), dtype='uint8')
            neg_R = np.zeros((cam_height, cam_width), dtype='uint8')
            while i_R < len(events_R) and events_R[i_R]['timestamp'] < time_next:
                xy = np.array([events_R[i_R]['x'], events_R[i_R]['y']], dtype='float64')
                xy_undistorted = cv2.undistortPoints(xy, K_R, D_R, None, R_R, P_R)[0, 0]
                xy_int = np.rint(xy_undistorted).astype('int32')
                xy_bounded = all(xy_int >= 0) and all(xy_int < [cam_width, cam_height])
                if xy_bounded:
                    if events_R[i_R]['polarity']:
                        pos_R[xy_int[1], xy_int[0]] += 1
                    else:
                        neg_R[xy_int[1], xy_int[0]] += 1
                i_R += 1
            pos_R = cv2.resize(pos_R, dsize=(resize_width, resize_height), interpolation=resize_interpolation)
            neg_R = cv2.resize(neg_R, dsize=(resize_width, resize_height), interpolation=resize_interpolation)

            yield pos_L, neg_L, pos_R, neg_R
