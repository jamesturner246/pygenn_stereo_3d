import numpy as np
import cv2
import dv


def view_aedat_frames(f):
    frame = np.zeros((cam_height, cam_width * 2, 3), dtype="uint8")
    for pos_L, neg_L, pos_R, neg_R in f:
        frame[:, :cam_width, 1] = pos_L
        frame[:, :cam_width, 2] = neg_L
        frame[:, cam_width:, 1] = pos_R
        frame[:, cam_width:, 2] = neg_R
        frame[frame > 0] = 255
        cv2.imshow("frame", frame)
        k = cv2.waitKey(0)
        if k == ord("q"):
            cv2.destroyWindow("frame")
            break


def aedat_read(data_path, calib_path, cam_height=260, cam_width=346, interval_usec=5000):
    K0 = np.load(calib_path + '/K0.npy')
    K1 = np.load(calib_path + '/K1.npy')
    D0 = np.load(calib_path + '/D0.npy')
    D1 = np.load(calib_path + '/D1.npy')
    R0 = np.load(calib_path + '/R0.npy')
    R1 = np.load(calib_path + '/R1.npy')
    P0 = np.load(calib_path + '/P0.npy')
    P1 = np.load(calib_path + '/P1.npy')

    with dv.AedatFile(data_path) as f:
        events_0 = np.hstack([packet for packet in f['events'].numpy()])
        events_1 = np.hstack([packet for packet in f['events_1'].numpy()])

        i_0 = i_1 = 0
        time_next = max(events_0[i_0]['timestamp'], events_1[i_1]['timestamp'])

        while i_0 < len(events_0) or i_1 < len(events_1):
            time = time_next
            time_next = time + interval_usec

            pos_0 = np.zeros((cam_height, cam_width), dtype='uint8')
            neg_0 = np.zeros((cam_height, cam_width), dtype='uint8')
            while i_0 < len(events_0) and events_0[i_0]['timestamp'] < time_next:
                xy = np.array([events_0[i_0]['x'], events_0[i_0]['y']], dtype='float64')
                xy_undistorted = cv2.undistortPoints(xy, K0, D0, None, R0, P0)[0, 0]
                xy_int = np.rint(xy_undistorted).astype('int32')
                xy_bounded = all(xy_int >= 0) and all(xy_int < [cam_width, cam_height])
                if xy_bounded:
                    if events_0[i_0]['polarity']:
                        pos_0[xy_int[1], xy_int[0]] += 1
                    else:
                        neg_0[xy_int[1], xy_int[0]] += 1
                i_0 += 1

            pos_1 = np.zeros((cam_height, cam_width), dtype='uint8')
            neg_1 = np.zeros((cam_height, cam_width), dtype='uint8')
            while i_1 < len(events_1) and events_1[i_1]['timestamp'] < time_next:
                xy = np.array([events_1[i_1]['x'], events_1[i_1]['y']], dtype='float64')
                xy_undistorted = cv2.undistortPoints(xy, K1, D1, None, R1, P1)[0, 0]
                xy_int = np.rint(xy_undistorted).astype('int32')
                xy_bounded = all(xy_int >= 0) and all(xy_int < [cam_width, cam_height])
                if xy_bounded:
                    if events_1[i_1]['polarity']:
                        pos_1[xy_int[1], xy_int[0]] += 1
                    else:
                        neg_1[xy_int[1], xy_int[0]] += 1
                i_1 += 1

            yield pos_0, neg_0, pos_1, neg_1
