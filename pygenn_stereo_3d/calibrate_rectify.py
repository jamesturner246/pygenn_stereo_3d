import matplotlib.pyplot as plt
import numpy as np
import cv2
import dv
import os
import time


def calibrate_camera(dv_address, dv_port, cb_shape=(6, 9), cb_square_size=0.024):

    # prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ...., (5, 8, 0)
    objp = np.zeros((cb_shape[0] * cb_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_shape[0], 0:cb_shape[1]].T.reshape(-1, 2)
    objp *= cb_square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    sample_i = 0

    with dv.NetworkFrameInput(address=dv_address, port=dv_port) as f:

        for frame in f:
            image_colour = frame.image.copy()
            image_grey = cv2.cvtColor(image_colour, cv2.COLOR_BGR2GRAY)

            flags = 0
            flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            flags |= cv2.CALIB_CB_FILTER_QUADS
            ret, corners = cv2.findChessboardCorners(
                image_grey, cb_shape, flags=flags)

            if ret:
                # If found, refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(image_grey, corners, (5, 5), (-1, -1), criteria)

                # Draw and display the corners
                cv2.drawChessboardCorners(image_colour, cb_shape, corners, ret)

            # Show frame
            cv2.imshow('frame image', image_colour)
            k = cv2.waitKey(1)

            # Save good frame
            if k == ord(' ') and ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)
                sample_i += 1
                print(sample_i)
            elif k == ord('q'):
                break

    cv2.destroyWindow('frame image')

    # Calibrate
    image_size = image_grey.shape[::-1]

    ret, K, D, R, T = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print('cv2.calibrateCamera ret:', ret)

    return K, D, R, T


def test_calibrate(dv_address, dv_port, K, D):

    with dv.NetworkFrameInput(address=dv_address, port=dv_port) as f:
        frame = next(f)
        image_distorted = frame.image.copy()

    point_distorted = []
    for j in range(33):
        for k in range(24):
            point = np.array([10 * j + 10, 10 * k + 10])
            point_distorted.append(point.astype('float64'))
            image_distorted[point[1], point[0], :] = np.array([0, 0, 0])
    point_distorted = np.vstack(point_distorted)

    image_undistorted = cv2.undistort(image_distorted, K, D, None, K)

    point_undistorted = []
    for xy in point_distorted:
        xy_undistorted = cv2.undistortPoints(xy, K, D, None, K)[0]
        point_undistorted.append(xy_undistorted)
    point_undistorted = np.vstack(point_undistorted)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_distorted[:, :, (2, 1, 0)])
    ax[1].imshow(image_undistorted[:, :, (2, 1, 0)])
    ax[1].plot(point_undistorted[:, 0], point_undistorted[:, 1], 'b.', markerfacecolor=None)
    plt.show()

    return


def rectify_cameras(dv_address, dv_port0, dv_port1, K0, K1, D0, D1, cb_shape=(6, 9), cb_square_size=0.024):

    # prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ...., (6, 5, 0)
    objp = np.zeros((cb_shape[0] * cb_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_shape[0], 0:cb_shape[1]].T.reshape(-1, 2)
    objp *= cb_square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints0 = [] # 2d points in image (0) plane.
    imgpoints1 = [] # 2d points in image (1) plane.

    sample_i = 0

    # Connect to cameas
    f0 = dv.NetworkFrameInput(address=dv_address, port=dv_port0)
    f1 = dv.NetworkFrameInput(address=dv_address, port=dv_port1)

    with f0, f1:
        frame0 = next(f0)
        frame1 = next(f1)

        assert frame0.image.shape == frame1.image.shape

        while True:
            if frame0.timestamp < frame1.timestamp:
                frame0 = next(f0)
            else:
                frame1 = next(f1)

            image0_colour = frame0.image.copy()
            image0_grey = cv2.cvtColor(image0_colour, cv2.COLOR_BGR2GRAY)
            image1_colour = frame1.image.copy()
            image1_grey = cv2.cvtColor(image1_colour, cv2.COLOR_BGR2GRAY)

            flags = 0
            flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            flags |= cv2.CALIB_CB_FILTER_QUADS
            ret0, corners0 = cv2.findChessboardCorners(
                image0_grey, cb_shape, flags=flags)
            ret1, corners1 = cv2.findChessboardCorners(
                image1_grey, cb_shape, flags=flags)

            if ret0 and ret1:

                # If found, refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners0 = cv2.cornerSubPix(image0_grey, corners0, (5, 5), (-1, -1), criteria)
                corners1 = cv2.cornerSubPix(image1_grey, corners1, (5, 5), (-1, -1), criteria)

                # Draw and display the corners
                cv2.drawChessboardCorners(image0_colour, cb_shape, corners0, ret0)
                cv2.drawChessboardCorners(image1_colour, cb_shape, corners1, ret1)

            image_colour = np.append(image0_colour, image1_colour, axis=1)

            # Show frame
            cv2.imshow('frame image', image_colour)
            k = cv2.waitKey(1)

            # Save good frame
            if k == ord(' '):
                objpoints.append(objp)
                imgpoints0.append(corners0)
                imgpoints1.append(corners1)
                sample_i += 1
                print(sample_i)
            elif k == ord('q'):
                break

    cv2.destroyWindow('frame image')

    # Calibrate and rectify
    image_size = image0_grey.shape[::-1]

    flags = 0
    #flags |= cv2.CALIB_FIX_INTRINSIC
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_ASPECT_RATIO
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints0, imgpoints1, K0, D0, K1, D1, image_size, flags=flags, criteria=criteria)
    print('cv2.stereoCalibrate ret:', ret)

    flags = 0
    flags |= cv2.CALIB_ZERO_DISPARITY
    #alpha = 0.9
    alpha = 0.3
    R0, R1, P0, P1, Q, roi_left, roi_right = cv2.stereoRectify(
        K0, D0, K1, D1, image_size, R, T, flags=flags, alpha=alpha)

    return R, T, E, F, K0, K1, D0, D1, R0, R1, P0, P1, Q


def test_rectify(dv_address, dv_port0, dv_port1, K0, K1, D0, D1, R0, R1, P0, P1):

    # Connect to cameas
    f0 = dv.NetworkFrameInput(address=dv_address, port=dv_port0)
    f1 = dv.NetworkFrameInput(address=dv_address, port=dv_port1)

    with f0, f1:
        frame0 = next(f0)
        frame1 = next(f1)

        image0_rectified = np.zeros_like(frame0.image)
        image1_rectified = np.zeros_like(frame1.image)

        assert frame0.image.shape == frame1.image.shape

        while True:
            if frame0.timestamp < frame1.timestamp:
                frame0 = next(f0)
                image0 = frame0.image.copy()
                image0_shape = (image0.shape[1], image0.shape[0])
                mapx0, mapy0 = cv2.initUndistortRectifyMap(K0, D0, R0, P0, image0_shape, cv2.CV_32FC1)
                image0_rectified = cv2.remap(image0, mapx0, mapy0, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            else:
                frame1 = next(f1)
                image1 = frame1.image.copy()
                image1_shape = (image1.shape[1], image1.shape[0])
                mapx1, mapy1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image1_shape, cv2.CV_32FC1)
                image1_rectified = cv2.remap(image1, mapx1, mapy1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            image_rectified = np.append(image0_rectified, image1_rectified, axis=1)

            # Show frame
            cv2.imshow('frame image', image_rectified)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

    cv2.destroyWindow('frame image')


if __name__ == '__main__':

    # DVS camera servers
    dv_address = '127.0.0.1'
    dv_port0 = 36002
    dv_port1 = 36003

    # Calibrate file names
    date = time.strftime('%Y%m%d')
    path = f'./camera_calibration/{date}'
    K0_file_name = f'{path}/K0.npy'
    D0_file_name = f'{path}/D0.npy'
    R0_file_name = f'{path}/R0.npy'
    T0_file_name = f'{path}/T0.npy'
    K1_file_name = f'{path}/K1.npy'
    D1_file_name = f'{path}/D1.npy'
    R1_file_name = f'{path}/R1.npy'
    T1_file_name = f'{path}/T1.npy'

    # # Calibrate cameras
    # print('calibrating camera 0')
    # K0, D0, R0, T0 = calibrate_camera(dv_address, dv_port0)
    # print('calibrating camera 1')
    # K1, D1, R1, T1 = calibrate_camera(dv_address, dv_port1)

    # # Save camera calibrations
    # os.makedirs(path, exist_ok=True)
    # np.save(K0_file_name, K0)
    # np.save(D0_file_name, D0)
    # np.save(R0_file_name, R0)
    # np.save(T0_file_name, T0)
    # np.save(K1_file_name, K1)
    # np.save(D1_file_name, D1)
    # np.save(R1_file_name, R1)
    # np.save(T1_file_name, T1)

    # Load camera calibrations
    K0 = np.load(K0_file_name)
    D0 = np.load(D0_file_name)
    R0 = np.load(R0_file_name)
    T0 = np.load(T0_file_name)
    K1 = np.load(K1_file_name)
    D1 = np.load(D1_file_name)
    R1 = np.load(R1_file_name)
    T1 = np.load(T1_file_name)

    # # Test camera calibration
    # print('testing camera 0 calibration')
    # test_calibrate(dv_address, dv_port0, K0, D0)
    # print('testing camera 1 calibration')
    # test_calibrate(dv_address, dv_port1, K1, D1)

    # rectify file names
    date = time.strftime('%Y%m%d')
    path = f'./camera_rectify/{date}'
    R_file_name = f'{path}/R.npy'
    T_file_name = f'{path}/T.npy'
    E_file_name = f'{path}/E.npy'
    F_file_name = f'{path}/F.npy'
    K0_file_name = f'{path}/K0.npy'
    K1_file_name = f'{path}/K1.npy'
    D0_file_name = f'{path}/D0.npy'
    D1_file_name = f'{path}/D1.npy'
    R0_file_name = f'{path}/R0.npy'
    R1_file_name = f'{path}/R1.npy'
    P0_file_name = f'{path}/P0.npy'
    P1_file_name = f'{path}/P1.npy'
    Q_file_name = f'{path}/Q.npy'

    # Calibrate and rectify stereo cameras
    R, T, E, F, K0, K1, D0, D1, R0, R1, P0, P1, Q = rectify_cameras(
        dv_address, dv_port0, dv_port1, K0, K1, D0, D1)

    # Save camera calibrations
    os.makedirs(path, exist_ok=True)
    np.save(R_file_name, R)
    np.save(T_file_name, T)
    np.save(E_file_name, E)
    np.save(F_file_name, F)
    np.save(K0_file_name, K0)
    np.save(K1_file_name, K1)
    np.save(D0_file_name, D0)
    np.save(D1_file_name, D1)
    np.save(R0_file_name, R0)
    np.save(R1_file_name, R1)
    np.save(P0_file_name, P0)
    np.save(P1_file_name, P1)
    np.save(Q_file_name, Q)

    # Load camera calibrations
    os.makedirs(path, exist_ok=True)
    R = np.load(R_file_name)
    T = np.load(T_file_name)
    E = np.load(E_file_name)
    F = np.load(F_file_name)
    K0 = np.load(K0_file_name)
    K1 = np.load(K1_file_name)
    D0 = np.load(D0_file_name)
    D1 = np.load(D1_file_name)
    R0 = np.load(R0_file_name)
    R1 = np.load(R1_file_name)
    P0 = np.load(P0_file_name)
    P1 = np.load(P1_file_name)
    Q = np.load(Q_file_name)

    # Test camera stereo rectification
    print('testing stereo camera rectification')
    test_rectify(dv_address, dv_port0, dv_port1, K0, K1, D0, D1, R0, R1, P0, P1)
