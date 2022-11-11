import numpy as np
import cv2


def view_frames(f, height, width):
    frame = np.empty((height, width * 2, 3), dtype=np.uint8)

    for pos_L, neg_L, pos_R, neg_R in f:
        frame[:, :width, 0] = pos_L
        frame[:, :width, 2] = neg_L
        frame[:, width:, 0] = pos_R
        frame[:, width:, 2] = neg_R
        frame[frame > 0] = 255

        cv2.imshow("Input Frame", frame)
        k = cv2.waitKey(0)
        if k == ord("q"):
            cv2.destroyWindow("Input Frame")
            break
