import PIL.Image
import numpy as np


def _compute_disparity_shift_matrices(disparity_image):
    # There will be 'shadows' of undefined mappings where higher-disparity
    # pixels are shifted far, while lower-disparity pixels are not shifted
    # far enough to fill the gaps.

    # If multiple pixels are mapped to the same coordinates, then the closer
    # (higher-disparity) pixel mapping is preserved, while the lower-disparity
    # mapping is left undefined.

    # The 'undefined' mapping value is -1.

    shift_x_L = np.full(disparity_image.shape, -1, dtype=np.int32)
    shift_x_R = np.full(disparity_image.shape, -1, dtype=np.int32)
    visible_L = np.full(disparity_image.shape, True, dtype=bool)
    visible_R = np.full(disparity_image.shape, True, dtype=bool)

    # From nearest to furthest (highest to lowest disparity)
    for d in np.flip(np.unique(disparity_image)):
        d_y_all, d_x_all = np.nonzero(disparity_image == d)

        # Left frame (shifted right by disparity)
        d_x_L = d_x_all + d // 2
        d_in_frame_L = d_x_L < disparity_image.shape[1]
        d_y = d_y_all[d_in_frame_L]
        d_x = d_x_all[d_in_frame_L]
        d_x_L = d_x_L[d_in_frame_L]

        d_visible_L = visible_L[d_y, d_x_L]
        d_y = d_y[d_visible_L]
        d_x = d_x[d_visible_L]
        d_x_L = d_x_L[d_visible_L]

        shift_x_L[d_y, d_x] = d_x_L
        visible_L[d_y, d_x_L] = False

        # Right frame (shifted left by disparity)
        d_x_R = d_x_all - d // 2
        d_in_frame_R = d_x_R >= 0
        d_y = d_y_all[d_in_frame_R]
        d_x = d_x_all[d_in_frame_R]
        d_x_R = d_x_R[d_in_frame_R]

        d_visible_R = visible_R[d_y, d_x_R]
        d_y = d_y[d_visible_R]
        d_x = d_x[d_visible_R]
        d_x_R = d_x_R[d_visible_R]

        shift_x_R[d_y, d_x] = d_x_R
        visible_R[d_y, d_x_R] = False

    return shift_x_L, shift_x_R


def _project_events(retina, events, shift_x):
    # Project events onto retinae.
    # Ignores undefined (-1) mappings.
    retina.fill(False)
    y_all, x_all = np.nonzero(events)
    defined = shift_x[y_all, x_all] != -1
    y = y_all[defined]
    x = shift_x[y, x_all[defined]]
    retina[y, x] = True


def drds_generator(
        disparity_image_path, disparity_scale=1.0, size=(260, 346),
        total_time_msec=1000, update_frequency=100, p_flip=0.2):

    # Disparity image and X coordinate shift matrices.
    disparity_image = PIL.Image.open(disparity_image_path)
    disparity_image = disparity_image.resize(np.flip(size), resample=PIL.Image.NEAREST)
    disparity_image = np.array(disparity_image)
    disparity_image = np.rint(disparity_image * disparity_scale).astype(np.uint8)
    shift_x_L, shift_x_R = _compute_disparity_shift_matrices(disparity_image)

    print("Disparity image statistics")
    print("Min:  ", disparity_image.min())
    print("Max:  ", disparity_image.max())
    print("Mean: ", disparity_image.mean())
    print("Std:  ", disparity_image.std())

    # Random noise frame.
    noise_frame = np.random.choice([False, True], size=size, p=[0.5, 0.5])
    pos_events = np.empty_like(noise_frame)
    neg_events = np.empty_like(noise_frame)

    # Retina frames.
    pos_L = np.empty_like(noise_frame)
    neg_L = np.empty_like(noise_frame)
    pos_R = np.empty_like(noise_frame)
    neg_R = np.empty_like(noise_frame)

    # For each frame until end time:
    update_time_msec = 1000 // update_frequency
    time_msec = 0

    while time_msec < total_time_msec:
        time_msec += update_time_msec

        # Update noise frame.
        flip_mask = np.random.choice([False, True], size=size, p=[1.0-p_flip, p_flip])
        np.logical_xor(noise_frame, flip_mask, out=noise_frame)

        # Map pixel ON events.
        np.logical_and(noise_frame, flip_mask, out=pos_events)
        _project_events(pos_L, pos_events, shift_x_L)
        _project_events(pos_R, pos_events, shift_x_R)

        # Map pixel OFF events.
        np.logical_and(~noise_frame, flip_mask, out=neg_events)
        _project_events(neg_L, neg_events, shift_x_L)
        _project_events(neg_R, neg_events, shift_x_R)

        yield pos_L, neg_L, pos_R, neg_R
