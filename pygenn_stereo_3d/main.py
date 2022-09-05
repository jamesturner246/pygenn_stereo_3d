import numpy as np
import cv2
import pygenn

from aedat import aedat_read_frames, aedat_view_frames
from models import *


def main(aedat_file_path,
         calibration_path="./calibration",
         cam_height=260,
         cam_width=346,
         cam_resize_height=260,
         cam_resize_width=346,
         skip_usec=1600000,
         interval_usec=5000,
         record_video=False,
         model_name="pygenn_stereo_3d",
         model_fp_type="float",
         model_dt=1.0):

    # aedat frame generator
    f = aedat_read_frames(
        aedat_file_path, calibration_path,
        cam_height=cam_height, cam_width=cam_width,
        cam_resize_height=cam_resize_height, cam_resize_width=cam_resize_width,
        skip_usec=skip_usec, interval_usec=interval_usec)

    # # DEBUG: view aedat frames
    # aedat_view_frames(f, cam_resize_height, cam_resize_width)
    # exit(0)

    # Receptive field distance
    receptive_distance = 25

    # Coincidence Detectors
    coincidence_V_thres = 1.0
    coincidence_tau = 1.4
    retina_coincidence_W = 0.65 # tune such that 1 < (V_thres / W) <= 2
    assert 1 < (coincidence_V_thres / retina_coincidence_W) <= 2

    coincidence_sensitivity = coincidence_tau * np.log(
        1 / ((coincidence_V_thres / retina_coincidence_W) - 1))
    print("coincidence sensitivity:", coincidence_sensitivity)

    # Disparity Detectors
    disparity_V_thres = 1.0
    disparity_tau = 14.0 # significantly larger than coincidence_tau
    coincidence_disparity_exc_W = 100.0 / (((receptive_distance * 2.0) + 1.0) ** 2.0)
    coincidence_disparity_inh_W = -coincidence_disparity_exc_W


    # Construct model
    model = pygenn.GeNNModel(model_fp_type, model_name)
    model.dT = model_dt

    ### NEURONS ###
    ###############

    # Retinae (inputs)
    retina_params = {}
    retina_vars = {"input": 0.0}
    retina_pos_L_nrn = model.add_neuron_population(
        "retina_pos_L", cam_resize_height * cam_resize_width, spike_input, retina_params, retina_vars)
    retina_neg_L_nrn = model.add_neuron_population(
        "retina_neg_L", cam_resize_height * cam_resize_width, spike_input, retina_params, retina_vars)
    retina_pos_R_nrn = model.add_neuron_population(
        "retina_pos_R", cam_resize_height * cam_resize_width, spike_input, retina_params, retina_vars)
    retina_neg_R_nrn = model.add_neuron_population(
        "retina_neg_R", cam_resize_height * cam_resize_width, spike_input, retina_params, retina_vars)

    # Coincidence detectors
    coincidence_params = {
        "V_thres": coincidence_V_thres,
        "tau": coincidence_tau,
    }
    coincidence_vars = {"V": 0.0}
    coincidence_pos_nrn = model.add_neuron_population(
        "coincidence_pos", cam_resize_height * cam_resize_width**2, LIF, coincidence_params, coincidence_vars)
    coincidence_neg_nrn = model.add_neuron_population(
        "coincidence_neg", cam_resize_height * cam_resize_width**2, LIF, coincidence_params, coincidence_vars)

    # Disparity detectors
    disparity_params = {
        "V_thres": disparity_V_thres,
        "tau": disparity_tau,
    }
    disparity_vars = {"V": 0.0}
    disparity_nrn = model.add_neuron_population(
        "disparity", cam_resize_height * cam_resize_width**2, LIF, disparity_params, disparity_vars)

    ### SYNAPSES ###
    ################

    # Left Retina -> Coincidence
    conn_init = pygenn.genn_model.init_connectivity(
        retina_L_coincidence_syn_init,
        {"cam_height": cam_resize_height, "cam_width": cam_resize_width})
    retina_pos_L_coincidence_pos_syn = model.add_synapse_population(
        "retina_pos_L_coincidence_pos", "PROCEDURAL_GLOBALG", 0, retina_pos_L_nrn, coincidence_pos_nrn,
        "StaticPulse", {}, {"g": retina_coincidence_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)
    retina_neg_L_coincidence_neg_syn = model.add_synapse_population(
        "retina_neg_L_coincidence_neg", "PROCEDURAL_GLOBALG", 0, retina_neg_L_nrn, coincidence_neg_nrn,
        "StaticPulse", {}, {"g": retina_coincidence_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Right Retina -> Coincidence
    conn_init = pygenn.genn_model.init_connectivity(
        retina_R_coincidence_syn_init,
        {"cam_height": cam_resize_height, "cam_width": cam_resize_width})
    retina_pos_R_coincidence_pos_syn = model.add_synapse_population(
        "retina_pos_R_coincidence_pos", "PROCEDURAL_GLOBALG", 0, retina_pos_R_nrn, coincidence_pos_nrn,
        "StaticPulse", {}, {"g": retina_coincidence_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)
    retina_neg_R_coincidence_neg_syn = model.add_synapse_population(
        "retina_neg_R_coincidence_neg", "PROCEDURAL_GLOBALG", 0, retina_neg_R_nrn, coincidence_neg_nrn,
        "StaticPulse", {}, {"g": retina_coincidence_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Excitatory Coincidence -> Disparity (plane of constant disparity)
    conn_init = pygenn.genn_model.init_connectivity(
        coincidence_disparity_exc_syn_init,
        {"cam_height": cam_resize_height, "cam_width": cam_resize_width, "receptive_distance": receptive_distance})
    coincidence_pos_disparity_exc_syn = model.add_synapse_population(
        "coincidence_pos_disparity_exc", "PROCEDURAL_GLOBALG", 0, coincidence_pos_nrn, disparity_nrn,
        "StaticPulse", {}, {"g": coincidence_disparity_exc_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)
    coincidence_neg_disparity_exc_syn = model.add_synapse_population(
        "coincidence_neg_disparity_exc", "PROCEDURAL_GLOBALG", 0, coincidence_neg_nrn, disparity_nrn,
        "StaticPulse", {}, {"g": coincidence_disparity_exc_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Inhibitory Coincidence -> Disparity (plane of constant horizontal cyclopean position)
    conn_init = pygenn.genn_model.init_connectivity(
        coincidence_disparity_inh_syn_init,
        {"cam_height": cam_resize_height, "cam_width": cam_resize_width, "receptive_distance": receptive_distance})
    coincidence_pos_disparity_inh_syn = model.add_synapse_population(
        "coincidence_pos_disparity_inh", "PROCEDURAL_GLOBALG", 0, coincidence_pos_nrn, disparity_nrn,
        "StaticPulse", {}, {"g": coincidence_disparity_inh_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)
    coincidence_neg_disparity_inh_syn = model.add_synapse_population(
        "coincidence_neg_disparity_inh", "PROCEDURAL_GLOBALG", 0, coincidence_neg_nrn, disparity_nrn,
        "StaticPulse", {}, {"g": coincidence_disparity_inh_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Inhibitory Recurrent Disparity -> Disparity
    conn_init = pygenn.genn_model.init_connectivity(
        disparity_disparity_inh_syn_init,
        {"cam_height": cam_resize_height, "cam_width": cam_resize_width})
    disparity_disparity_inh_syn = model.add_synapse_population(
        "disparity_disparity_inh", "PROCEDURAL_GLOBALG", 0, disparity_nrn, disparity_nrn,
        disparity_disparity_inh_weight_update, {}, {}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Build and load GeNN model
    model.build()
    model.load()


    # Set retinae inputs
    def set_input(pos_L, neg_L, pos_R, neg_R):
        retina_pos_L_nrn.vars["input"].view.reshape(cam_resize_height, cam_resize_width)[:] = pos_L
        retina_pos_L_nrn.push_var_to_device("input")
        retina_neg_L_nrn.vars["input"].view.reshape(cam_resize_height, cam_resize_width)[:] = neg_L
        retina_neg_L_nrn.push_var_to_device("input")
        retina_pos_R_nrn.vars["input"].view.reshape(cam_resize_height, cam_resize_width)[:] = pos_R
        retina_pos_R_nrn.push_var_to_device("input")
        retina_neg_R_nrn.vars["input"].view.reshape(cam_resize_height, cam_resize_width)[:] = neg_R
        retina_neg_R_nrn.push_var_to_device("input")

    # Coordinate mapping
    def coordinates_to_disparity_space(x_L, x_R, y):
        x = x_R + x_L
        d = x_R - x_L
        return x, y, d

    # Inverse coordinate mapping
    def disparity_space_to_coordinate(x, y, d):
        pass

    # Neuron coordinates to index
    def coordinates_to_index(coordinates):
        i = coordinates["y"] * cam_resize_width**2 + coordinates["x_L"] * cam_resize_width + coordinates["x_R"]
        return i

    # Neuron index to coordinates
    def index_to_coordinates(i):
        coordinates = {}
        coordinates["y"] = (i // cam_resize_width) // cam_resize_width
        coordinates["x_L"] = (i // cam_resize_width) % cam_resize_width
        coordinates["x_R"] = i % cam_resize_width
        return coordinates


    # Output frames
    output = np.empty((cam_resize_height, cam_resize_width * 2, 3), dtype="float32")
    output_L = output[:cam_resize_height, :cam_resize_width]
    output_R = output[:cam_resize_height, cam_resize_width:]

    if record_video:
        video_file_name = "stereo_3d_video.avi"
        video_file = cv2.VideoWriter(
            video_file_name, cv2.VideoWriter_fourcc(*"MJPG"),
            1000000 / interval_usec, (cam_resize_width * 2, cam_resize_height))

    # Simulate model
    for pos_L, neg_L, pos_R, neg_R in f:

        # Step time
        set_input(pos_L, neg_L, pos_R, neg_R)
        model.step_time()

        # Get spikes
        coincidence_pos_nrn.pull_current_spikes_from_device()
        coincidence_neg_nrn.pull_current_spikes_from_device()
        disparity_nrn.pull_current_spikes_from_device()
        coincidence_pos_spikes = coincidence_pos_nrn.current_spikes
        coincidence_neg_spikes = coincidence_neg_nrn.current_spikes
        disparity_spikes = disparity_nrn.current_spikes

        # Filter spikes (handles broadly tuned cyclopean distance)
        pos_spikes = np.intersect1d(coincidence_pos_spikes, disparity_spikes)
        neg_spikes = np.intersect1d(coincidence_neg_spikes, disparity_spikes)

        # Tune plot range
        event_offset = 10
        event_range = 10

        # Plot colour mapping
        cmap_B = np.linspace(0.0, 1.0, event_range, dtype="float32")
        cmap_G = np.zeros(event_range, dtype="float32")
        cmap_R = np.linspace(1.0, 0.0, event_range, dtype="float32")
        cmap = np.vstack([cmap_B, cmap_G, cmap_R]).T

        output_L.fill(1.0)
        output_R.fill(1.0)

        for spike_i in pos_spikes:
            c = index_to_coordinates(spike_i)
            d = np.abs(c["x_R"] - c["x_L"])
            event_distance = event_offset + (event_range - d)
            if event_distance < 0:
                event_distance = 0
            elif event_distance >= event_range:
                event_distance = event_range - 1
            output_L[c["y"], c["x_L"]] = cmap[event_distance]
            output_R[c["y"], c["x_R"]] = cmap[event_distance]

        for spike_i in neg_spikes:
            c = index_to_coordinates(spike_i)
            d = np.abs(c["x_R"] - c["x_L"])
            event_distance = event_offset + (event_range - d)
            if event_distance < 0:
                event_distance = 0
            elif event_distance >= event_range:
                event_distance = event_range - 1
            output_L[c["y"], c["x_L"]] = cmap[event_distance]
            output_R[c["y"], c["x_R"]] = cmap[event_distance]

        if record_video:
            video_frame = (output * 255).astype("uint8")
            video_file.write(video_frame)

        cv2.imshow("output", output)
        k = cv2.waitKey(1)
        if k == ord("q"):
            cv2.destroyWindow("output")
            exit(0)

    if record_video:
        video_file.release()


if __name__  == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="pygenn_stereo_3d")

    # Recording parameters
    parser.add_argument("aedat_file_path", type=str)
    parser.add_argument("--calibration-path", type=str, default="./calibration")
    parser.add_argument("--cam-height", type=int, default=260)
    parser.add_argument("--cam-width", type=int, default=346)
    parser.add_argument("--cam-resize-height", type=int, default=260)
    parser.add_argument("--cam-resize-width", type=int, default=346)
    parser.add_argument("--skip-usec", type=int, default=1600000)
    parser.add_argument("--interval-usec", type=int, default=5000)
    parser.add_argument("--record-video", type=bool, default=False)

    # GeNN model parameters
    parser.add_argument("--model-name", type=str, default="pygenn_stereo_3d")
    parser.add_argument("--model-fp-type", type=str, default="float", choices=["float", "double"])
    parser.add_argument("--model-dt", type=float, default=1.0)

    # Parse arguments and run
    args = parser.parse_args()
    main(**vars(args))
