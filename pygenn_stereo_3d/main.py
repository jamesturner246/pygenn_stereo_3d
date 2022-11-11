import numpy as np
import cv2
import pygenn

from drds import drds_generator
from aedat import aedat_generator
from utils import view_frames
from models import *


def main(aedat_file_path=None,
         calibration_path=None,
         cam_height=260, cam_width=346,
         height=260, width=346,
         skip_usec=1600000,
         interval_usec=5000,
         model_name="pygenn_stereo_3d",
         model_fp_type="float",
         model_dt=1.0,
         record_video=False):


    #disparity_scale = 1.0
    #disparity_scale = 0.5
    disparity_scale = 0.25 # WORKS WELL
    #disparity_scale = 0.1


    # DRDS frame generator
    f = drds_generator(
        "./MM_gnd.png", disparity_scale=disparity_scale, size=(height, width),
        total_time_msec=1000, update_frequency=100, p_flip=0.2)

    # # AEDAT frame generator
    # f = aedat_generator(
    #     aedat_file_path, calibration_path,
    #     cam_size=(cam_height, cam_width), size=(height, width),
    #     skip_usec=skip_usec, interval_usec=interval_usec)




    # # DEBUG: view event frames
    # view_frames(f, height, width)
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
        "retina_pos_L", height * width, spike_input, retina_params, retina_vars)
    retina_neg_L_nrn = model.add_neuron_population(
        "retina_neg_L", height * width, spike_input, retina_params, retina_vars)
    retina_pos_R_nrn = model.add_neuron_population(
        "retina_pos_R", height * width, spike_input, retina_params, retina_vars)
    retina_neg_R_nrn = model.add_neuron_population(
        "retina_neg_R", height * width, spike_input, retina_params, retina_vars)

    # Coincidence detectors
    coincidence_params = {
        "V_thres": coincidence_V_thres,
        "tau": coincidence_tau,
    }
    coincidence_vars = {"V": 0.0}
    coincidence_pos_nrn = model.add_neuron_population(
        "coincidence_pos", height * width**2, LIF, coincidence_params, coincidence_vars)
    coincidence_neg_nrn = model.add_neuron_population(
        "coincidence_neg", height * width**2, LIF, coincidence_params, coincidence_vars)

    # Disparity detectors
    disparity_params = {
        "V_thres": disparity_V_thres,
        "tau": disparity_tau,
    }
    disparity_vars = {"V": 0.0}
    disparity_nrn = model.add_neuron_population(
        "disparity", height * width**2, LIF, disparity_params, disparity_vars)

    ### SYNAPSES ###
    ################

    # Left Retina -> Coincidence
    conn_init = pygenn.genn_model.init_connectivity(
        retina_L_coincidence_syn_init,
        {"height": height, "width": width})
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
        {"height": height, "width": width})
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
        {"height": height, "width": width, "receptive_distance": receptive_distance})
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
        {"height": height, "width": width, "receptive_distance": receptive_distance})
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
        {"height": height, "width": width})
    disparity_disparity_inh_syn = model.add_synapse_population(
        "disparity_disparity_inh", "PROCEDURAL_GLOBALG", 0, disparity_nrn, disparity_nrn,
        disparity_disparity_inh_weight_update, {}, {}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Build and load GeNN model
    model.build()
    model.load()


    # Set retinae inputs
    def set_input(pos_L, neg_L, pos_R, neg_R):
        retina_pos_L_nrn.vars["input"].view.reshape(height, width)[:] = pos_L
        retina_pos_L_nrn.push_var_to_device("input")
        retina_neg_L_nrn.vars["input"].view.reshape(height, width)[:] = neg_L
        retina_neg_L_nrn.push_var_to_device("input")
        retina_pos_R_nrn.vars["input"].view.reshape(height, width)[:] = pos_R
        retina_pos_R_nrn.push_var_to_device("input")
        retina_neg_R_nrn.vars["input"].view.reshape(height, width)[:] = neg_R
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
        i = coordinates["y"] * width**2 + coordinates["x_L"] * width + coordinates["x_R"]
        return i

    # Neuron index to coordinates
    def index_to_coordinates(i):
        coordinates = {}
        coordinates["y"] = (i // width) // width
        coordinates["x_L"] = (i // width) % width
        coordinates["x_R"] = i % width
        return coordinates

    # Process output spikes
    def process_polarised_spikes(spikes):

        # Convert spikes to event coordinates and disparity.
        c = index_to_coordinates(spikes)
        d = np.absolute(c["x_L"] - c["x_R"])

        # Clip events' disparity to range of interest.
        d = d - disparity_offset
        d[d < 0] = 0
        d[d >= disparity_range] = disparity_range - 1



        # TODO: DEBUG POS/NEG DISPARITY OVERWRITING

        # print()
        # print()
        # print()
        # if spikes is pos_spikes:
        #     print("POSITIVE")
        # elif spikes is neg_spikes:
        #     print("NEGATIVE")

        # if stereo_disparity_L[c["y"], c["x_L"]] == -1:
        #     print("GOOD")
        # else:
        #     print("BAD")
        #     print("  A", stereo_disparity_L[c["y"], c["x_L"]])
        #     print("  B", d)



        # Record event disparity.
        stereo_disparity_L[c["y"], c["x_L"]] = d
        stereo_disparity_R[c["y"], c["x_R"]] = d
        cyclopean_disparity[c["y"], (c["x_L"] + c["x_R"]) // 2] = d



        # TODO: DRDS ACCUMULATOR EXPERIMENTS
        accumulator[c["y"], (c["x_L"] + c["x_R"]) // 2, d] += 1





    # Output frames
    stereo_disparity = np.empty((height, width * 2), dtype=np.int32)
    stereo_disparity_L = stereo_disparity[:height, :width]
    stereo_disparity_R = stereo_disparity[:height, width:]
    stereo_image = np.empty((height, width * 2, 3), dtype=np.float32)
    stereo_image_L = stereo_image[:height, :width]
    stereo_image_R = stereo_image[:height, width:]
    cyclopean_disparity = np.empty((height, width), dtype=np.int32)
    cyclopean_image = np.empty((height, width, 3), dtype=np.float32)

    # Tune disparity range for plot


    # # TODO: DRDS (disparity_scale = 1.0)
    # disparity_lo = 56
    # disparity_hi = 196

    # # TODO: DRDS (disparity_scale = 0.5)
    # disparity_lo = 28
    # disparity_hi = 98

    # TODO: DRDS (disparity_scale = 0.25)
    disparity_lo = 14
    disparity_hi = 49

    # # TODO: DRDS (disparity_scale = 0.1)
    # disparity_lo = 6
    # disparity_hi = 20


    disparity_offset = disparity_lo
    disparity_range = disparity_hi + 1 - disparity_offset

    # Colour mapping for plot
    cmap_B = np.linspace(1.0, 0.0, disparity_range, dtype=np.float32)
    cmap_G = np.zeros(disparity_range, dtype=np.float32)
    cmap_R = np.linspace(0.0, 1.0, disparity_range, dtype=np.float32)
    cmap = np.vstack([cmap_B, cmap_G, cmap_R]).T

    # Create output video files
    if record_video:
        stereo_video_file_name = "stereo_video.avi"
        stereo_video_file = cv2.VideoWriter(
            stereo_video_file_name, cv2.VideoWriter_fourcc(*"MJPG"),
            1000000 / interval_usec, (width * 2, height))
        cyclopean_video_file_name = "cyclopean_video.avi"
        cyclopean_video_file = cv2.VideoWriter(
            cyclopean_video_file_name, cv2.VideoWriter_fourcc(*"MJPG"),
            1000000 / interval_usec, (width, height))






    # TODO: DRDS ACCUMULATOR EXPERIMENTS
    accumulator = np.zeros((height, width, disparity_range), dtype=np.int64)





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

        # Filter spikes
        pos_spikes = np.intersect1d(coincidence_pos_spikes, disparity_spikes)
        neg_spikes = np.intersect1d(coincidence_neg_spikes, disparity_spikes)

        # Clear disparity frames (no events where disparity == -1)
        stereo_disparity.fill(-1)
        cyclopean_disparity.fill(-1)





        # TODO: ORIGINAL CODE: THIS WORKS FOR THE AEDAT FILE IN THE REPO

        # # Tune plot range
        # disparity_offset = 10
        # disparity_range = 10

        # # Plot colour mapping
        # cmap_B = np.linspace(0.0, 1.0, disparity_range, dtype=np.float32)
        # cmap_G = np.zeros(disparity_range, dtype=np.float32)
        # cmap_R = np.linspace(1.0, 0.0, disparity_range, dtype=np.float32)
        # cmap = np.vstack([cmap_B, cmap_G, cmap_R]).T

        # # Clear output frames
        # stereo_image.fill(1.0)
        # cyclopean_image.fill(1.0)

        # for spike_i in pos_spikes:
        #     c = index_to_coordinates(spike_i)
        #     d = np.absolute(c["x_L"] - c["x_R"])
        #     d = disparity_offset + (disparity_range - d)
        #     if d < 0:
        #         d = 0
        #     elif d >= disparity_range:
        #         d = disparity_range - 1
        #     stereo_image_L[c["y"], c["x_L"]] = cmap[d]
        #     stereo_image_R[c["y"], c["x_R"]] = cmap[d]

        # for spike_i in neg_spikes:
        #     c = index_to_coordinates(spike_i)
        #     d = np.absolute(c["x_L"] - c["x_R"])
        #     d = disparity_offset + (disparity_range - d)
        #     if d < 0:
        #         d = 0
        #     elif d >= disparity_range:
        #         d = disparity_range - 1
        #     stereo_image_L[c["y"], c["x_L"]] = cmap[d]
        #     stereo_image_R[c["y"], c["x_R"]] = cmap[d]









        process_polarised_spikes(pos_spikes)
        process_polarised_spikes(neg_spikes)





        # TODO: HANDLE CLOSE AND FAR EVENTS OVERWRITING EACH OTHER
        # IT HAPPENS, BUT MOSTLY DOESN'T

        # WHICH DISPARITY DOMINATES, IF ANY ???
        # WHICH DISPARITY *SHOULD* DOMINATE ???






        # Compute stereo images (where disparity is not -1).
        stereo_image.fill(1.0)
        idx = stereo_disparity_L != -1
        stereo_image_L[idx] = cmap[stereo_disparity_L[idx]]
        idx = stereo_disparity_R != -1
        stereo_image_R[idx] = cmap[stereo_disparity_R[idx]]

        # Compute cyclopean image (where disparity is not -1).
        cyclopean_image.fill(1.0)
        idx = cyclopean_disparity != -1
        cyclopean_image[idx] = cmap[cyclopean_disparity[idx]]

        # Record output frames
        if record_video:
            stereo_video_frame = (stereo_image * 255).astype(np.uint8)
            stereo_video_file.write(stereo_video_frame)
            cyclopean_video_frame = (cyclopean_image * 255).astype(np.uint8)
            cyclopean_video_file.write(cyclopean_video_frame)

        # Plot outputs and handle interrupt
        cv2.imshow("Stereo Output", stereo_image)
        cv2.imshow("Cyclopean Output", cyclopean_image)
        k = cv2.waitKey(1)
        if k == ord("q"):
            cv2.destroyWindow("Stereo Output")
            cv2.destroyWindow("Cyclopean Output")
            exit(0)

    # Release output video files
    if record_video:
        stereo_video_file.release()
        cyclopean_video_file.release()




    # TODO: DRDS ACCUMULATOR EXPERIMENTS



    import matplotlib.pyplot as plt
    import matplotlib.cm as cm



    print(disparity_range)


    accumulator_bound = accumulator.max() + 1


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    colours = cm.hot(np.linspace(0, 1, accumulator_bound))

    for i in range(1, accumulator_bound):
    #for i in range(1, 10):
    #for i in range(20, accumulator_bound):
        row, col, depth = np.nonzero(accumulator == i)
        ax.scatter(row, col, depth, marker=".", color=colours[i])

    plt.show()




    # # TODO: THE BELOW STATS WON'T WORK
    # # WE WANT MORE LIKE A PIXELWISE HISTOGRAM OR 3D SCATTER PLOT

    # plt.figure()
    # accumulator_min = accumulator.min(axis=2)
    # plt.imshow(accumulator_min)
    # plt.title("Disparity Min")
    # plt.colorbar()

    # plt.figure()
    # accumulator_max = accumulator.max(axis=2)
    # plt.imshow(accumulator_max)
    # plt.title("Disparity Max")
    # plt.colorbar()

    # plt.figure()
    # accumulator_mean = accumulator.mean(axis=2)
    # plt.imshow(accumulator_mean)
    # plt.title("Disparity Mean")
    # plt.colorbar()

    # plt.figure()
    # accumulator_std = accumulator.std(axis=2)
    # plt.imshow(accumulator_std)
    # plt.title("Disparity Std")
    # plt.colorbar()

    # plt.show()






if __name__  == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="pygenn_stereo_3d")

    # Input parameters
    parser.add_argument("--aedat-file-path", type=str, default=None)
    parser.add_argument("--calibration-path", type=str, default=None)
    parser.add_argument("--cam-height", type=int, default=260)
    parser.add_argument("--cam-width", type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
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
