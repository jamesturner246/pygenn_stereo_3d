import numpy as np
import cv2
import pygenn

from aedat_read import aedat_read, view_aedat_frames
from models import *


def main():

    #dt = 1.0
    dt = 10.0

    cam_height = 260
    cam_width = 346

    #conn_matrix = "SPARSE_GLOBALG"
    conn_matrix = "PROCEDURAL_GLOBALG"

    f = aedat_read(
        #"stereo_tests-2022_05_19_16_14_51.aedat4",
        #"stereo_tests-2022_05_19_16_15_00.aedat4",
        #"stereo_tests-2022_05_19_16_15_10.aedat4",
        #"stereo_tests-2022_05_19_16_20_35.aedat4",
        "stereo_tests-2022_05_24_14_11_01.aedat4",
        "./camera_rectify/20220519",
        interval_usec=dt*1000)

    # # DEBUG: view aedat frames
    # view_aedat_frames(f)
    # exit(0)





    # TODO: what should params be?

    # TODO: coincidence detector weight threshold ratio
    # The ratio thres / w for coincidence detectors is chosen so that while
    # two retinal spikes can trigger a response, a single spike will not.
    # so maybe thres=1.5 and w=1.0
    # time constant selection notes follow, after eqn. 5 in supplementary notes

    # AMPA SYNAPSE W SHOULD BE LOW, NMDA SYNAPSE W SHOULD BE HIGH

    # Disparity Detectors: broad tuning to horizontal cyclopean position and fine tuning to disparity




    # Coincidence Detectors
    coincidence_V_thres = 1.5
    coincidence_tau = 2.0
    retina_coincidence_W = 1.0
    assert 1 < (coincidence_V_thres / retina_coincidence_W) <= 2 # Tune such that 1 < (V_thres / W) <= 2
    coincidence_sensitivity = coincidence_tau * np.log(1 / ((coincidence_V_thres / retina_coincidence_W) - 1))
    print("coincidence sensitivity:", coincidence_sensitivity)

    # Disparity Detectors
    disparity_V_thres = 1.5
    disparity_tau = 5.0
    coincidence_disparity_exc_W = 1.0
    coincidence_disparity_inh_W = -coincidence_disparity_exc_W
    disparity_disparity_inh_W = -disparity_V_thres

    # Receptive field distance
    #window = 13
    window = 21
    assert window % 2 == 1
    receptive_distance = int((window - 1) / 2)



    # Construct model
    model = pygenn.GeNNModel("float", "stereo_3d")
    model.dT = dt


    ### NEURONS ###
    ###############

    # Retinae (inputs)
    retina_params = {}
    retina_vars = {"input": 0.0}
    retina_pos_L_nrn = model.add_neuron_population(
        "retina_pos_L", cam_height * cam_width, spike_input, retina_params, retina_vars)
    retina_neg_L_nrn = model.add_neuron_population(
        "retina_neg_L", cam_height * cam_width, spike_input, retina_params, retina_vars)
    retina_pos_R_nrn = model.add_neuron_population(
        "retina_pos_R", cam_height * cam_width, spike_input, retina_params, retina_vars)
    retina_neg_R_nrn = model.add_neuron_population(
        "retina_neg_R", cam_height * cam_width, spike_input, retina_params, retina_vars)

    # Coincidence detectors
    coincidence_params = {
        "V_thres": coincidence_V_thres,
        "tau": coincidence_tau,
    }
    coincidence_vars = {"V": 0.0}
    coincidence_pos_nrn = model.add_neuron_population(
        "coincidence_pos", cam_height * cam_width**2, LIF, coincidence_params, coincidence_vars)
    coincidence_neg_nrn = model.add_neuron_population(
        "coincidence_neg", cam_height * cam_width**2, LIF, coincidence_params, coincidence_vars)

    # Disparity detectors
    disparity_params = {
        "V_thres": disparity_V_thres,
        "tau": disparity_tau,
    }
    disparity_vars = {"V": 0.0}
    disparity_nrn = model.add_neuron_population(
        "disparity", cam_height * cam_width**2, LIF, disparity_params, disparity_vars)


    ### SYNAPSES ###
    ################

    # Left Retina -> Coincidence
    conn_init = pygenn.genn_model.init_connectivity(
        retina_L_coincidence_syn_init,
        {"cam_height": cam_height, "cam_width": cam_width})
    retina_pos_L_coincidence_pos_syn = model.add_synapse_population(
        "retina_pos_L_coincidence_pos", conn_matrix, 0, retina_pos_L_nrn, coincidence_pos_nrn,
        "StaticPulse", {}, {"g": retina_coincidence_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)
    retina_neg_L_coincidence_neg_syn = model.add_synapse_population(
        "retina_neg_L_coincidence_neg", conn_matrix, 0, retina_neg_L_nrn, coincidence_neg_nrn,
        "StaticPulse", {}, {"g": retina_coincidence_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Right Retina -> Coincidence
    conn_init = pygenn.genn_model.init_connectivity(
        retina_R_coincidence_syn_init,
        {"cam_height": cam_height, "cam_width": cam_width})
    retina_pos_R_coincidence_pos_syn = model.add_synapse_population(
        "retina_pos_R_coincidence_pos", conn_matrix, 0, retina_pos_R_nrn, coincidence_pos_nrn,
        "StaticPulse", {}, {"g": retina_coincidence_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)
    retina_neg_R_coincidence_neg_syn = model.add_synapse_population(
        "retina_neg_R_coincidence_neg", conn_matrix, 0, retina_neg_R_nrn, coincidence_neg_nrn,
        "StaticPulse", {}, {"g": retina_coincidence_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Excitatory Coincidence -> Disparity (plane of constant disparity)
    conn_init = pygenn.genn_model.init_connectivity(
        coincidence_disparity_exc_syn_init,
        {"cam_height": cam_height, "cam_width": cam_width, "receptive_distance": receptive_distance})
    coincidence_pos_disparity_exc_syn = model.add_synapse_population(
        "coincidence_pos_disparity_exc", conn_matrix, 0, coincidence_pos_nrn, disparity_nrn,
        "StaticPulse", {}, {"g": coincidence_disparity_exc_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)
    coincidence_neg_disparity_exc_syn = model.add_synapse_population(
        "coincidence_neg_disparity_exc", conn_matrix, 0, coincidence_neg_nrn, disparity_nrn,
        "StaticPulse", {}, {"g": coincidence_disparity_exc_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # Inhibitory Coincidence -> Disparity (plane of constant horizontal cyclopean position)
    conn_init = pygenn.genn_model.init_connectivity(
        coincidence_disparity_inh_syn_init,
        {"cam_height": cam_height, "cam_width": cam_width, "receptive_distance": receptive_distance})
    coincidence_pos_disparity_inh_syn = model.add_synapse_population(
        "coincidence_pos_disparity_inh", conn_matrix, 0, coincidence_pos_nrn, disparity_nrn,
        "StaticPulse", {}, {"g": coincidence_disparity_inh_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)
    coincidence_neg_disparity_inh_syn = model.add_synapse_population(
        "coincidence_neg_disparity_inh", conn_matrix, 0, coincidence_neg_nrn, disparity_nrn,
        "StaticPulse", {}, {"g": coincidence_disparity_inh_W}, {}, {},
        "DeltaCurr", {}, {}, conn_init)

    # # Inhibitory Recurrent Disparity -> Disparity
    # conn_init = pygenn.genn_model.init_connectivity(
    #     disparity_disparity_inh_syn_init,
    #     {"cam_height": cam_height, "cam_width": cam_width})
    # disparity_disparity_inh_syn = model.add_synapse_population(
    #     "disparity_disparity_inh", conn_matrix, 0, disparity_nrn, disparity_nrn,
    #     "StaticPulse", {}, {"g": disparity_disparity_inh_W}, {}, {},
    #     "DeltaCurr", {}, {}, conn_init)



    # Build and load GeNN model
    model.build()
    model.load()






    # Set retinae inputs
    def set_input(pos_L, neg_L, pos_R, neg_R):
        retina_pos_L_nrn.vars["input"].view.reshape(cam_height, cam_width)[:] = pos_L
        retina_pos_L_nrn.push_var_to_device("input")
        retina_neg_L_nrn.vars["input"].view.reshape(cam_height, cam_width)[:] = neg_L
        retina_neg_L_nrn.push_var_to_device("input")
        retina_pos_R_nrn.vars["input"].view.reshape(cam_height, cam_width)[:] = pos_R
        retina_pos_R_nrn.push_var_to_device("input")
        retina_neg_R_nrn.vars["input"].view.reshape(cam_height, cam_width)[:] = neg_R
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
        i = coordinates["y"] * cam_width**2 + coordinates["x_L"] * cam_width + coordinates["x_R"]
        return i

    # Neuron index to coordinates
    def index_to_coordinates(i):
        coordinates = {}
        coordinates["y"] = (i // cam_width) // cam_width
        coordinates["x_L"] = (i // cam_width) % cam_width
        coordinates["x_R"] = i % cam_width
        return coordinates





    output = np.empty((cam_height, cam_width * 2, 3), dtype="float32")
    output_L = output[:cam_height, :cam_width]
    output_R = output[:cam_height, cam_width:]



    # Simulate model
    for pos_L, neg_L, pos_R, neg_R in f:

        # Step time
        set_input(pos_L, neg_L, pos_R, neg_R)
        model.step_time()

        # Get spikes
        coincidence_pos_nrn.pull_current_spikes_from_device()
        coincidence_pos_spikes = coincidence_pos_nrn.current_spikes
        coincidence_neg_nrn.pull_current_spikes_from_device()
        coincidence_neg_spikes = coincidence_neg_nrn.current_spikes
        disparity_nrn.pull_current_spikes_from_device()
        disparity_spikes = disparity_nrn.current_spikes

        # Filter spikes (handles broadly tuned cyclopean distance)
        pos_spikes = np.intersect1d(coincidence_pos_spikes, disparity_spikes)
        neg_spikes = np.intersect1d(coincidence_neg_spikes, disparity_spikes)
        #pos_spikes = disparity_spikes
        #neg_spikes = disparity_spikes






        output_L.fill(0)
        output_R.fill(0)

        for spike_i in pos_spikes:
            coordinates = index_to_coordinates(spike_i)
            y = coordinates["y"]
            x_L = coordinates["x_L"]
            x_R = coordinates["x_R"]
            #assert output_L[y, x_L, 1] == 0
            #assert output_R[y, x_R, 1] == 0
            output_L[y, x_L, 1] = (x_R - x_L) / cam_width
            output_R[y, x_R, 1] = (x_R - x_L) / cam_width

        for spike_i in neg_spikes:
            coordinates = index_to_coordinates(spike_i)
            y = coordinates["y"]
            x_L = coordinates["x_L"]
            x_R = coordinates["x_R"]
            #assert output_L[y, x_L, 2] == 0
            #assert output_R[y, x_R, 2] == 0
            output_L[y, x_L, 2] = (x_R - x_L) / cam_width
            output_R[y, x_R, 2] = (x_R - x_L) / cam_width

        cv2.imshow("output", output)
        k = cv2.waitKey(0)
        if k == ord("q"):
            cv2.destroyWindow("output")
            exit(0)









    ############################ TESTING ############################

    # spikes = []
    # for i in range(2):

        # if i == 0:

            # # Retinae
            # test_pos_L = np.zeros((cam_height, cam_width))
            # test_pos_L[10, 4] = 1.0
            # retina_pos_L_nrn.vars["input"].view.reshape(cam_height, cam_width)[:] = test_pos_L
            # test_pos_R = np.zeros((cam_height, cam_width))
            # #test_pos_R[10, 4] = 1.0
            # #test_pos_R[11, 4] = 1.0
            # test_pos_R[10, 5] = 1.0
            # retina_pos_R_nrn.vars["input"].view.reshape(cam_height, cam_width)[:] = test_pos_R


            # # Coincidence
            # test_pos = np.zeros((cam_height, cam_width, cam_width))

            # # Cyclopean distance window (centred on [10, 46, 56])
            # test_pos[10, 40, 50] = 5.0 # side A
            # test_pos[10, 52, 62] = 5.0 # side B
            # #test_pos[10, 53, 62] = 5.0 # side B (just outside)

            # # Disparity window (centred on [10, 46, 56])
            # test_pos[10, 52, 50] = 5.0 # side A
            # #test_pos[10, 53, 49] = 5.0 # side A (just outside)

            # coincidence_pos_nrn.vars["V"].view.reshape(cam_height, cam_width, cam_width)[:] = test_pos


            # DONE: RETINAE -> COINCIDENCE SYNAPSES WORKING
            # DONE: COINCIDENCE -> DISPARITY EXC AND INH SYNAPSES WORK


        # else:

            # # Retinae
            # retina_pos_L_nrn.vars["input"].view.reshape(cam_height, cam_width)[:] = 0.0
            # retina_pos_R_nrn.vars["input"].view.reshape(cam_height, cam_width)[:] = 0.0

            # # Coincidence detectors
            # coincidence_pos_nrn.vars["V"].view.reshape(cam_height, cam_width, cam_width)[:] = 0.0

        # # Retinae
        # retina_pos_L_nrn.push_var_to_device("input")
        # retina_pos_R_nrn.push_var_to_device("input")

        # # Coincidence detectors
        # coincidence_pos_nrn.push_var_to_device("V")


        # model.step_time()


        # # Coincidence detectors
        # coincidence_pos_nrn.pull_current_spikes_from_device()
        # spikes.append(coincidence_pos_nrn.current_spikes)

        # # Disparity detectors
        # disparity_nrn.pull_current_spikes_from_device()
        # spikes.append(disparity_nrn.current_spikes)


    # #print(spikes)
    # for t, s in enumerate(spikes):
    #     print("t:", t)
    #     for i in s:
    #         print("i:", i, "  coordinates:", index_to_coordinates(i))

    ############################






if __name__  == "__main__":
    main()
