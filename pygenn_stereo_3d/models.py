from pygenn import genn_model
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE


#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------

spike_input = genn_model.create_custom_neuron_class(
    'spike_input',

    var_name_types=[('input', 'scalar', VarAccess_READ_ONLY_DUPLICATE)],

    sim_code='''
    const bool spike = $(input) != 0.0;
    ''',

    threshold_condition_code='''
    spike
    ''',

    is_auto_refractory_required=False,
)

LIF = genn_model.create_custom_neuron_class(
    "LIF",

    param_names=[
        "V_thres",
        "tau",
    ],

    var_name_types=[
        ("V", "scalar"),
    ],

    sim_code="""
    if ($(V) < 0.0) $(V) = 0.0;
    $(V) += ($(Isyn) - ($(V) / $(tau))) * DT;
    """,

    threshold_condition_code="""
    $(V) >= $(V_thres)
    """,

    reset_code="""
    $(V) = 0.0;
    """,

    is_auto_refractory_required=False,
)


#----------------------------------------------------------------------------
# Synapse models
#----------------------------------------------------------------------------

retina_L_coincidence_syn_init = genn_model.create_custom_sparse_connect_init_snippet_class(
    "retina_L_coincidence_syn",

    param_names=[
        "cam_height",
        "cam_width",
    ],

    row_build_state_vars=[
        ("pre_y", "int", "(int) $(id_pre) / (int) $(cam_width)"),
        ("pre_x_L", "int", "(int) $(id_pre) % (int) $(cam_width)"),
    ],

    row_build_code="""
    int id_post_y = $(pre_y) * (int) $(cam_width) * (int) $(cam_width);
    int id_post_x_L = $(pre_x_L) * (int) $(cam_width);
    for (int id_post_x_R = 0; id_post_x_R < (int) $(cam_width); id_post_x_R++) {
        int id_post = id_post_y + id_post_x_L + id_post_x_R;
        $(addSynapse, id_post);
    }
    $(endRow);
    """,

    calc_max_row_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: int(pars[1]))(),
)

retina_R_coincidence_syn_init = genn_model.create_custom_sparse_connect_init_snippet_class(
    "retina_R_coincidence_syn",

    param_names=[
        "cam_height",
        "cam_width",
    ],

    row_build_state_vars=[
        ("pre_y", "int", "(int) $(id_pre) / (int) $(cam_width)"),
        ("pre_x_R", "int", "(int) $(id_pre) % (int) $(cam_width)"),
    ],

    row_build_code="""
    int id_post_y = $(pre_y) * (int) $(cam_width) * (int) $(cam_width);
    int id_post_x_R = $(pre_x_R);
    for (int post_x_L = 0; post_x_L < (int) $(cam_width); post_x_L++) {
        int id_post_x_L = post_x_L * (int) $(cam_width);
        int id_post = id_post_y + id_post_x_L + id_post_x_R;
        $(addSynapse, id_post);
    }
    $(endRow);
    """,

    calc_max_row_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: int(pars[1]))(),
)

#  C+  =   {  c in C  |  (|x_c - x_d| <= U)  and  (|y_c - y_d| <= U)  and  (d_c == d_d)  }
coincidence_disparity_exc_syn_init = genn_model.create_custom_sparse_connect_init_snippet_class(
    "coincidence_disparity_exc_syn",

    param_names=[
        "cam_height",
        "cam_width",
        "receptive_distance",
    ],

    row_build_state_vars=[
        ("pre_y", "int", "((int) $(id_pre) / (int) $(cam_width)) / (int) $(cam_width)"),
        ("pre_x_L", "int", "((int) $(id_pre) / (int) $(cam_width)) % (int) $(cam_width)"),
        ("pre_x_R", "int", "(int) $(id_pre) % (int) $(cam_width)"),
    ],

    row_build_code="""
    for (int y_offset = -$(receptive_distance); y_offset <= $(receptive_distance); y_offset++) {
        int post_y = $(pre_y) + y_offset;
        if ((post_y >= 0) && (post_y < $(cam_height))) {
            int id_post_y = post_y * (int) $(cam_width) * (int) $(cam_width);
            for (int x_offset = -$(receptive_distance); x_offset <= $(receptive_distance); x_offset++) {
                int post_x_L = $(pre_x_L) + x_offset;
                int post_x_R = $(pre_x_R) + x_offset;
                if ((post_x_L >= 0) && (post_x_L < $(cam_width)) && (post_x_R >= 0) && (post_x_R < $(cam_width))) {
                    int id_post_x_L = post_x_L * (int) $(cam_width);
                    int id_post_x_R = post_x_R;
                    int id_post = id_post_y + id_post_x_L + id_post_x_R;
                    $(addSynapse, id_post);
                }
            }
        }
    }
    $(endRow);
    """,

    calc_max_row_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: ((int(pars[2]) * 2) + 1) ** 2)(),
)

#  C-  =   {  c in C  |  (x_c == x_d)  and  (|y_c - y_d| <= U)  and  (|d_c - d_d| <= U)  }
coincidence_disparity_inh_syn_init = genn_model.create_custom_sparse_connect_init_snippet_class(
    "coincidence_disparity_inh_syn",

    param_names=[
        "cam_height",
        "cam_width",
        "receptive_distance",
    ],

    row_build_state_vars=[
        ("pre_y", "int", "((int) $(id_pre) / (int) $(cam_width)) / (int) $(cam_width)"),
        ("pre_x_L", "int", "((int) $(id_pre) / (int) $(cam_width)) % (int) $(cam_width)"),
        ("pre_x_R", "int", "(int) $(id_pre) % (int) $(cam_width)"),
    ],

    row_build_code="""
    for (int y_offset = -$(receptive_distance); y_offset <= $(receptive_distance); y_offset++) {
        int post_y = $(pre_y) + y_offset;
        if ((post_y >= 0) && (post_y < $(cam_height))) {
            int id_post_y = post_y * (int) $(cam_width) * (int) $(cam_width);
            for (int x_offset = -$(receptive_distance); x_offset <= $(receptive_distance); x_offset++) {
                int post_x_L = $(pre_x_L) - x_offset;
                int post_x_R = $(pre_x_R) + x_offset;
                if ((post_x_L >= 0) && (post_x_L < $(cam_width)) && (post_x_R >= 0) && (post_x_R < $(cam_width))) {
                    int id_post_x_L = post_x_L * (int) $(cam_width);
                    int id_post_x_R = post_x_R;
                    int id_post = id_post_y + id_post_x_L + id_post_x_R;
                    $(addSynapse, id_post);
                }
            }
        }
    }
    $(endRow);
    """,

    calc_max_row_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: ((int(pars[2]) * 2) + 1) ** 2)(),
)

#  D-  =   {  d in D  |  (x_d - d_d == 2 * x_L)  or  (x_d + d_d == 2 * x_R)  }
disparity_disparity_inh_syn_init = genn_model.create_custom_sparse_connect_init_snippet_class(
    "disparity_disparity_inh_syn",

    param_names=[
        "cam_height",
        "cam_width",
    ],

    row_build_state_vars=[
        ("pre_y", "int", "((int) $(id_pre) / (int) $(cam_width)) / (int) $(cam_width)"),
        ("pre_x_L", "int", "((int) $(id_pre) / (int) $(cam_width)) % (int) $(cam_width)"),
        ("pre_x_R", "int", "(int) $(id_pre) % (int) $(cam_width)"),
    ],

    row_build_code="""
    int pre_x = pre_x_R + pre_x_L;
    int pre_d = pre_x_R - pre_x_L;
    int post_y = $(pre_y);
    int id_post_y = post_y * (int) $(cam_width) * (int) $(cam_width);
    for (int post_x_L = 0; post_x_L < (int) $(cam_width); post_x_L++) {
        for (int post_x_R = 0; post_x_R < (int) $(cam_width); post_x_R++) {
            if ((pre_x - pre_d == 2 * post_x_L) || (pre_x + pre_d == 2 * post_x_R)) {
                int id_post_x_L = post_x_L * (int) $(cam_width);
                int id_post_x_R = post_x_R;
                int id_post = id_post_y + id_post_x_L + id_post_x_R;
                $(addSynapse, id_post);
            }
        }
    }
    $(endRow);
    """,

    # calc_max_row_len_func=genn_model.create_cmlf_class(
    #     lambda num_pre, num_post, pars: int(pars[1]))(),
)


# TODO: TEST RECURRENT INH SYNAPSES

# THINK: WHAT IS MAX ROW LENGTH IN LINE OF SIGHT?
