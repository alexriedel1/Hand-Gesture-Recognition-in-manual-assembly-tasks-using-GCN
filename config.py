class CFG:
    debug = False
    batch_size = 16
    sequence_length = 32
    num_classes = 6
    num_feats = 3
    lr = 1e-4
    min_lr = 1e-5
    epochs = 500
    print_freq = 100
    resume = False

    model_type = "AAGCN" # AAGCN, LSTM, STCONV

    add_feats = False
    add_phi = False

    add_joints1 = True     #Abl
    add_joints2 = True     #Abl    
    add_joints_mode = "ori"
    sam = True             #Abl
    only_dist = False       #Abl
    loss_fn = "Focal"       #Abl

    no_release = False      #Abl

    experiment_name = f"2{model_type}_{loss_fn}_seqlen{sequence_length}_{'release_' if not no_release else 'no_release_'}{'SAM_' if sam else ''}{'joints1_' if add_joints1 else ''}{'joints2_' if add_joints2 else ''}{'dist' if only_dist else ''}"

    if no_release:
         num_classes = 5

    plot_weights = True
    
    if add_feats:
        num_feats = 6
    
    stconv_spatial_channels = 16
    stconv_out_channels = 64

    lstm_num_layers = 2
    lstm_hidden_layers = 120
    
    classes = ["Grasp",   "Move",    "Negative",    "Position",    "Reach",   "Release"]

    if no_release:
        classes = ["Grasp",   "Move",    "Negative",    "Position",    "Reach"]

    
                     