import os
import torch
import itertools
import glob as gb
import numpy as np
import json
from datetime import datetime
from data import FlowPair, FlowEval


def setup_path(args):
    dataset = args.dataset
    num_slots = args.num_slots
    iters = args.num_iterations
    batch_size = args.batch_size
    resolution = args.resolution
    flow_to_rgb = args.flow_to_rgb
    verbose = args.verbose if args.verbose else 'none'
    flow_to_rgb_text = 'rgb' if flow_to_rgb else 'uv'
    inference = args.inference

    # make all the essential folders, e.g. models, logs, results, etc.
    global dt_string, logPath, modelPath, resultsPath
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H")

    os.makedirs('../logs/', exist_ok=True)
    os.makedirs('../models/', exist_ok=True)
    os.makedirs('../results/', exist_ok=True)

    logPath = os.path.join('../logs/', f'{dt_string}-dataset_{dataset}-{flow_to_rgb_text}-'
                                           f'slots_{num_slots}-VGGNet_D256-'
                                           f'iter_{iters}-bs_{batch_size}-res_{resolution[0]}x{resolution[1]}-{verbose}')

    modelPath = os.path.join('../models/', f'{dt_string}-dataset_{dataset}-{flow_to_rgb_text}-'
                                               f'slots_{num_slots}-VGGNet_D256-'
                                               f'iter_{iters}-bs_{batch_size}-res_{resolution[0]}x{resolution[1]}-{verbose}')

    if inference:
        resultsPath = os.path.join('../results/', args.resume_path.split('/')[-1])
        os.makedirs(resultsPath, exist_ok=True)
    else:
        os.makedirs(logPath, exist_ok=True)
        os.makedirs(modelPath, exist_ok=True)
        resultsPath = None

        # save all the experiment settings.
        with open('{}/running_command.txt'.format(modelPath), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return [logPath, modelPath, resultsPath]


def setup_dataset(args):
    resolution = args.resolution  # h,w
    res = ""
    with_gt = True
    pairs = [1, 2, -1, -2]
    if args.dataset == 'DAVIS':
        
        basepath = '/path/to/SlotAttention/data/DAVIS2016'
        img_dir = '/path/to/SlotAttention/data/DAVIS2016/JPEGImages/480p'
        gt_dir ='/path/to/SlotAttention/data/DAVIS2016/Annotations/480p'

        val_flow_dir = '/path/to/SlotAttention/data/DAVIS2016/Flows_gap1/1080p'
        val_seq = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees', 
                    'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane', 
                    'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch']            
        val_data_dir = [val_flow_dir, img_dir, gt_dir]
        res = "1080p"

    elif args.dataset == 'FBMS':
        basepath = '/path/to/FBMS_clean'
        img_dir = '/path/to/FBMS_clean/JPEGImages/'
        gt_dir ='/path/to/FBMS_clean/Annotations/'    

        val_flow_dir = '/path/to/FBMS_val/Flows_gap1/'
        val_seq = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06', 
                    'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04', 
                    'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9', 
                    'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']
        val_img_dir = '/path/to/FBMS_val/JPEGImages/'
        val_gt_dir ='/path/to/FBMS_val/Annotations/'
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]
        with_gt = False
        pairs = [3, 6, -3, -6]

    elif args.dataset == 'STv2':
        basepath = '/path/to/SegTrackv2'
        img_dir = '/path/to/SegTrackv2/JPEGImages'
        gt_dir ='/path/to/SegTrackv2/Annotations'

        val_flow_dir = '/path/to/SegTrackv2/Flows_gap1/'
        val_seq = ['drift', 'birdfall', 'girl', 'cheetah', 'worm', 'parachute', 'monkeydog',
                    'hummingbird', 'soldier', 'bmx', 'frog', 'penguin', 'monkey', 'bird_of_paradise']
        val_data_dir = [val_flow_dir, img_dir, gt_dir]

    elif args.dataset == 'MoCA':
        basepath = '/path/to/MoCA_filtered'
        img_dir = '/path/to/MoCA_filtered/JPEGImages'
        gt_dir ='/path/to/MoCA_filtered/Annotations'

        val_flow_dir = '/path/to/MoCA_filtered/Flows_gap1/'
        val_seq = ['arabian_horn_viper', 'arctic_fox_1', 'arctic_wolf_1', 'black_cat_1', 'crab', 'crab_1', 
                    'cuttlefish_0', 'cuttlefish_1', 'cuttlefish_4', 'cuttlefish_5', 
                    'devil_scorpionfish', 'devil_scorpionfish_1', 'flatfish_2', 'flatfish_4', 'flounder', 
                    'flounder_3', 'flounder_4', 'flounder_5', 'flounder_6', 'flounder_7', 
                    'flounder_8', 'flounder_9', 'goat_1', 'hedgehog_1', 'hedgehog_2', 'hedgehog_3', 
                    'hermit_crab', 'jerboa', 'jerboa_1', 'lion_cub_0', 'lioness', 'marine_iguana', 
                    'markhor', 'meerkat', 'mountain_goat', 'peacock_flounder_0', 
                    'peacock_flounder_1', 'peacock_flounder_2', 'polar_bear_0', 'polar_bear_2', 
                    'scorpionfish_4', 'scorpionfish_5', 'seal_1', 'shrimp', 
                    'snow_leopard_0', 'snow_leopard_1', 'snow_leopard_2', 'snow_leopard_3', 'snow_leopard_6', 
                    'snow_leopard_7', 'snow_leopard_8', 'spider_tailed_horned_viper_0', 
                    'spider_tailed_horned_viper_2', 'spider_tailed_horned_viper_3',
                    'arctic_fox', 'arctic_wolf_0', 'devil_scorpionfish_2', 'elephant', 
                    'goat_0', 'hedgehog_0', 
                    'lichen_katydid', 'lion_cub_3', 'octopus', 'octopus_1', 
                    'pygmy_seahorse_2', 'rodent_x', 'scorpionfish_0', 'scorpionfish_1', 
                    'scorpionfish_2', 'scorpionfish_3', 'seal_2',
                    'bear', 'black_cat_0', 'dead_leaf_butterfly_1', 'desert_fox', 'egyptian_nightjar', 
                    'pygmy_seahorse_4', 'seal_3', 'snowy_owl_0',
                    'flatfish_0', 'flatfish_1', 'fossa', 'groundhog', 'ibex', 'lion_cub_1', 'nile_monitor_1',
                    'polar_bear_1', 'spider_tailed_horned_viper_1']
        val_data_dir = [val_flow_dir, img_dir, gt_dir]
    else:
        raise ValueError('Unknown Setting.')

    pair_list = [p for p in itertools.combinations(pairs , 2)]
    folders = [os.path.basename(x) for x in gb.glob(os.path.join(basepath, 'Flows_gap1/{}/*'.format(res)))]
    flow_dir = {}
    for pair in pair_list:
        p1, p2 = pair
        flowpairs = []
        for f in folders:
            path1 = os.path.join(basepath, 'Flows_gap{}/{}/{}'.format(p1, res, f))
            path2 = os.path.join(basepath, 'Flows_gap{}/{}/{}'.format(p2, res, f))

            flows1 = [os.path.basename(x) for x in gb.glob(os.path.join(path1, '*'))]
            flows2 = [os.path.basename(x) for x in gb.glob(os.path.join(path2, '*'))]

            intersect = list(set(flows1).intersection(flows2))
            intersect.sort()

            flowpair = np.array([[os.path.join(path1, i), os.path.join(path2, i)] for i in intersect])
            flowpairs += [flowpair]
        flow_dir['gap_{}_{}'.format(p1, p2)] = flowpairs

    # flow_dir is a dictionary, with keys indicating the flow gap, and each value is a list of sequence names,
    # each item then is an array with Nx2, N indicates the number of available pairs.
    data_dir = [flow_dir, img_dir, gt_dir]
    trn_dataset = FlowPair(data_dir=data_dir, resolution=resolution, to_rgb=args.flow_to_rgb,
                            with_rgb=False, with_gt=with_gt)
    
    val_dataset = FlowEval(data_dir=val_data_dir, resolution=resolution, pair_list=pairs, 
                            val_seq=val_seq, to_rgb=args.flow_to_rgb, with_rgb=False)

    in_out_channels = 3 if args.flow_to_rgb else 2
    use_flow = False if args.flow_to_rgb else True
    loss_scale = 1e2
    ent_scale = 1e-2
    cons_scale = 1e-2

    return [trn_dataset, val_dataset, resolution, in_out_channels, use_flow, loss_scale, ent_scale, cons_scale]
