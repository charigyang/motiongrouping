import os
import shutil
import numpy as np
import cv2

base_dir = '/path/to/FBMS_clean/'
save_dir = '/path/to/FBMS_val/'
os.makedirs(save_dir, exist_ok=True)
im_save_dir = '/path/to/FBMS_val/JPEGImages'
os.makedirs(im_save_dir, exist_ok=True)
anno_save_dir = '/path/to/FBMS_val/Annotations'
os.makedirs(anno_save_dir, exist_ok=True)

flow_dirs = ['/path/to/FBMS_clean/Flows_gap-1/',
            '/path/to/FBMS_clean/Flows_gap-2/',
            '/path/to/FBMS_clean/Flows_gap1/',
            '/path/to/FBMS_clean/Flows_gap2/']

flow_save_dirs = ['/path/to/FBMS_val/Flows_gap-1/',
                '/path/to/FBMS_val/Flows_gap-2/',
                '/path/to/FBMS_val/Flows_gap1/',
                '/path/to/FBMS_val/Flows_gap2/']

for f in flow_save_dirs:
    os.makedirs(f, exist_ok=True)

test_vids = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06', 
            'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04', 
            'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9', 
            'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']

all_vids = test_vids

for vid in all_vids:
    vid_dir = os.path.join(base_dir, "JPEGImages", vid)
    images = sorted(os.listdir(vid_dir))
    images = [x for x in images if x.endswith(".jpg")]

    annotation_dir = os.path.join(base_dir, "Annotations", vid)
    annotations = sorted(os.listdir(annotation_dir))
    annotations = [x for x in annotations if x.endswith(".png")]
    ims_with_annos = [x.replace(".png", ".jpg") for x in annotations]
    flos = [x.replace(".png", ".flo") for x in annotations]

    save1 = os.path.join(im_save_dir, vid)
    save2 = os.path.join(anno_save_dir, vid)
    os.makedirs(save1, exist_ok=True)
    os.makedirs(save2, exist_ok=True)
    for i in annotations:
        shutil.copy2(os.path.join(annotation_dir, i), save2)
    for i in ims_with_annos:
        shutil.copy2(os.path.join(vid_dir, i), save1)
    for flo in flos:
        for f in range(4):
            os.makedirs(os.path.join(flow_save_dirs[f], vid), exist_ok=True)
            if os.path.isfile(os.path.join(flow_dirs[f], vid, flo)):
                shutil.copy2(os.path.join(flow_dirs[f], vid, flo), os.path.join(flow_save_dirs[f], vid, flo))
            else:
                print("does not exist:", os.path.join(flow_dirs[f], vid, flo))
