import os
import shutil
import numpy as np
import cv2

base_dir = '/path/to/FBMS/'
save_dir = '/path/to/FBMS_clean/'
os.makedirs(save_dir, exist_ok=True)
im_save_dir = '/path/to/FBMS_clean/JPEGImages'
os.makedirs(im_save_dir, exist_ok=True)
anno_save_dir = '/path/to/FBMS_clean/Annotations'
os.makedirs(anno_save_dir, exist_ok=True)


train_dir = sorted(os.listdir(base_dir + "Trainingset"))
test_dir = sorted(os.listdir(base_dir + "Testset"))
train_vids = ['bear01', 'bear02', 'cars2', 'cars3', 'cars6', 'cars7', 'cars8', 'cars9', 'cats02', 
            'cats04', 'cats05', 'cats07', 'ducks01', 'horses01', 'horses03', 'horses06', 'lion02', 
            'marple1', 'marple10', 'marple11', 'marple13', 'marple3', 'marple5', 'marple8', 
            'meerkats01', 'people04', 'people05', 'rabbits01', 'rabbits05']
test_vids = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06', 
            'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04', 
            'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9', 
            'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']
all_vids = train_vids + test_vids

def find_gt(directory):
    all_files = os.listdir(directory)
    # Check in which kind of folder you are
    type_weird=False
    for file in all_files:
        if file.endswith('ppm'):
            type_weird=True
            break
    if not type_weird:
        all_files = [file for file in all_files if file.endswith('pgm')]
        # Sort them
        try:
            all_files = sorted(all_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
            numbers = [int(file.split('.')[0].split('_')[-1]) for file in all_files]
        except:
            all_files = sorted(all_files, key=lambda x: int(re.search(r'\d+', x).group()))
            numbers = [int(re.search(r'\d+', file).group()) for file in all_files]
        return all_files, numbers, type_weird
    # Solve weird type
    all_files = [file for file in all_files if file.endswith('ppm') and not 'PROB' in file]
    all_files = sorted(all_files, key=lambda x: int(x.split('_')[1]))
    numbers = [int(file.split('_')[1]) for file in all_files]
    return all_files, numbers, type_weird

for vid in all_vids:
    if vid in train_vids:
        vid_dir = os.path.join(base_dir, "Trainingset", vid)
    elif vid in test_vids:
        vid_dir = os.path.join(base_dir, "Testset", vid)
    images = sorted(os.listdir(vid_dir))
    images = [x for x in images if x.endswith(".jpg")]

    save1 = os.path.join(im_save_dir, vid)
    os.makedirs(save1, exist_ok=True)
    for i, im in enumerate(images):
    	img = os.path.join(vid_dir, im)
    	save_name = str(i+1).zfill(5) + ".jpg"
    	shutil.copy2(img, os.path.join(save1, save_name))

    anno_dir = os.path.join(vid_dir, "GroundTruth")
    annotation_fnames, n_with_gt, type_weird = find_gt(anno_dir)

    save2 = os.path.join(anno_save_dir, vid)
    os.makedirs(save2, exist_ok=True)
    goal_annotation_fnames = [os.path.join(save2, str(n).zfill(5)) + '.png' for n in n_with_gt]
    annotation_fnames = [os.path.join(anno_dir, f) for f in annotation_fnames]
    for i in range(len(goal_annotation_fnames)):
        mask = cv2.imread(annotation_fnames[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask / 255.0
        if type_weird:
            mask[mask>0.99] = 0.0
        if 'marple7' == vid:
            mask = mask>0.05
        elif 'marple2' == vid:
            mask = mask>0.4
        else:
            mask = mask>0.1
        mask = np.asarray(mask*255, dtype=np.uint8)
        cv2.imwrite(goal_annotation_fnames[i], mask)