import glob
import cv2
import os
import pandas as pd
import numpy as np
import ast
import shutil

"""
This code cleans MoCA
- crops out black space
- resize all to 720p
- crops out all logos

and saves the annotations as .png files (same as DAVIS format)
"""

base_dir = "/scratch/shared/beegfs/charig/MoCA"

category_dir = os.path.join(base_dir, "JPEGImages_cropped")
annotation_dir = os.path.join(base_dir, "Annotations_cropped")

save_dir = "/scratch/shared/beegfs/charig/MoCA_filtered"
os.makedirs(save_dir, exist_ok=True)
cat_save_dir = os.path.join(save_dir, "JPEGImages")
os.makedirs(cat_save_dir, exist_ok=True)
anno_save_dir = os.path.join(save_dir, "Annotations")
os.makedirs(anno_save_dir, exist_ok=True)
categories = sorted(os.listdir(category_dir))

filtered_cats = ['arabian_horn_viper', 'arctic_fox_1', 'arctic_wolf_1', 'black_cat_1', 'crab', 'crab_1', 
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

for cat in filtered_cats:
	print(cat)
	images = sorted(os.listdir(os.path.join(category_dir, cat)))
	annos = sorted(os.listdir(os.path.join(annotation_dir, cat)))
	save1 = os.path.join(cat_save_dir, cat)
	save2 = os.path.join(anno_save_dir, cat)
	os.makedirs(save1, exist_ok=True)
	os.makedirs(save2, exist_ok=True)
	j = 0
	for i in range(min(len(images),300)):
		assert images[i][:-4] == annos[i][:-4]
		if i % 3 == 0:
			im = os.path.join(category_dir, cat, images[i])
			anno = os.path.join(annotation_dir, cat, annos[i])
			save_name1 = "{}.jpg".format(str(j).zfill(5))
			save_name2 = "{}.png".format(str(j).zfill(5))
			shutil.copy2(im, os.path.join(save1, save_name1))
			shutil.copy2(anno, os.path.join(save2, save_name2))
			j+=1
