import glob
import cv2
import os
import pandas as pd
import numpy as np
import ast

"""
This code cleans MoCA
- crops out black space
- resize all to 720p
- crops out all logos

and saves the annotations as .png files (same as DAVIS format)
"""

base_dir = "/path/to/MoCA"

category_dir = os.path.join(base_dir, "JPEGImages")
categories = sorted(os.listdir(category_dir))

#load annotations
anno = pd.read_csv(os.path.join(base_dir, "Annotations", "annotations.csv"), header=9)
file = anno['file_list'].tolist()
anno_cats = [f.split('/')[1] for f in file]
anno_nums = [int(f.split('/')[2].strip('.jpg')) for f in file]
anno_bbox = anno['spatial_coordinates'].tolist()
anno_x = [float(ast.literal_eval(b)[1]) for b in anno_bbox]
anno_y = [float(ast.literal_eval(b)[2]) for b in anno_bbox]
anno_w = [float(ast.literal_eval(b)[3]) for b in anno_bbox]
anno_h = [float(ast.literal_eval(b)[4]) for b in anno_bbox]

logo_crops = {'arctic_fox': [89, 37], #bbc
'arctic_fox_1': [89, 37], 
'arctic_fox_2': [89, 37], 
'black_cat_0': [107, 106], #nature
'black_cat_1': [107, 106], 
'chameleon': [1055, 139], #natgeo wild top right
'copperhead_snake': [828, 77], #kinemaster
'crab': [1011, 601], #kpbs hd
'crab_1': [1205, 90], #wulfi
'crab_2': [1205, 90], 
'cuttlefish_0': [1011, 601], 
'cuttlefish_1': [1011, 601], 
'cuttlefish_3': [1011, 601], 
'cuttlefish_4': [1011, 601], 
'desert_fox': [89, 37], 
'egyptian_nightjar': [318, 650], #lironziv.com
'elephant': [1189, 109], #kamikatze 
'flounder': [1149, 81], #blue world 
'flounder_3': [1029, 589], #natgeo wild bottom right 
'flounder_4': [1029, 589], #natgeo wild bottom right 
'fossa': [248, 103], #nat geo top left
'hyena': [0, 120], #nat geo abu dabhi, but needs to not crop animal 
'jerboa': [89, 37], 
'jerboa_1': [89, 37], 
'lichen_katydid': [962, 672], 
'lioness': [260, 120], #nat geo abu dabhi
'marine_iguana': [89, 37], 
'markhor': [89, 37], 
'meerkat': [89, 37], 
'mangoose': [200, 0], #nat geo abu dabhi, but needs to not crop animal 
'octopus': [1149, 81], #blue world 
'octopus_1': [1149, 81], #blue world 
'flounder': [1149, 81], #blue world 
'flounder': [1149, 81], #blue world 
'flounder': [1149, 81], #blue world 
'polar_bear_0': [1202, 102], #natgeo logo only
'polar_bear_1': [1202, 102], 
'polar_bear_cub': [1202, 102], 
'pygmy_seahorse_0': [1030, 615], #jean michel 
'pygmy_seahorse_1': [1030, 615],
'pygmy_seahorse_2': [1030, 615],
'pygmy_seahorse_4': [1149, 81],
'rodent_x': [260, 0], #nat geo abu dabhi, but cropping
'seal_1': [1027, 0], #natgeo crop 
'seal_2': [0, 95],
'seal_3': [1027, 95],
'shrimp': [1027, 0], #natgeo crop 
'smallfish': [1149, 81], #blue world 
'snow_leopard_0': [89, 37], 
'snow_leopard_1': [89, 37], 
'snow_leopard_10': [1026, 658],
'snow_leopard_2': [89, 37], 
'snow_leopard_3': [89, 37], 
'snow_leopard_6': [1202, 102], 
'snow_leopard_7': [1202, 102], 
'snow_leopard_8': [1202, 102], 
'snowy_owl_0': [1029, 629],
'snowy_owl_2': [346, 585],
'spider_tailed_horned_viper_0': [1152, 97], 
'spider_tailed_horned_viper_2': [120, 57], #bbc larger
'spider_tailed_horned_viper_3': [120, 57], 
}

if '.DS_Store' in categories:
	categories.remove('.DS_Store')
categories = ['arabian_horn_viper', 'arctic_fox_1', 'arctic_wolf_1', 'black_cat_1', 'crab', 'crab_1', 
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
				'bear', 'black_cat_0', 'dead_leaf_butterfly', 'desert_fox', 'egyptian_nightjar', 
				'pygmy_seahorse_4', 'seal_3', 'snowy_owl_0',
				'flatfish_0', 'flatfish_1', 'fossa', 'groundhog', 'ibex', 'lion_cub_1', 'nile_monitor_1',
				'polar_bear_1', 'spider_tailed_horned_viper_1']

for category in categories:

	#get the indices for the annotations to choose the right category

	anno_indices = [n for n,x in enumerate(anno_cats) if x==category]
	image_ids = [anno_nums[i] for i in anno_indices]

	#get category bbox (every 5 frames)
	x_s = [anno_x[i] for i in anno_indices]
	y_s = [anno_y[i] for i in anno_indices]
	w_s = [anno_w[i] for i in anno_indices]
	h_s = [anno_h[i] for i in anno_indices]

	#interpolate the bboxes
	num_images = image_ids[-1]+1
	X_s = np.interp(range(num_images), image_ids, x_s)
	Y_s = np.interp(range(num_images), image_ids, y_s)
	W_s = np.interp(range(num_images), image_ids, w_s)
	H_s = np.interp(range(num_images), image_ids, h_s)

	#load first image
	images = sorted(os.listdir(os.path.join(category_dir, category)))
	first_img_dir = os.path.join(category_dir, category, images[0])
	first_img = cv2.imread(first_img_dir)
	H, W, _ = np.shape(first_img)
	H_ = 720
	W_ = 1280

	y_low = 0
	y_high = H
	x_low = 0
	x_high = W
	
	if category in logo_crops:
		x, y = logo_crops[category]
		if x > W/2: #logo on right side, crop away right side
			x_high = x
		else:
			x_low = x
		if y > H/2: #logo on top, crop away top
			y_high = y
		else:
			y_low = y
	
	#these are the one that need black space crops
	if category == 'hedgehog_0':
		x_low = 10
		x_high = 631
		y_low = 0
		y_high = H
	if category == 'moth':
		x_low = 0
		x_high = W
		y_low = 45
		y_high = 320
	if category == 'peacock_flounder_0' or category == 'peacock_flounder_1' or category == 'peacock_flounder_2':
		x_low = 0
		x_high = W
		y_low = 92
		y_high = 632

	print(category)
	
	for i in range(min(num_images, 300)):
		#read image
		image_dir = os.path.join(category_dir, category, images[i])
		img = cv2.imread(image_dir)
		x = X_s[i]
		y = Y_s[i]
		w = W_s[i]
		h = H_s[i]

		#save original
		anno_img = np.zeros([H, W, 1])
		anno_img[int(y):int(y+h), int(x):int(x+w)] = 255.
		os.makedirs(os.path.join(base_dir, "Annotations_original", category), exist_ok=True)
		save_dir = os.path.join(base_dir, "Annotations_original", category,'{:05d}.png'.format(i))
		cv2.imwrite(save_dir, anno_img)

		#get the cropped
		img_combined = np.concatenate([img, anno_img], -1)
		img_crop = img_combined[y_low:y_high, x_low:x_high]
		img_crop = cv2.resize(img_crop, (W_, H_), cv2.INTER_NEAREST)
		img = img_crop[:,:,:3]
		anno_img = img_crop[:,:,3:4]

		#save cropped
		os.makedirs(os.path.join(base_dir, "Annotations_cropped", category), exist_ok=True)
		os.makedirs(os.path.join(base_dir, "JPEGImages_cropped", category), exist_ok=True)
		save_dir = os.path.join(base_dir, "Annotations_cropped", category,'{:05d}.png'.format(i))
		cv2.imwrite(save_dir, anno_img)
		save_dir = os.path.join(base_dir, "JPEGImages_cropped", category,'{:05d}.jpg'.format(i))
		cv2.imwrite(save_dir, img)