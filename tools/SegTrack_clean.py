import cv2
import glob as gb
import numpy as np


def combine(dir1, dir2):
	for i in range(len(dir1)):
		im1 = cv2.imread(dir1[i])
		im2 = cv2.imread(dir2[i])
		ims = np.clip(im1+im2, 0, 255)
		cv2.imwrite(dir1[i].replace('/1/', '/'), ims)

cats = ['hummingbird', 'drift', 'bmx', 'monkeydog', 'cheetah']
for cat in cats:
	dir1 = sorted(gb.glob('/scratch/shared/beegfs/charig/SegTrackv2/Annotations/{}/1/*.png'.format(cat)))
	dir2 = sorted(gb.glob('/scratch/shared/beegfs/charig/SegTrackv2/Annotations/{}/2/*.png'.format(cat)))
	combine(dir1, dir2)
"""
cat = 'penguin'
dir1 = sorted(gb.glob('/scratch/shared/beegfs/charig/SegTrackv2/Annotations/{}/1/*.png'.format(cat)))
dir2 = sorted(gb.glob('/scratch/shared/beegfs/charig/SegTrackv2/Annotations/{}/2/*.png'.format(cat)))
dir3 = sorted(gb.glob('/scratch/shared/beegfs/charig/SegTrackv2/Annotations/{}/3/*.png'.format(cat)))
dir4 = sorted(gb.glob('/scratch/shared/beegfs/charig/SegTrackv2/Annotations/{}/4/*.png'.format(cat)))
dir5 = sorted(gb.glob('/scratch/shared/beegfs/charig/SegTrackv2/Annotations/{}/5/*.png'.format(cat)))
dir6 = sorted(gb.glob('/scratch/shared/beegfs/charig/SegTrackv2/Annotations/{}/6/*.png'.format(cat)))

for i in range(len(dir1)):
	im1 = cv2.imread(dir1[i])
	im2 = cv2.imread(dir2[i])
	im3 = cv2.imread(dir3[i])
	im4 = cv2.imread(dir4[i])
	im5 = cv2.imread(dir5[i])
	im6 = cv2.imread(dir6[i])
	ims = np.clip(im1+im2+im3+im4+im5+im6, 0, 255)
	cv2.imwrite(dir1[i].replace('/1/', '/'), ims)
"""