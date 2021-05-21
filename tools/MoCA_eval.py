import os
import numpy as np
import cv2
import ast
import operator
import csv
import glob
import pandas as pd
from argparse import ArgumentParser

def read_annotation(annotation):
	reader = csv.reader(open(annotation, 'r'))
	next(reader, None) 
	d = {}
	reader = sorted(reader, key=operator.itemgetter(1))
	for row in reader:
		_, fn, bbox, motion = row
		if bbox != '[]':
			if motion == '{}':
				motion = old_motion
			old_motion = motion
			name = fn.split('/')[-2]
			number = fn.split('/')[-1][:-4]
			if name not in d:
				d[name] = {}
			if number not in d[name]:
				d[name][number] = {}
			d[name][number]['fn'] = fn
			motion = ast.literal_eval(motion)
			d[name][number]['motion'] = motion['1']
			bbox = ast.literal_eval(bbox)
			_, xmin, ymin, width, height = list(bbox)
			xmin = max(xmin, 0.)
			ymin = max(ymin, 0.)
			d[name][number]['xmin'] = xmin
			d[name][number]['xmax'] = xmin + width
			d[name][number]['ymin'] = ymin
			d[name][number]['ymax'] = ymin + height
	return d

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def heuristic_fg_bg(mask):
    mask = mask.copy()
    h, w = mask.shape
    mask[1:-1, 1:-1] = 0
    borders = 2*h+2*w-4
    return np.sum(mask>127.5)/borders

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(args):
    #create output directory, and csv file
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    cout_csv = os.path.join(args.out_dir,'MoCA_results.csv')
    
    with open(cout_csv, 'w') as f:
       df = pd.DataFrame([], columns = ['Seq_name','Locomotion_IoU','Locomotion_S_0.5','Locomotion_S_0.6','Locomotion_S_0.7','Locomotion_S_0.8','Locomotion_S_0.9',
                                        'Deformation_IoU','Deformation_S_0.5','Deformation_S_0.6','Deformation_S_0.7','Deformation_S_0.8','Deformation_S_0.9',
                                        'Static_IoU','Static_S_0.5','Static_S_0.6','Static_S_0.7','Static_S_0.8','Static_S_0.9',
                                        'All_motion_IoU','All_motion_S_0.5','All_motion_S_0.6','All_motion_S_0.7','All_motion_S_0.8','All_motion_S_0.9',
                                        'locomotion_num','deformation_num','static_num'])
       
       df.to_csv(f, index=False, columns =  ['Seq_name','Locomotion_IoU','Locomotion_S_0.5','Locomotion_S_0.6','Locomotion_S_0.7','Locomotion_S_0.8','Locomotion_S_0.9',
                                        'Deformation_IoU','Deformation_S_0.5','Deformation_S_0.6','Deformation_S_0.7','Deformation_S_0.8','Deformation_S_0.9',
                                        'Static_IoU','Static_S_0.5','Static_S_0.6','Static_S_0.7','Static_S_0.8','Static_S_0.9',
                                        'All_motion_IoU','All_motion_S_0.5','All_motion_S_0.6','All_motion_S_0.7','All_motion_S_0.8','All_motion_S_0.9',
                                        'locomotion_num','deformation_num','static_num'])
       pass

    annotations = read_annotation(args.MoCA_csv)
    Js = AverageMeter()
    

    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    success_rates_overall = np.zeros(5)
    total_frames_l = 0
    total_frames_d = 0
    total_frames_s = 0
    success_l_overall = [0, 0, 0, 0, 0]
    success_d_overall = [0, 0, 0, 0, 0]
    success_s_overall = [0, 0, 0, 0, 0]
    
    video_names =  sorted(os.listdir(args.masks_dir))
    for video in video_names:
        if not os.path.exists(os.path.join(args.out_dir, video)):
            os.mkdir(os.path.join(args.out_dir, video))
        res_path = os.path.join(args.masks_dir,video)
        res_list = sorted([f for f in glob.glob(res_path+'/gap1/*.png' , recursive=False)]) #for our model

        n_frames = len(res_list)
        js=[]
        if video not in annotations:
             continue
         
        iou_static = AverageMeter()
        iou_locomotion = AverageMeter()
        iou_deformation = AverageMeter()
        ns = 0; nl =0; nd =0
        success_l = [0, 0, 0, 0, 0]
        success_d = [0, 0, 0, 0, 0]
        success_s = [0, 0, 0, 0, 0]
        if args.resize:
            image = np.array(cv2.imread(os.path.join(args.MoCA_dir,'JPEGImages',video,'{:05d}.jpg'.format(0))))
            H, W, _ = image.shape
        for ff in range(n_frames):
            number = str(ff).zfill(5)
            if number in annotations[video]:
                #get annotation
                motion = annotations[video][number]['motion']
                x_min = annotations[video][number]['xmin']
                x_max = annotations[video][number]['xmax']
                y_min = annotations[video][number]['ymin']
                y_max = annotations[video][number]['ymax']
                bbox_gt = [x_min,y_min,x_max,y_max]
                
                

                #get mask
                mask = np.array(cv2.imread(res_list[ff]), dtype=np.uint8)
                if len(mask.shape)>2:
                    mask = mask.mean(2)
                H_, W_ = mask.shape

                if heuristic_fg_bg(mask) > 0.5:
                    mask = (255-mask).astype(np.uint8)

                thres = 0.1*255
                mask[mask>thres]=255
                mask[mask<=thres]=0
                
                
                    
                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                area = 0

                for cnt in contours:
                    (x_,y_,w_,h_) = cv2.boundingRect(cnt)
                    if w_*h_ > area:
                        x=x_; y=y_; w=w_; h=h_;
                        area = w_ * h_
                H_, W_ = mask.shape
                
                
                #cv2.imwrite(os.path.join(args.out_dir, video, number +'.png'), mask)
                if area>0:
                    bbox = np.array([x, y, x+w, y+h],dtype=float)
                    #if the size reference for the annotation (the original jpg image) is different from the size of the mask
                    if args.resize:
                        bbox[0]*= W/W_; bbox[2]*= W/W_
                        bbox[1]*= H/H_; bbox[3]*= H/H_
                    iou = bb_intersection_over_union(np.array(bbox_gt,dtype=float),np.array(bbox,dtype=float))
                else:
                    iou = 0.
                js.append(iou)

                #get motion
                if motion == '1':
                    iou_deformation.update(iou)
                    nd+= 1
                    for k in range(len(thresholds)):
                        success_d[k]+=int(iou>thresholds[k])

                elif motion == '0':
                    iou_locomotion.update(iou)
                    nl+= 1
                    for k in range(len(thresholds)):
                        success_l[k]+=int(iou>thresholds[k])

                elif motion == '2':
                    iou_static.update(iou)
                    ns+= 1
                    for k in range(len(thresholds)):
                        success_s[k]+=int(iou>thresholds[k])
            
        total_frames_l += nl   
        total_frames_s += ns   
        total_frames_d += nd
        for k in range(len(thresholds)):
            success_l_overall[k]+=success_l[k]
            success_s_overall[k]+=success_s[k]
            success_d_overall[k]+=success_d[k]
            
        js_m = np.average(js) 
        locomotion_val = -1.
        static_val = -1.
        deformation_val = -1.
        if iou_locomotion.count>0:
                locomotion_val = iou_locomotion.avg
        if iou_deformation.count>0:
                deformation_val = iou_deformation.avg
        if iou_static.count>0:
                static_val = iou_static.avg
                
        all_motion_S = np.array(success_l)+np.array(success_s)+np.array(success_d)
        success_rates_overall+=all_motion_S
        with open(cout_csv, 'a') as f:
                raw_data = {'Seq_name':video,'Locomotion_IoU': [locomotion_val],
                            'Locomotion_S_0.5' :[success_l[0]],'Locomotion_S_0.6' :[success_l[1]],
                            'Locomotion_S_0.7' :[success_l[2]],'Locomotion_S_0.8' :[success_l[3]],
                            'Locomotion_S_0.9' :[success_l[4]],
                            'Deformation_IoU': [deformation_val],
                            'Deformation_S_0.5' :[success_d[0]],'Deformation_S_0.6' :[success_d[1]],
                            'Deformation_S_0.7' :[success_d[2]],'Deformation_S_0.8' :[success_d[3]],
                            'Deformation_S_0.9' :[success_d[4]],
                            'Static_IoU': [static_val],
                            'Static_S_0.5' :[success_s[0]],'Static_S_0.6' :[success_s[1]],
                            'Static_S_0.7' :[success_s[2]],'Static_S_0.8' :[success_s[3]],
                            'Static_S_0.9' :[success_s[4]],
                            'All_motion_IoU': [js_m],
                            'All_motion_S_0.5' :[all_motion_S[0]],'All_motion_S_0.6' :[all_motion_S[1]],
                            'All_motion_S_0.7' :[all_motion_S[2]],'All_motion_S_0.8' :[all_motion_S[3]],
                            'All_motion_S_0.9' :[all_motion_S[4]],
                            'locomotion_num': [nl],'deformation_num': [nd],'static_num': [ns]}
                df = pd.DataFrame(raw_data, columns = ['Seq_name','Locomotion_IoU','Locomotion_S_0.5','Locomotion_S_0.6','Locomotion_S_0.7','Locomotion_S_0.8','Locomotion_S_0.9',
                                        'Deformation_IoU','Deformation_S_0.5','Deformation_S_0.6','Deformation_S_0.7','Deformation_S_0.8','Deformation_S_0.9',
                                        'Static_IoU','Static_S_0.5','Static_S_0.6','Static_S_0.7','Static_S_0.8','Static_S_0.9',
                                        'All_motion_IoU','All_motion_S_0.5','All_motion_S_0.6','All_motion_S_0.7','All_motion_S_0.8','All_motion_S_0.9',
                                        'locomotion_num','deformation_num','static_num'])
                df.to_csv(f, header=False, index=False)
        Js.update(js_m)
        info = video+' processed'
        print(info)


    success_rates_overall = success_rates_overall/(total_frames_l+total_frames_s+total_frames_d)
    info = 'Overall :  Js: ({:.3f}). SR at '
    for k in range(len(thresholds)):
        info += str(thresholds[k])
        info += ': ({:.3f}), '
    info = info.format(Js.avg, success_rates_overall[0], success_rates_overall[1], success_rates_overall[2], success_rates_overall[3], success_rates_overall[4])
    
    print(info)

    return



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--masks_dir', type=str, default='')
    parser.add_argument('--resize', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default='../vis_final/MoCA')
    parser.add_argument('--MoCA_dir', type=str, default='/path/to/MoCA_filtered')
    parser.add_argument('--MoCA_csv', type=str, default='/path/to/MoCA_filtered/annotations.csv')
    args = parser.parse_args()
    main(args)


