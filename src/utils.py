import os
import torch
import einops
import cv2
import numpy as np
import torch.nn.functional as F
import os
from cvbase.optflow.visualize import flow2rgb


def set_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def convert_for_vis(inp, use_flow=False):
    dim = len(inp.size())
    if not use_flow:
        return torch.clamp((0.5*inp+0.5)*255,0,255).type(torch.ByteTensor)
    else:
        if dim == 4:
            inp = einops.rearrange(inp, 'b c h w -> b h w c').detach().cpu().numpy()
            rgb = [flow2rgb(inp[x]) for x in range(np.shape(inp)[0])]
            rgb = np.stack(rgb, axis=0)
            rgb = einops.rearrange(rgb, 'b h w c -> b c h w')
        if dim == 5:
            b, s, w, h, c = inp.size()
            inp = einops.rearrange(inp, 'b s c h w -> (b s) h w c').detach().cpu().numpy()
            rgb = [flow2rgb(inp[x]) for x in range(np.shape(inp)[0])]
            rgb = np.stack(rgb, axis=0)
            rgb = einops.rearrange(rgb, '(b s) h w c -> b s c h w', b=b, s=s)
        return torch.Tensor(rgb*255).type(torch.ByteTensor)

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
    return np.sum(mask>0.5)/borders

def rectangle_iou(masks, gt):
    t, s, c, H_, W_ = masks.size()
    H, W = gt.size()
    masks = F.interpolate(masks, size=(1, H, W))
    ms = []
    for t_ in range(t):
        m = masks[t_,0,0] #h w
        m = m.detach().cpu().numpy()
        if heuristic_fg_bg(m) > 0.5: m = 1-m
        ms.append(m)
    masks = np.stack(ms, 0)
    gt = gt.detach().cpu().numpy()
    for idx, m in enumerate([masks[0], masks.mean(0)]):
        m[m>0.1]=1
        m[m<=0.1]=0
        contours = cv2.findContours((m*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        area = 0
        for cnt in contours:
            (x_,y_,w_,h_) = cv2.boundingRect(cnt)
            if w_*h_ > area:
                x=x_; y=y_; w=w_; h=h_;
                area = w_ * h_
        if area>0:
            bbox = np.array([x, y, x+w, y+h],dtype=float)
            #if the size reference for the annotation (the original jpg image) is different than the size of the mask
            i, j = np.where(gt==1.)
            bbox_gt = np.array([min(j), min(i), max(j)+1, max(i)+1],dtype=float)
            iou = bb_intersection_over_union(bbox_gt,bbox)
        else:
            iou = 0.
        if idx == 0: iou_single = iou
        if idx == 1: iou_mean = iou
    masks = np.expand_dims(masks, 1)
    return masks, masks.mean(0), iou_mean, iou_single

def iou(masks, gt, thres=0.5):
    masks = (masks>thres).float()
    intersect = torch.tensordot(masks, gt, dims=([-2, -1], [0, 1]))
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    return intersect/(union + 1e-12)

def ensemble_hungarian_iou(masks, gt, moca=False):
    thres = 0.5
    b, c, h, w = gt.size()
    gt = gt[0,0,:,:] #h ,w

    if moca:
        #return masks, masks.mean(0), 0, rectangle_iou(masks[0], gt) 
        masks, mean_mask, iou_mean, iou_single_gap = rectangle_iou(masks, gt)
    else:
        masks = F.interpolate(masks, size=(1, h, w))  # t s 1 h w
        mask_iou = iou(masks[:,:,0], gt, thres)  # t s # t s
        iou_max, slot_max = mask_iou.max(dim=1)
        masks = masks[torch.arange(masks.size(0)), slot_max]  # pick the slot for each mask
        mean_mask = masks.mean(0)
        gap_1_mask = masks[0]  # note last frame will use gap of -1, not major.
        iou_mean = iou(mean_mask, gt, thres).detach().cpu().numpy()
        iou_single_gap = iou(gap_1_mask, gt, thres).detach().cpu().numpy()
        mean_mask = mean_mask.detach().cpu().numpy()  # c h w
        masks = masks.detach().cpu().numpy()

    return masks, mean_mask, iou_mean, iou_single_gap


def hungarian_iou(masks, gt):
    thres = 0.5
    masks = (masks>thres).float()
    gt = gt[:,0:1,:,:]
    b, c, h, w = gt.size()
    iou_max = []
    for i in range(masks.size(1)):
        mask = masks[:,i,:,:,:]
        mask = F.interpolate(mask, size=(h, w))
        #IOU
        intersect = (mask*gt).sum(dim=[-2, -1])
        union = mask.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
        iou = intersect/(union + 1e-12)
        iou_max += [iou]
    iou_max, slot_max = torch.cat(iou_max, -1).max(dim=-1)
    return iou_max.mean(), slot_max


TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
