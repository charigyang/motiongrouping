import os
import cv2
import glob
import torch
import random
import einops
import numpy as np
from utils import read_flo
from torch.utils.data import Dataset
from cvbase.optflow.visualize import flow2rgb


def readFlow(sample_dir, resolution, to_rgb):
    flow = read_flo(sample_dir)
    h, w, _ = np.shape(flow)
    flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] = flow[:, :, 0] * resolution[1] / w
    flow[:, :, 1] = flow[:, :, 1] * resolution[0] / h
    if to_rgb: flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1., 1.)
    return einops.rearrange(flow, 'h w c -> c h w')

def readRGB(sample_dir, resolution):
    rgb = cv2.imread(sample_dir)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = ((rgb / 255.0) - 0.5) * 2.0
    rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(rgb, -1., 1.)
    return einops.rearrange(rgb, 'h w c -> c h w')

def readSeg(sample_dir):
    gt = cv2.imread(sample_dir) / 255
    return einops.rearrange(gt, 'h w c -> c h w')

class FlowPair(Dataset):
    def __init__(self, data_dir, resolution, to_rgb=False, with_rgb=False, with_gt=True):
        self.eval = eval
        self.to_rgb = to_rgb
        self.with_rgb = with_rgb
        self.data_dir = data_dir
        self.flow_dir = data_dir[0]
        self.resolution = resolution
        self.with_gt = with_gt

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        flowgaps = random.choice(list(self.flow_dir.values()))
        vid = random.choice(flowgaps)
        flos = random.choice(vid)
        rgbs, flows, gts, imgdirs = [], [], [], []

        for flo in flos:
            flosplit = flo.split(os.sep)
            rgb_dir = os.path.join(self.data_dir[1], flosplit[-2], flosplit[-1]).replace('.flo','.jpg')
            gt_dir = os.path.join(self.data_dir[2], flosplit[-2], flosplit[-1]).replace('.flo','.png')
            img_dir = gt_dir.split('/')[-2:]

            flows.append(readFlow(str(flo), self.resolution, self.to_rgb))
            if self.with_rgb: rgbs.append(readRGB(rgb_dir, self.resolution))
            if self.with_gt: gts.append(readSeg(gt_dir))
            imgdirs.append(img_dir)

        out = np.stack(flows, 0) if not self.with_rgb else np.stack([np.stack(flows, 0), np.stack(rgbs, 0)], -1)
        gt_out = np.stack(gts, 0) if self.with_gt else 0
        return out, gt_out


class FlowEval(Dataset):
    def __init__(self, data_dir, resolution, pair_list, val_seq, to_rgb=False, with_rgb=False):
        self.val_seq = val_seq
        self.to_rgb = to_rgb
        self.with_rgb = with_rgb
        self.data_dir = data_dir
        self.pair_list = pair_list
        self.resolution = resolution

        self.samples = []
        for v in self.val_seq:
            self.samples.extend(sorted(glob.glob(os.path.join(self.data_dir[1], v, '*.jpg'))))
        
        self.samples = [os.path.join(x.split('/')[-2], x.split('/')[-1]) for x in self.samples]
        self.gaps = ['gap{}'.format(i) for i in pair_list]
        self.neg_gaps = ['gap{}'.format(-i) for i in pair_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        out = []
        fgap = []
        for gap, _gap in zip(self.gaps, self.neg_gaps):
            flow_dir = os.path.join(self.data_dir[0], self.samples[idx]).replace('gap1', gap).replace('.jpg', '.flo')
            if os.path.exists(flow_dir):
                out.append(readFlow(flow_dir, self.resolution, self.to_rgb))
                fgap.append(gap)
            else:
                flow_dir = os.path.join(self.data_dir[0], self.samples[idx]).replace('gap1', _gap).replace('.jpg', '.flo')
                out.append(readFlow(flow_dir, self.resolution, self.to_rgb))
                fgap.append(_gap)
        out = np.stack(out, 0)

        if self.with_rgb:
            rgb_dir = os.path.join(self.data_dir[1], self.samples[idx])
            out = np.stack([out, readRGB(rgb_dir, self.resolution)], -1)

        gt_dir = os.path.join(self.data_dir[2], self.samples[idx]).replace('.jpg', '.png')
        img_dir = gt_dir.split('/')[-2:]
        return out, readSeg(gt_dir), img_dir, fgap
