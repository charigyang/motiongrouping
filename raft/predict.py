import sys
sys.path.append('core')

import os
import cv2
import glob
import torch
import argparse
import numpy as np

import PIL
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!")
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


def load_image(imfile, resolution=None):
    img = Image.open(imfile)
    if resolution:
        img = img.resize(resolution, PIL.Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def predict(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        folder = os.path.basename(args.path)
        floout = os.path.join(args.outroot, folder)
        rawfloout = os.path.join(args.raw_outroot, folder)

        os.makedirs(floout, exist_ok=True)
        os.makedirs(rawfloout, exist_ok=True)

        gap = args.gap
        images = sorted(images)
        images_ = images[:-gap]

        for index, imfile1 in enumerate(images_):
            if args.reverse:
                image1 = load_image(images[index+gap])
                image2 = load_image(imfile1)
                svfile = images[index+gap]
            else:
                image1 = load_image(imfile1)
                image2 = load_image(images[index + gap])
                svfile = imfile1

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            flopath = os.path.join(floout, os.path.basename(svfile))
            rawflopath = os.path.join(rawfloout, os.path.basename(svfile))

            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
            
            # save raw flow
            writeFlowFile(rawflopath[:-4]+'.flo', flo)

            # save image.
            flo = flow_viz.flow_to_image(flo)
            cv2.imwrite(flopath[:-4]+'.png', flo[:, :, [2, 1, 0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--resolution', nargs='+', type=int)
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for prediction")
    parser.add_argument('--gap', type=int, help="gap between frames")
    parser.add_argument('--outroot', help="path for output flow as image")
    parser.add_argument('--reverse', type=int, help="video forward or backward")
    parser.add_argument('--raw_outroot', help="path for output flow as xy displacement")

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    predict(args)
