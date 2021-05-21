# Motion Grouping
This code accompanies the paper: Self-supervised Video Object Segmentation by Motion Grouping
Charig Yang, Hala Lamdouar, Erika Lu, Andrew Zisserman, Weidi Xie.
Project page: https://charigyang.github.io/motiongroup/

#### Requirements :
    pytorch (tested on 1.7, although any recent version should work)
    cvbase
    einops
    tensorboardX

#### Datasets :
* DAVIS 2016 can be used as-is.
* The rest has to be converted to DAVIS format. Some helper functions are available in tools.
* MoCA needs to be processed. See Supplementary Material for the paper for details. Helper functions are available in tools. The dataset is also available on google drive: https://drive.google.com/drive/u/2/folders/1x-owzr9Voz65NQghrN_H1LEYDaaQP5n1
* Precomputed flows can be generated from raft/run_inference.py

#### Training :
    python train.py --dataset DAVIS --flow_to_rgb

#### Inference :
    python eval.py --dataset DAVIS  --flow_to_rgb --inference --resume_path {}

#### Benchmark :
* For DAVIS: use the official evaluation code: https://github.com/fperazzi/davis
* For MOCA: use tools/MoCA_eval.py

#### How to use this on your own data :
* Generate optical flow from your dataset using raft/inference.py
* Edit setup_dataset in config.py to include your dataset, and add this to the choices in parser.add_argument('--dataset') in train.py and eval.py
* Follow the training and inference instructions above with your own --dataset argument.

#### Issues/questions/pull requests :
are very welcome.