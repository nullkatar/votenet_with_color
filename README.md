# Deep Hough Voting for 3D Object Detection in Point Clouds
Created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="https://orlitany.github.io/" target="_blank">Or Litany</a>, <a href="http://kaiminghe.com/" target="_blank">Kaiming He</a> and <a href="https://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas Guibas</a> from <a href="https://research.fb.com/category/facebook-ai-research/" target="_blank">Facebook AI Research</a> and <a href="http://www.stanford.edu" target="_blank">Stanford University</a>.

Modifyed by Kochiev Leon

![teaser](https://github.com/facebookresearch/votenet/blob/master/doc/teaser.jpg)

## Introduction
This repository contains code for [VoteNet](https://arxiv.org/pdf/1904.09664.pdf) model with color prediction. 

## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs. Matlab is required to prepare data for SUN RGB-D. The code is tested with Ubuntu 18.04, Pytorch v1.1, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4. Note: After a code update on 2/6/2020, the code is now also compatible with Pytorch v1.2+

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

To see if the compilation is successful, try to run `python models/votenet.py` to see if a forward pass works.

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

## Training and evaluating

### Data preparation

For ScanNet, follow the [README](https://github.com/facebookresearch/votenet/blob/master/scannet/README.md) under the `scannet` folder.

### Train and test on ScanNet

To train a VoteNet model on Scannet data (fused scan):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset scannet --log_dir log_scannet --num_point 40000

To test the trained model with its checkpoint:

    python eval.py --dataset scannet --checkpoint_path log_scannet/checkpoint.tar --dump_dir eval_scannet --num_point 40000 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

Example results will be dumped in the `eval_scannet` folder (or any other folder you specify). In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on axis aligned boxes. A properly trained VoteNet should have around 58 mAP@0.25 and 35 mAP@0.5.

### Train on your own data

[For Pro Users] If you have your own dataset with point clouds and annotated 3D bounding boxes, you can create a new dataset class and train VoteNet on your own data. To ease the proces, some tips are provided in this [doc](https://github.com/facebookresearch/votenet/blob/master/doc/tips.md).

## Acknowledgements
We want to thank Erik Wijmans for his PointNet++ implementation in Pytorch ([original codebase](https://github.com/erikwijmans/Pointnet2_PyTorch)).

## License
votenet is relased under the MIT License. See the [LICENSE file](https://arxiv.org/pdf/1904.09664.pdf) for more details.

