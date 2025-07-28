# AGFormer: An Anchor-Guided Transformer for Class Imbalance in Remote Sensing Change Detection

# Usage
## Requirements
Please refer to `requirements.txt`  and review the tutorials on [open-CD](https://github.com/likyoo/open-cd), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), and [MMdetection](https://github.com/open-mmlab/mmdetection).

## Traning
First, please use `anchor_generator.py` to pre-define class anchors with the following command:
```
python anchor_generator.py --num_centroids 2 --anchor_num 10 --batch_size 16
```
Next, train the AGFormer by following the code:
```
python tools/train.py the path of the cfg file --work-dir the dir path of your workspace
```
Download checkpoints from [Baidu Netdisk](https://pan.baidu.com/s/1ecP9tYC_3JBel8HB6mPJnA?pwd=f3jf).

# citation
```
@article{CHEN2025111839,
title = {AGFormer: An anchor-guided transformer for class imbalance in remote sensing change detection},
journal = {Pattern Recognition},
volume = {168},
pages = {111839},
year = {2025},
author = {Jiaen Chen and Da Wu and Quanqing Ma and Shengjie Xu and Yuchen Zheng},
}
```
