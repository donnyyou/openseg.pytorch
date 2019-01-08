# OpenSeg: Open Source for Semantic Segmentation.
```
@misc{CV2018,
  author =       {Donny You (youansheng@gmail.com)},
  howpublished = {\url{https://github.com/youansheng/OpenSeg}},
  year =         {2018}
}
```

This repository provides source code for some deep learning based semantic segmentation. We'll do our best to keep this repository up to date.  If you do find a problem about this repository, please raise it as an issue. We will fix it immediately.


## Implemented Papers

- [Semantic Segmentation](https://github.com/youansheng/OpenSeg/tree/master/methods)
    - OCNet: Object Context Network for Scene Parsing
    - PSPNet: Pyramid Scene Parsing Network
    - DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation
    - DenseASPP: DenseASPP for Semantic Segmentation in Street Scenes
    


## Performances with OpenSeg

- CityScapes (Single Scale Whole Image Test)

| Checkpoints | Backbone | Train | Test | mIOU | BS | Iters | Scripts |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------|
| [PSPNet](https://drive.google.com/open?id=1bjQ8c-h1IBQPgp7DDwXl-U3tBo1lW6wB) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | 78.13 | 8 | 4W | [PSPNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/cityscapes/run_fs_pspnet_cityscapes_seg.sh) |
| [DeepLabV3](https://drive.google.com/open?id=15f--MUIMtiPHL8HyH_2A7EofJIPmA-oa) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | 79.15 | 8 | 4W | [DeepLabV3](https://github.com/youansheng/OpenSeg/blob/master/scripts/cityscapes/run_fs_deeplabv3_cityscapes_seg.sh) |
| [BaseOCNet](https://drive.google.com/open?id=13-z3PTLMxt2XdcQgP80nddkDS9Jz5SXI) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val/test | 79.72/77.83 | 8 | 4W| [BaseOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/cityscapes/run_fs_baseocnet_cityscapes_seg.sh) |
| [PyramidOCNet](https://drive.google.com/open?id=1oXiMpIxbcfoFC4xMZmhJ-c3yxpzRFcAS) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | 78.87 | 8 | 4W| [PyramidOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/cityscapes/run_fs_pyramidocnet_cityscapes_seg.sh) |
| [AspOCNet](https://drive.google.com/open?id=1TyaDXOeGwP1yy55kYQJd2rQch3QxXzCr) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val/test | 79.52/78.95 | 8 | 4W | [AspOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/cityscapes/run_fs_aspocnet_cityscapes_seg.sh) |
| [FastBaseOCNet1](https://drive.google.com/open?id=1e_xe2rcPZ1YmeWCK-x6wbxNv1xU0zhb4)<br>[FastBaseOCNet2](https://drive.google.com/open?id=1d7nDj5cdxeMCiUdC4wd6E8-8mW2TvnGI)<br>[FastBaseOCNet3](https://drive.google.com/open?id=1nBmP7ZzoN8MgbHuPJonh0w23FOCcF9qk) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | train/val/test | 85.57/78.70/78.22<br>85.47/79.72/77.15<br>85.50/77.01/77.67 | 8 | 4W| [FastBaseOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/cityscapes/run_fs_fastbaseocnet_cityscapes_seg.sh) |
| [FastAspOCNet1](https://drive.google.com/open?id=1vGNC0pUMhJaS_b0Xn2-p968iityGY2lu)<br>[FastAspOCNet2](https://drive.google.com/open?id=15ojuzRS9_xFzSsT5GOlJkn2YbibCA3J_)<br>[FastAspOCNet3](https://drive.google.com/open?id=1pibZba-l7rGgGcprYkzspiV5SeTfaIPh) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | train/val/test | 86.32/80.13/78.50<br>86.32/79.28/79.21<br>86.38/79.96/78.60 | 8 | 4W | [FastAspOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/cityscapes/run_fs_fastaspocnet_cityscapes_seg.sh) |


- ADE20K (Single Scale Whole Image Test): Base LR 0.02, Crop Size 520

| Checkpoints | Backbone | Train | Test | mIOU | PixelACC | BatchSize | Iters | Scripts |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|:------|
| [PSPNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 40.98 | 79.75 | 16 | 7.5W | [PSPNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_pspnet_ade20k_seg.sh) |
| [DeepLabv3](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.49 | 80.09 | 16 | 7.5W | [DeepLabV3](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_deeplabv3_ade20k_seg.sh) |
| [BaseOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.61 | 79.85 | 16 | 7.5W | [BaseOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_baseocnet_ade20k_seg.sh) |
| [PyramidOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | - | - | 16 | 7.5W | [PyramidOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_pyramidocnet_ade20k_seg.sh) |
| [AspOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.93 | 80.12 | 16 | 7.5W | [AspOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_aspocnet_ade20k_seg.sh) |
| [FastBaseOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 40.92 | 80.05 | 16 | 7.5W | [FastBaseOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_baseocnet_ade20k_seg.sh) |
| [FastAspOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 42.45 | 80.28 | 16 | 7.5W | [FastAspOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_aspocnet_ade20k_seg.sh) |
| [PSPNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | - | - | 16 | 15W | [PSPNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_pspnet_ade20k_seg.sh) |
| [DeepLabv3](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | - | - | 16 | 15W | [DeepLabV3](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_deeplabv3_ade20k_seg.sh) |
| [BaseOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | - | - | 16 | 15W | [BaseOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_baseocnet_ade20k_seg.sh) |
| [PyramidOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | - | - | 16 | 15W | [PyramidOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_pyramidocnet_ade20k_seg.sh) |
| [AspOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 42.76 | 80.62 | 16 | 15W | [AspOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res50_aspocnet_ade20k_seg.sh) |
| [PSPNet](https://drive.google.com/open?id=15C7hcNxzOB6hjRrWVfM-AYBPI4-u_w3m) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | - | 16 | 15W | [PSPNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res101_pspnet_ade20k_seg.sh) |
| [DeepLabv3](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | - | 16 | 15W | [DeepLabV3](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res101_deeplabv3_ade20k_seg.sh) |
| [BaseOCNet](https://drive.google.com/open?id=1s-5caZSXy-fL2RYH4-JY2WYBZbd5PTFp) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | - | 16 | 15W | [BaseOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res101_baseocnet_ade20k_seg.sh) |
| [PyramidOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | - | 16 | 15W | [PyramidOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res101_pyramidocnet_ade20k_seg.sh) |
| [AspOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | - | 16 | 15W | [AspOCNet](https://github.com/youansheng/OpenSeg/blob/master/scripts/ade20k/run_fs_res101_aspocnet_ade20k_seg.sh) |
