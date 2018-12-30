# PyTorchCV-SemSeg: Semantic Segmentation ported from PyTorchCV
```
@misc{CV2018,
  author =       {Donny You (youansheng@gmail.com)},
  howpublished = {\url{https://github.com/youansheng/PyTorchCV-SemSeg}},
  year =         {2018}
}
```

This repository provides source code for some deep learning based cv problems. We'll do our best to keep this repository up to date.  If you do find a problem about this repository, please raise it as an issue. We will fix it immediately.


## Implemented Papers

- [Semantic Segmentation](https://github.com/youansheng/PyTorchCV-SemSeg/tree/master/methods)
    - OCNet: Object Context Network for Scene Parsing
    - PSPNet: Pyramid Scene Parsing Network
    - DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation
    - DenseASPP: DenseASPP for Semantic Segmentation in Street Scenes
    


## Performances with PyTorchCV-SemSeg

- CityScapes (Single Scale Whole Image Test)

| Model | Backbone | Train | Test | mIOU | BatchSize | Iters | Scripts |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------|
| [PSPNet](https://drive.google.com/open?id=1bjQ8c-h1IBQPgp7DDwXl-U3tBo1lW6wB) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val | 78.13 | 8 | 40000 | [PSPNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_pspnet_cityscape_seg.sh) |
| [DeepLabV3](https://drive.google.com/open?id=15f--MUIMtiPHL8HyH_2A7EofJIPmA-oa) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val | 79.15 | 8 | 40000 | [DeepLabV3](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_deeplabv3_cityscape_seg.sh) |
| [BaseOCNet](https://drive.google.com/open?id=13-z3PTLMxt2XdcQgP80nddkDS9Jz5SXI) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val/test | 79.72/77.83 | 8 | 40000| [BaseOCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_baseocnet_cityscape_seg.sh) |
| [PyramidOCNet](https://drive.google.com/open?id=1oXiMpIxbcfoFC4xMZmhJ-c3yxpzRFcAS) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val | 78.87 | 8 | 40000| [PyramidOCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_pyramidocnet_cityscape_seg.sh) |
| [AspOCNet](https://drive.google.com/open?id=1TyaDXOeGwP1yy55kYQJd2rQch3QxXzCr) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val/test | 79.52/78.95 | 8 | 40000 | [AspOCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_aspocnet_cityscape_seg.sh) |


- ADE20K (Single Scale Whole Image Test)

| Model | Backbone | Train | Test | mIOU | PixelACC | BatchSize | Iters | Scripts |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|:------|
| [PSPNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 40.65 | 79.64 | 16 | 75000 | [PSPNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_pspnet_ade20k_seg.sh) |
| [PSPNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 40.65 | 79.66 | 16 | 150000 | [PSPNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_pspnet_ade20k_seg.sh) |
| [DeepLabv3](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.15 | 79.81 | 16 | 75000 | [DeepLabV3](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_deeplabv3_ade20k_seg.sh) |
| [DeepLabv3](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 42.24 | 80.29 | 16 | 150000 | [DeepLabV3](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_deeplabv3_ade20k_seg.sh) |
| [BaseOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 40.48 | 79.75 | 16 | 75000 | [BaseOCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_baseocnet_ade20k_seg.sh) |
| [PyramidOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.03 | 79.74 | 16 | 75000 | [PyramidOCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_pyramidocnet_ade20k_seg.sh) |
| [AspOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.47 | 80.05 | 16 | 75000 | [AspOCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_aspocnet_ade20k_seg.sh) |
| [PSPNet](https://drive.google.com/open?id=15C7hcNxzOB6hjRrWVfM-AYBPI4-u_w3m) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | 43.37 | 80.93 | 16 | 150000 | [PSPNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_pspnet_ade20k_seg.sh) |
| [BaseOCNet](https://drive.google.com/open?id=1s-5caZSXy-fL2RYH4-JY2WYBZbd5PTFp) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | 43.21 | 81.01 | 16 | 150000 | [BaseOCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_baseocnet_ade20k_seg.sh) |
| [AspOCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | - | 16 | 150000 | [AspOCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_aspocnet_ade20k_seg.sh) |
