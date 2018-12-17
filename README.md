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

#### CityScapes (Single Scale Whole Image Test)

| Model | Backbone | Training data  | Testing data | mIOU | Setting |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [PSPNet](https://drive.google.com/open?id=1nnQJ9U14eDxaPE1KvV5C8iuUFBwiHQL6) | [3x3-ResNet101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | CityScapes train | CityScapes val | 79.05 | [PSPNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/hypes/cityscape/fs_pspnet_cityscape_seg.json) |
| [Base-OCNet](https://drive.google.com/open?id=1n4yYrVq1lzT7Q0HhMgbB2TJcppAjmuT0) | [3x3-ResNet101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | CityScapes train | CityScapes val | 79.16 | [Base-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/hypes/cityscape/fs_baseocnet_cityscape_seg.json) |
| [ASP-OCNet](https://drive.google.com/open?id=1_jPHJmqnej6tCK3CB2YSDK3xTU6AhYMW) | [3x3-ResNet101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | CityScapes train | CityScapes val | 78.81 | [ASP-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/hypes/cityscape/fs_aspocnet_cityscape_seg.json) |


#### ADE20K (Single Scale Whole Image Test)
| Model | Backbone | Training data  | Testing data | mIOU | PixelACC | Setting |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|
| [PSPNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-ResNet101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | ADE20K train | ADE20K val | - | - | [PSPNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/hypes/ade20k/fs_pspnet_ade20k_seg.json) |
| [Base-OCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-ResNet101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | ADE20K train | ADE20K val | - | - | [Base-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/hypes/ade20k/fs_baseocnet_ade20k_seg.json) |
| [ASP-OCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-ResNet101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | ADE20K train | ADE20K val | - | - | [ASP-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/hypes/ade20k/fs_aspocnet_ade20k_seg.json) |
