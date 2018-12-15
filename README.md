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
    - DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation
    - PSPNet: Pyramid Scene Parsing Network
    - DenseASPP: DenseASPP for Semantic Segmentation in Street Scenes
    


## Performances with PyTorchCV-SemSeg

#### CityScapes

| Model | Backbone | Training data  | Testing data | mIOU | Setting |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [Base OCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-ResNet101](https://drive.google.com/open?id=1ROewKyaGPynox_-a50wHkSv1-0jYWyvc) | CityScapes train | CityScapes val | 79.13 | [PSPNet](https://github.com/youansheng/PyTorchCV/blob/master/hypes/seg/ade20k/fs_pspnet_ade20k_seg.json) |


