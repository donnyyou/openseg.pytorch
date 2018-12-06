#!/usr/bin/env bash


cd ../../


if [ $1 == "train" ]; then
  python main.py --hypes hypes/ade20k/fs_pspnet_ade20k.json
                 --phase train --gpu 0 1 2 3
                 --pretrained ../imagenet_101.pth

elif [ $1 == "debug" ]; then
  python main.py --hypes hypes/ade20k/fs_pspnet_ade20k.json --phase debug --gpu 0

elif [ $1 == "test" ]; then
  python main.py --hypes hypes/ade20k/fs_pspnet_ade20k.json --phase test --gpu 0 --resume $2
  cd val/scripts/ade20k
  python ade20k_evaluator.py --hypes_file ././../../hypes/ade20k/fs_pspnet_ade20k.json
                             --gt_dir path-to-gt
                             --pred_dir path-to-pred
else
  echo $1" is invalid..."
fi