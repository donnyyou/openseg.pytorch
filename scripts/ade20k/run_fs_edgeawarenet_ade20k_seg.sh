#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON=python

cd ../../

LOG_FILE="./log/cityscape/fs_edgeawarenet_ade20k_seg.log"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_edgeawarenet_ade20k_seg.json \
                       --phase train --gathered n --loss_balance y --gpu 0 1 2 3 \
                       --pretrained ./pretrained_model/resnet101-imagenet.pth

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_edgeawarenet_ade20k_seg.json --phase debug --gpu 0

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_edgeawarenet_ade20k_seg.json --phase test --gpu 0 --resume $2
  cd val/scripts
  ${PYTHON} -u cityscape_evaluator.py --hypes_file ../../hypes/ade20k/fs_edgeawarenet_ade20k_seg.json \
                                      --gt_dir path-to-gt \
                                      --pred_dir path-to-pred
else
  echo "$1"x" is invalid..."
fi