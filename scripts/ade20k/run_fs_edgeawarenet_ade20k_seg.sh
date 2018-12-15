#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
export PYTHONPATH=/home/yishendiman/Projects/donny/PyTorchCV-SemSeg:$PYTHONPATH
PYTHON=python

cd ../../

LOG_FILE="./log/cityscape/fs_edgeawarenet_ade20k_seg.log"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_edgeawarenet_ade20k_seg.json \
                       --phase train --gathered n --loss_balance y --gpu 2 3 4 5 \
                       --data_dir /mnt/hdd/yizhaoguangyu/data/ade20k \
                       --pretrained ./pretrained_model/resnet101-imagenet.pth

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_edgeawarenet_ade20k_seg.json --phase debug --gpu 0

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_edgeawarenet_ade20k_seg.json --phase test --gpu 2 \
                       --test_dir /mnt/hdd/yizhaoguangyu/data/ade20k/val/image/ \
                       --resume checkpoints/ade20k_6/fs_edgeawarenet_ade20k_seg_max_performance.pth
  cd val/scripts
  ${PYTHON} -u ade20k_evaluator.py --hypes_file hypes/ade20k/fs_edgeawarenet_ade20k_seg.json \
                                   --gt_dir /mnt/hdd/yizhaoguangyu/data/ade20k/val/label  \
                                   --pred_dir ./val/results/ade20k_6/test_dir/image/label

elif [ "$1"x == "val"x ]; then
  cd val/scripts
  ${PYTHON} -u ade20k_evaluator.py --hypes_file ../../hypes/ade20k/fs_edgeawarenet_ade20k_seg.json \
                                   --gt_dir /mnt/hdd/yizhaoguangyu/data/ade20k/val/label  \
                                   --pred_dir ../results/ade20k_6/test_dir/image/label

else
  echo "$1"x" is invalid..."
fi
