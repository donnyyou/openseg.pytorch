#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

cd ../../

LOG_FILE="./log/cityscape/fs_baseocnet_cityscape_syncbn_sgd_seg.log"


if [ "$1"x == "train"x ]; then
  $PYTHON main.py --hypes hypes/cityscape/fs_baseocnet_cityscape_seg_sgd.json \
                  --phase train --gathered n --loss_balance y --bn_type syncbn \
                  --checkpoints_name fs_baseocnet_cityscape_syncbn_sgd_seg \
                  --data_dir /teamscratch/msravcshare/v-ansheng/DataSet/CityScape \
                  --pretrained ./pretrained_model/resnet101-imagenet.pth  > $LOG_FILE 2>&1

elif [ "$1"x == "debug"x ]; then
  $PYTHON main.py --hypes hypes/cityscape/fs_baseocnet_cityscape_seg_sgd.json --phase debug --gpu 0

elif [ "$1"x == "test"x ]; then
  $PYTHON main.py --hypes hypes/cityscape/fs_baseocnet_cityscape_seg_sgd.json --phase test --gpu 0 --resume $2
  cd val/scripts
  $PYTHON cityscape_evaluator.py --hypes_file ../../hypes/cityscape/fs_baseocnet_cityscape_seg_sgd.json \
                                 --gt_dir path-to-gt \
                                 --pred_dir path-to-pred
else
  echo "$1"x" is invalid..."
fi