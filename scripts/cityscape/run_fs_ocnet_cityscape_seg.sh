#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

cd ../../

STD_LOG_FILE="./log/std_log.log"+"`date +%Y-%m-%d,%H:%m:%s`"


if [ "$1"x == "train"x ]; then
  $PYTHON main.py --hypes hypes/cityscape/fs_baseocnet_cityscape_seg.json \
                  --phase train --gathered n --loss_balance y \
                  --data_dir /teamscratch/msravcshare/v-ansheng/DataSet/CityScape \
                  --pretrained ./pretrained_model/resnet101-imagenet.pth  > $STD_LOG_FILE 2>&1

elif [ "$1"x == "debug"x ]; then
  $PYTHON main.py --hypes hypes/cityscape/fs_baseocnet_cityscape_seg.json --phase debug --gpu 0

elif [ "$1"x == "test"x ]; then
  $PYTHON main.py --hypes hypes/cityscape/fs_baseocnet_cityscape_seg.json --phase test --gpu 0 --resume $2
  cd val/scripts
  $PYTHON cityscape_evaluator.py --hypes_file ../../hypes/cityscape/fs_baseocnet_cityscape_seg.json \
                                 --gt_dir path-to-gt \
                                 --pred_dir path-to-pred
else
  echo "$1"x" is invalid..."
fi
