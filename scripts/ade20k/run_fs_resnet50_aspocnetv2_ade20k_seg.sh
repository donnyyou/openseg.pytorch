#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

cd ../../

LOG_FILE="./log/ade20k/fs_resnet50_aspocnetv2_ade20k_seg.log"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_aspocnet_ade20k_seg.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n --max_iters 75000 \
                       --data_dir /msravcshare/v-ansheng/DataSet/ADE20K --model_name asp_ocnetv2 \
                       --backbone deepbase_resnet50_dilated8 --checkpoints_name fs_resnet50_aspocnetv2_ade20k_seg \
                       --pretrained ./pretrained_model/resnet50-imagenet.pth  > $LOG_FILE 2>&1

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_aspocnet_ade20k_seg.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --data_dir /msravcshare/v-ansheng/DataSet/ADE20K --model_name asp_ocnetv2 \
                       --backbone deepbase_resnet50_dilated8 --checkpoints_name fs_resnet50_aspocnetv2_ade20k_seg \
                       --resume_continue y --resume ./checkpoints/ade20k/fs_resnet50_aspocnetv2_ade20k_seg_latest.pth \
                       --pretrained ./pretrained_model/resnet50-imagenet.pth  >> $LOG_FILE 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_aspocnet_ade20k_seg.json --phase debug --gpu 0 > $LOG_FILE 2>&1

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_aspocnet_ade20k_seg.json \
                       --phase test --gpu 0 --resume $2 > $LOG_FILE 2>&1
  cd val/scripts
  ${PYTHON} -u ade20k_evaluator.py --hypes_file ../../hypes/ade20k/fs_aspocnet_ade20k_seg.json \
                                   --gt_dir path-to-gt \
                                   --pred_dir path-to-pred >> $LOG_FILE 2>&1
else
  echo "$1"x" is invalid..."
fi