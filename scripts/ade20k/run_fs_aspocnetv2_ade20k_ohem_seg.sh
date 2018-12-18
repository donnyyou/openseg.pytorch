#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

cd ../../

CHECKPOINTS_NAME="fs_aspocnetv2_ade20k_ohem_seg"

LOG_FILE="./log/ade20k/${CHECKPOINTS_NAME}.log"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_aspocnet_ade20k_seg.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --data_dir /msravcshare/v-ansheng/DataSet/ADE20K \
                       --model_name asp_ocnetv2 --checkpoints_name ${CHECKPOINTS_NAME} \
                       --max_iters 300000 --loss_type fs_auxohemce_loss \
                       --pretrained ./pretrained_model/resnet101-imagenet.pth  > $LOG_FILE 2>&1

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_aspocnet_ade20k_seg.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --data_dir /msravcshare/v-ansheng/DataSet/ADE20K \
                       --model_name asp_ocnetv2 --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume_continue y --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --max_iters 300000 --loss_type fs_auxohemce_loss \
                       --pretrained ./pretrained_model/resnet101-imagenet.pth  >> $LOG_FILE 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_aspocnet_ade20k_seg.json --phase debug --gpu 0 > $LOG_FILE 2>&1

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_aspocnet_ade20k_seg.json \
                       --phase test --gpu 0 \
                       --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth > $LOG_FILE 2>&1
  cd val/scripts
  ${PYTHON} -u ade20k_evaluator.py --hypes_file ../../hypes/ade20k/fs_aspocnet_ade20k_seg.json \
                                   --gt_dir path-to-gt \
                                   --pred_dir path-to-pred >> $LOG_FILE 2>&1
else
  echo "$1"x" is invalid..."
fi