#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

cd ../../

LOG_FILE="./log/cityscape/fs_aspocnetv2_cityscape_ohem_seg.log"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_aspocnet_cityscape_seg.json \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --data_dir /msravcshare/v-ansheng/DataSet/CityScape \
                       --model_name asp_ocnetv2 --checkpoints_name fs_aspocnetv2_cityscape_ohem_seg \
                       --max_epoch 220 --loss_type fs_auxohemce_loss \
                       --pretrained ./pretrained_model/resnet101-imagenet.pth  > $LOG_FILE 2>&1

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_aspocnet_cityscape_seg.json \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --data_dir /msravcshare/v-ansheng/DataSet/CityScape \
                       --model_name asp_ocnetv2 --checkpoints_name fs_aspocnetv2_cityscape_ohem_seg \
                       --max_epoch 220 --loss_type fs_auxohemce_loss \
                       --resume_continue y --resume ./checkpoints/cityscape/fs_aspocnetv2_cityscape_ohem_seg_latest.pth \
                       --pretrained ./pretrained_model/resnet101-imagenet.pth  >> $LOG_FILE 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_aspocnet_cityscape_seg.json --phase debug --gpu 0 > $LOG_FILE 2>&1

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_aspocnet_cityscape_seg.json \
                       --phase test --gpu 0 --resume $2 > $LOG_FILE 2>&1
  cd val/scripts
  ${PYTHON} -u cityscape_evaluator.py --hypes_file ../../hypes/cityscape/fs_aspocnet_cityscape_seg.json \
                                      --gt_dir path-to-gt \
                                      --pred_dir path-to-pred >> $LOG_FILE 2>&1
else
  echo "$1"x" is invalid..."
fi