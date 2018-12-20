#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

cd ../../

DATA_DIR="/msravcshare/v-ansheng/DataSet/CityScape"
BACKBONE="deepbase_resnet101_dilated8"
MODEL_NAME="asp_ocnetv3"
LOSS_TYPE="fs_auxce_loss"
CHECKPOINTS_NAME="fs_aspocnetv3_cityscape_seg"
PRETRAINED_MODEL="./pretrained_model/resnet101-imagenet.pth"
MAX_ITERS=40000

LOG_FILE="./log/cityscape/${CHECKPOINTS_NAME}.log"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_aspocnet_cityscape_seg.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} > ${LOG_FILE} 2>&1

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_aspocnet_cityscape_seg.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} \
                       --resume_continue y --resume ./checkpoints/cityscape/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  >> ${LOG_FILE} 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_aspocnet_cityscape_seg.json --phase debug --gpu 0 > $LOG_FILE 2>&1

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes hypes/cityscape/fs_aspocnet_cityscape_seg.json \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/cityscape/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image  >> ${LOG_FILE} 2>&1
  cd val/scripts
  ${PYTHON} -u cityscape_evaluator.py --hypes_file ../../hypes/cityscape/fs_aspocnet_cityscape_seg.json \
                                      --gt_dir ${DATA_DIR}/val/image \
                                      --pred_dir ../results/cityscape/test_dir/image/label >> ${LOG_FILE} 2>&1
else
  echo "$1"x" is invalid..."
fi