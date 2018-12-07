#!/usr/bin/env bash


cd ../../


if [ "$1"x == "train"x ]; then
  python main.py --hypes hypes/cityscape/fs_ocnet_cityscape_seg.json \
                 --phase train --gpu 3 --train_batch_size 1 --val_batch_size 1 --display_iter 1 --gathered n --loss_balance y --test_interval 10 

elif [ "$1"x == "debug"x ]; then
  python main.py --hypes hypes/cityscape/fs_ocnet_cityscape_seg.json --phase debug --gpu 0

elif [ "$1"x == "test"x ]; then
  python main.py --hypes hypes/cityscape/fs_ocnet_cityscape_seg.json --phase test --gpu 0 --resume $2
  cd val/scripts
  python cityscape_evaluator.py --hypes_file ../../../hypes/cityscape/fs_ocnet_cityscape_seg.json \
                             --gt_dir path-to-gt \
                             --pred_dir path-to-pred
else
  echo "$1"x" is invalid..."
fi
