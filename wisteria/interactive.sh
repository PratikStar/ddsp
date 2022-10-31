#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N ddsprun
#PJM -j

source /work/01/gk77/k77021/.bashrc
echo "loaded source"
export HOME=/work/01/gk77/k77021

ddsp_run \
  --mode=train \
  --save_dir=/work/gk77/k77021/ddsp/training \
  --gin_file=/work/gk77/k77021/repos/ddsp/ddsp/training/gin/models/ae.gin \
  --gin_file=/work/gk77/k77021/repos/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/work/gk77/k77021/repos/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/work/gk77/k77021/data/ddsp/monophonic/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr
