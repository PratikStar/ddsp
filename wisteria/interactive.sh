#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N ddsp_data_all1
#PJM -j

source /work/01/gk77/k77021/.bashrc
echo "loaded source"
export HOME=/work/01/gk77/k77021
pwd
#pip install .

echo "====== GPU info ======"
nvidia-smi
echo "======================"

#DEBUG=1 python /work/gk77/k77021/repos/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
#--input_audio_filepatterns='/work/gk77/k77021/data/timbre/monophonic-4secchunks/*1.wav' \
#--output_tfrecord_path=/work/gk77/k77021/data/ddsp/tfrecord-all_passage_1-sr44k/train.tfrecord \
#--chunk_secs=0.0 \
#--frame_rate=250 \
#--num_shards=10 \
#--example_secs=4 \
#--sample_rate=44000 \
#--f0_from_di=True \
#--alsologtostderr
#

XLA_FLAGS=--xla_gpu_cuda_data_dir=/work/opt/local/x86_64/cores/cuda* ddsp_run \
--mode=train \
--run_name=ae_A_sharp_3_sr44k \
--gin_file=/work/gk77/k77021/repos/ddsp/ddsp/training/gin/models/ae.gin \
--gin_file=/work/gk77/k77021/repos/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
--gin_file=/work/gk77/k77021/repos/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
--gin_param="TFRecordProvider.file_pattern='/work/gk77/k77021/data/ddsp/tfrecord-A_sharp_3-sr44k/train.tfrecord*'" \
--gin_param="batch_size=8" \
--save_dir=/work/gk77/k77021/ddsp/save_dir_ \
--restore_dir=/work/gk77/k77021/ddsp/save_dir_ \
--steps_per_summary=2 \
--steps_per_save=1 \
--alsologtostderr \
--gin_param="TFRecordProvider.sample_rate=44000" \
--gin_param="Harmonic.sample_rate=44000" \
--gin_param="FilteredNoise.n_samples=176000" \
--gin_param="Harmonic.n_samples=176000" \
--gin_param='F0LoudnessPreprocessor.time_steps=1000' \
--gin_param='F0LoudnessPreprocessor.frame_rate=250' \
--gin_param='F0LoudnessPreprocessor.sample_rate=44000' \
--gin_param='oscillator_bank.use_angular_cumsum=True' \
--gin_param="TFRecordProvider.frame_rate=250"