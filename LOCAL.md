Anaconda doesn't work, install miniconda as per the doc below.
```shell
cd /Users/pratik/repos/ddsp
rsync -av "/Users/pratik/repos/ddsp" w:/work/gk77/k77021/repos


ddsp_run \
  --mode=train \
  --save_dir=/tmp/$USER-ddsp-0 \
  --gin_file=papers/iclr2020/nsynth_ae.gin \
  --gin_param="batch_size=16" \
  --alsologtostderr
```

### repo. local --> wisteria
```shell
cd ~/repos/ddsp
watch -d -n5 "rsync -av --exclude-from=\".rsyncignore_upload\" \"/Users/pratik/repos/ddsp\" w:/work/gk77/k77021/repos"
```



# Local. Install tensorflow on mac m1
https://developer.apple.com/metal/tensorflow-plugin/

Note: upgrade numpy if tensorflow throws numpy error!
```shell
# local
ddsp_prepare_tfrecord \
--input_audio_filepatterns='/Users/pratik/data/timbre/clips/*wav' \
--output_tfrecord_path=/Users/pratik/data/ddsp/train.tfrecord \
--num_shards=10 \
--alsologtostderr
```
ddsp_prepare_tfrecord \
--input_audio_filepatterns='/root/clips/*wav' \
--output_tfrecord_path=/root/ddsp/train.tfrecord \
--num_shards=10 \
--alsologtostderr


ddsp_prepare_tfrecord \
--input_audio_filepatterns='/home/pratik/clips/*wav' \
--output_tfrecord_path=/home/pratik/ddsp/train.tfrecord \
--num_shards=10 \
--alsologtostderr

# wisteria

```shell
# wisteria
ddsp_prepare_tfrecord \
--input_audio_filepatterns='/work/gk77/k77021/data/timbre/monophonic/*wav' \
--output_tfrecord_path=/work/gk77/k77021/data/ddsp/monophonic/train.tfrecord \
--num_shards=10 \
--sample_rate=44100 \
--alsologtostderr


ssh-agent bash -c 'ssh-add /work/gk77/k77021/.ssh/id_rsa; git clone git@github.com:PratikStar/ddsp.git'


```
## verifying tfrecord on wisteria
import tensorflow as tf
from tfrecord_lite import decode_example
raw_dataset = tf.data.TFRecordDataset("train.tfrecord-00000-of-00010")
raw_record = raw_dataset.take(1)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
    
    
it = tf.io.tf_record_iterator("train.tfrecord-00000-of-00010")

## misc
==========

## Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
--> ignored. no need to use tensorrt










# aws
https://drive.google.com/file/d/1kLigmxq8Lr5MEQqLnFPCRHjbiJbXosiJ/view?usp=sharing

gdown https://drive.google.com/uc?id=1kLigmxq8Lr5MEQqLnFPCRHjbiJbXosiJ





```shell
import ddsp.training, os

TRAIN_TFRECORD = '/work/gk77/k77021/data/ddsp/train.tfrecord'
TRAIN_TFRECORD_FILEPATTERN = TRAIN_TFRECORD + '*'
SAVE_DIR = '/work/gk77/k77021/data/ddsp/ddsp-solo-instrument'

data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
dataset = data_provider.get_dataset(shuffle=False)
PICKLE_FILE_PATH = os.path.join(SAVE_DIR, 'dataset_statistics.pkl')

ds_stats = ddsp.training.postprocessing.compute_dataset_statistics(data_provider=data_provider, batch_size=1, power_frame_size=256)

  # Save.
  if file_path is not None:
    with tf.io.gfile.GFile(file_path, 'wb') as f:
      pickle.dump(ds_stats, f)
    print(f'Done! Saved dataset statistics to: {file_path}')
_ = colab_utils.save_dataset_statistics(data_provider, PICKLE_FILE_PATH, batch_size=1)
```

ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --gin_file=models/solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='/work/gk77/k77021/data/ddsp/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --gin_param="train_util.train.num_steps=30000" \
  --gin_param="train_util.train.steps_per_save=300" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=10"


# ist

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pratik/miniconda3/pkgs/cudatoolkit-11.2.2-he111cf0_8/lib


sudo apt-get install --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
        libnvinfer-dev=6.0.1-1+cuda10.1 \
        libnvinfer-plugin6=6.0.1-1+cuda10.1








# GCP

```shell
sudo su


apt-get install git
apt-get install wget
# install miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3<tab>
source /root/.bashrc

sudo apt-get install libsndfile-dev
sudo apt install ffmpeg
pip install --upgrade pip
pip install tensorflow==2.11.0rc0
pip install apache-beam wandb
pip install --upgrade ddsp
mkdir ~/buckets ~/logs

ssh-keygen
<add key to github>
git clone git@github.com:PratikStar/ddsp.git
cd ddsp
pip install .

gsutil -u ddsp-366504 cp -r gs://pratik-ddsp-data ~/buckets
gsutil -u ddsp2-374016 cp -r gs://pratik-ddsp2-data ~/buckets
#gsutil -> https://hartwigmedical.github.io/documentation/accessing-hartwig-data-through-gcp.html#accessing-data THIS works!!

gsutil -u nws-oc cp -r gs://pratik-timbre-data ~/buckets

gcloud auth login
gcloud config set project ddsp2-374016
gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east1-c instance-gpu
wandb login

# install docker. IFF not already installed

```

## data prep

```shell
ddsp_prepare_tfrecord \
--input_audio_filepatterns='/root/buckets/pratik-ddsp-data/monophonic/*wav' \
--output_tfrecord_path=/root/tfrecord/train.tfrecord \
--chunk_secs=0.0 \
--num_shards=10 \
--alsologtostderr


python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-ddsp-data/monophonic/*wav' \
--output_tfrecord_path=/root/tfrecord/train.tfrecord \
--chunk_secs=0.0 \
--num_shards=10 \
--alsologtostderr >> ~/logs/data_prep_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# new GCP account. f0 from di stuff
DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-ddsp2-data/monophonic-passagebased/*wav' \
--output_tfrecord_path=/root/tfrecord/train.tfrecord \
--chunk_secs=0.0 \
--num_shards=10 \
--f0_from_di=True \
--alsologtostderr >> ~/logs/data_prep_16k_f0_from_di_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
### just download tfrecords from wisteria to test ddsp_run
rsync -av w:/work/gk77/k77021/data/ddsp/monophonic "/Users/pratik/data/ddsp_tfrecords"


## ddsp_run locally
ddsp_run \
  --mode=train \
  --save_dir=/root/save_dir \
  --gin_file=/root/ddsp/ddsp/training/gin/models/solo_instrument.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr >> ~/logs/data_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  

ddsp_run \
  --mode=train \
  --save_dir=/root/save_dir_ae \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr >> ~/logs/data_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &


ddsp_run \
  --mode=train \
  --save_dir=/root/save_dir_ae_rnn_mean \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_mean.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr >> ~/logs/data_run_rnn_mean_$(date +%Y%m%d_%H%M%S).log 2>&1 &



ddsp_run \
  --mode=train \
  --save_dir=/root/save_dir_tmp \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr 



gcloud compute ssh instance-1 --ssh-flag="-ServerAliveInterval=30" --zone us-east1-b --command "sudo expand-root.sh /dev/sda 1 ext4"

wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb

### tensorboard
[comment]: <> (https://www.montefischer.com/2020/02/20/tensorboard-with-gcp.html)

1. start tensorboard
tensorboard --logdir save_dir/ &

2. port forwarding
gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east1-c instance-gpu -- -NfL 6006:localhost:6006

go to localhost:6006
for reference. for ssh
gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east1-c instance-gpu


### Colab x GCE
dont follow this article though
[comment]: <> (https://medium.com/@senthilnathangautham/colab-gcp-compute-how-to-link-them-together-98747e8d940e)

```shell

# SSH to the instance and sudo su
gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east1-c instance-gpu
sudo su
cd

# Start jupyter notebook server
jupyter notebook \
--no-browser \
--NotebookApp.answer_yes=True \
--log-level=DEBUG \
--NotebookApp.allow_origin='https://colab.research.google.com' \
--port=8888 \
--ServerApp.port_retries=0 \
--ip=0.0.0.0 \
--allow-root >> ~/tmp/colab_$(date +%Y%m%d_%H%M%S).log 2>&1 &

gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east1-c instance-gpu -- -NfvL 8888:instance-gpu:8888

# Copy the URL from the jupyter notebook command 
# Open Colan and Connet to local runtime and use the copied URL from the first step.

```

### Upload to wandb
```shell
WANDB_PROJECT=ddsp wandb artifact put -t model save_dir_ae/ckpt-45000.index
WANDB_PROJECT=ddsp wandb artifact put -t model save_dir_ae/ckpt-45000.data-00000-of-00001
WANDB_PROJECT=ddsp wandb artifact put save_dir_ae/operative_config-0.gin


WANDB_PROJECT=ddsp wandb artifact put -n ae-rnn-mean-z-45600 -t model save_dir_ae_rnn_mean/ckpt-45600.index
WANDB_PROJECT=ddsp wandb artifact put -n ae-rnn-mean-z-45600_data -t model save_dir_ae_rnn_mean/ckpt-45600.data-00000-of-00001
WANDB_PROJECT=ddsp wandb artifact put -n ae-rnn-mean-z-45600_gin save_dir_ae_rnn_mean/operative_config-0.gin

WANDB_PROJECT=ddsp wandb artifact put -n ae-rnn-last-z-99900 -t model save_dir_ae_rnn_last/ckpt-99900.index
WANDB_PROJECT=ddsp wandb artifact put -n ae-rnn-last-z-99900_data -t model save_dir_ae_rnn_last/ckpt-99900.data-00000-of-00001
WANDB_PROJECT=ddsp wandb artifact put -n ae-rnn-last-z-99900_gin save_dir_ae_rnn_last/operative_config-0.gin

```

#### Note
```shell
======================================
Welcome to the Google Deep Learning VM
======================================

Version: common-cu113.m98
Based on: Debian GNU/Linux 10 (buster) (GNU/Linux 4.19.0-21-cloud-amd64 x86_64\n)

Resources:
 * Google Deep Learning Platform StackOverflow: https://stackoverflow.com/questions/tagged/google-dl-platform
 * Google Cloud Documentation: https://cloud.google.com/deep-learning-vm
 * Google Group: https://groups.google.com/forum/#!forum/google-dl-platform

To reinstall Nvidia driver (if needed) run:
sudo /opt/deeplearning/install-driver.sh
Linux instance-gpu-2 4.19.0-21-cloud-amd64 #1 SMP Debian 4.19.249-2 (2022-06-30) x86_64

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.

This VM requires Nvidia drivers to function correctly.   Installation takes ~1 minute.
```

#### NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.




# install package from source
conda create -n test_env python=3.9.12
conda activate test_env

cd ~/ddsp
pip install .

# using many to one rnn layer. No mean
ddsp_run \
  --mode=train \
  --save_dir=/root/save_dir_ae_rnn_last \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr >> ~/logs/data_run_rnn_last_$(date +%Y%m%d_%H%M%S).log 2>&1 &

tensorboard --logdir ~/save_dir_ae_rnn_last/ &

2. port forwarding
gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east1-c instance-gpu -- -NfL 6006:localhost:6006

### wanbd test
DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=rnn_last \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_multiloss.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr

# LSTM
DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=rnn_lstm_last \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr >> ~/logs/ddsp_run_rnn_lstm_last_$(date +%Y%m%d_%H%M%S).log 2>&1 &


# for 44.1khz crepe
## sr=44100 and frame rate= 210
DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-ddsp-data/monophonic/*wav' \
--output_tfrecord_path=/root/tfrecord_441sr_210fr/train.tfrecord \
--chunk_secs=0.0 \
--num_shards=10 \
--frame_rate=210 \
--sample_rate=44100 \
--alsologtostderr

DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=rnn_last_441_210 \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord_441sr_210fr/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr \
  --gin_param="TFRecordProvider.sample_rate=44100" \
  --gin_param="Harmonic.sample_rate=44100" \
  --gin_param="FilteredNoise.n_samples=176400" \
  --gin_param="Harmonic.n_samples=176400" \
  --gin_param="Reverb.reverb_length=176400" \
  --gin_param='F0LoudnessPreprocessor.time_steps=840' \
  --gin_param='F0LoudnessPreprocessor.frame_rate=210' \
  --gin_param='F0LoudnessPreprocessor.sample_rate=44100' \
  --gin_param="TFRecordProvider.frame_rate=210" >> ~/logs/ddsp_run_gru_last_441_210_$(date +%Y%m%d_%H%M%S).log 2>&1 &

## sr=44100 and frame rate=1000
DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-ddsp-data/monophonic/*wav' \
--output_tfrecord_path=/root/tfrecord_441sr_1000fr/train.tfrecord \
--chunk_secs=0.0 \
--num_shards=10 \
--frame_rate=1000 \
--sample_rate=44100 \
--alsologtostderr  >> ~/logs/ddsp_data_441_1000_$(date +%Y%m%d_%H%M%S).log 2>&1 &


DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=rnn_last_441_1000 \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord_441sr_1000fr/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr \
  --gin_param="TFRecordProvider.sample_rate=44100" \
  --gin_param="Harmonic.sample_rate=44100" \
  --gin_param="FilteredNoise.n_samples=176400" \
  --gin_param="Harmonic.n_samples=176400" \
  --gin_param="Reverb.reverb_length=176400" \
  --gin_param='F0LoudnessPreprocessor.time_steps=4000' \
  --gin_param='F0LoudnessPreprocessor.frame_rate=1000' \
  --gin_param='F0LoudnessPreprocessor.sample_rate=44100' \
  --gin_param="TFRecordProvider.frame_rate=1000"


## sr=44100 and frame rate=252

DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-ddsp-data/monophonic/*wav' \
--output_tfrecord_path=/root/tfrecord_441sr_252fr/train.tfrecord \
--chunk_secs=0.0 \
--num_shards=10 \
--frame_rate=252 \
--sample_rate=44100 \
--alsologtostderr  >> ~/logs/ddsp_data_441_1000_$(date +%Y%m%d_%H%M%S).log 2>&1 &

DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=rnn_last_441_252 \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord_441sr_252fr/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr \
  --gin_param="TFRecordProvider.sample_rate=44100" \
  --gin_param="Harmonic.sample_rate=44100" \
  --gin_param="FilteredNoise.n_samples=176400" \
  --gin_param="Harmonic.n_samples=176400" \
  --gin_param="Reverb.reverb_length=176400" \
  --gin_param='F0LoudnessPreprocessor.time_steps=1008' \
  --gin_param='F0LoudnessPreprocessor.frame_rate=252' \
  --gin_param='F0LoudnessPreprocessor.sample_rate=44100' \
  --gin_param="TFRecordProvider.frame_rate=252" >> ~/logs/ddsp_run_gru_last_441_252_$(date +%Y%m%d_%H%M%S).log 2>&1 &


## sr=32000 and frame rate=250

DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-ddsp-data/monophonic/*wav' \
--output_tfrecord_path=/root/tfrecord_320sr_252fr/train.tfrecord \
--chunk_secs=0.0 \
--num_shards=10 \
--frame_rate=250 \
--sample_rate=32000 \
--alsologtostderr  >> ~/logs/ddsp_data_320_250_$(date +%Y%m%d_%H%M%S).log 2>&1 &

DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=rnn_last_320_250 \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord_320sr_250fr/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr \
  --gin_param="TFRecordProvider.sample_rate=32000" \
  --gin_param="Harmonic.sample_rate=32000" \
  --gin_param="FilteredNoise.n_samples=96000" \
  --gin_param="Harmonic.n_samples=96000" \
  --gin_param="Reverb.reverb_length=96000" \
  --gin_param='F0LoudnessPreprocessor.time_steps=1000' \
  --gin_param='F0LoudnessPreprocessor.frame_rate=250' \
  --gin_param='F0LoudnessPreprocessor.sample_rate=32000' \
  --gin_param="TFRecordProvider.frame_rate=250" >> ~/logs/ddsp_run_gru_last_320_250_$(date +%Y%m%d_%H%M%S).log 2>&1 &


## sr=44100 and frame rate=700. batch size =8

DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-ddsp-data/monophonic/09*wav' \
--output_tfrecord_path=/root/tfrecord_441sr_700fr_09/train.tfrecord \
--chunk_secs=0.0 \
--frame_rate=700 \
--sample_rate=44100 \
--alsologtostderr  >> ~/logs/ddsp_data_441_700_09_$(date +%Y%m%d_%H%M%S).log 2>&1 &

DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=rnn_last_441_700_again \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord_441sr_700fr/train.tfrecord*'" \
  --gin_param="batch_size=8" \
  --alsologtostderr \
  --gin_param="TFRecordProvider.sample_rate=44100" \
  --gin_param="Harmonic.sample_rate=44100" \
  --gin_param="FilteredNoise.n_samples=176400" \
  --gin_param="Harmonic.n_samples=176400" \
  --gin_param="Reverb.reverb_length=132300" \
  --gin_param='F0LoudnessPreprocessor.time_steps=2800' \
  --gin_param='F0LoudnessPreprocessor.frame_rate=700' \
  --gin_param='F0LoudnessPreprocessor.sample_rate=44100' \
  --gin_param="TFRecordProvider.frame_rate=700" >> ~/logs/ddsp_run_gru_last_441_700_$(date +%Y%m%d_%H%M%S).log 2>&1 &


gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-west1-b instance-gpu-1
gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone asia-east1-a instance-gpu-2

# instance-gpu-1
Trained on 252 fr and 44.1khz sr


# one timbre


DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/test-audio/*' \
--output_tfrecord_path=/root/tfrecord_only-test_160_250-8-2/train.tfrecord \
--chunk_secs=0.0 \
--frame_rate=250 \
--num_shards=8 \
--example_secs=2 \
--sample_rate=16000 \
--alsologtostderr >> ~/logs/ddsp_data_test-audio_160_250_$(date +%Y%m%d_%H%M%S).log 2>&1 &


DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=test-audio_160_250-8-2 \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord_only-test_160_250-8-2/train.tfrecord*'" \
  --gin_param="batch_size=8" \
  --alsologtostderr \
  --gin_param="TFRecordProvider.sample_rate=16000" \
  --gin_param="Harmonic.sample_rate=16000" \
  --gin_param="FilteredNoise.n_samples=32000" \
  --gin_param="Harmonic.n_samples=32000" \
  --gin_param='F0LoudnessPreprocessor.time_steps=500' \
  --gin_param='F0LoudnessPreprocessor.frame_rate=250' \
  --gin_param='F0LoudnessPreprocessor.sample_rate=16000' \
  --gin_param="TFRecordProvider.frame_rate=250" >> ~/logs/ddsp_run_test-audio_160_250_$(date +%Y%m%d_%H%M%S).log 2>&1 &

DEBUG=1 ddsp_run \
  --mode=train \
  --run_name=ae_rnn_last_again \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr  >> ~/logs/ddsp_run-ae_rnn_last_again$(date +%Y%m%d_%H%M%S).log 2>&1 &


ddsp_run \
 --mode=train \
 --run_name=ae_rnn_last_again \
 --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
 --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
 --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
 --gin_param="TFRecordProvider.file_pattern='/root/tfrecord_test_sr32k_fr250_shards10/train.tfrecord*'" \
 --gin_param="batch_size=16" \
 --alsologtostderr \
  --gin_param="TFRecordProvider.sample_rate=32000" \
  --gin_param="Harmonic.sample_rate=32000" \
  --gin_param="FilteredNoise.n_samples=96000" \
  --gin_param="Harmonic.n_samples=128000" \
  --gin_param='F0LoudnessPreprocessor.time_steps=1000' \
  --gin_param='F0LoudnessPreprocessor.frame_rate=250' \
  --gin_param='F0LoudnessPreprocessor.sample_rate=32000' \
  --gin_param="TFRecordProvider.frame_rate=250"

[//]: # (2022/12 december while writing thesis)

DEBUG=1 ddsp_run \
--mode=train \
--run_name=ae_rnn_last_test_sr441k_fr252 \
--gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
--gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
--gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
--gin_param="TFRecordProvider.file_pattern='/root/tfrecord_test_sr441k_fr252_shards10/train.tfrecord*'" \
--gin_param="batch_size=8" \
--gin_param="train.steps_per_save=2" \
--alsologtostderr \
--gin_param="TFRecordProvider.sample_rate=44100" \
--gin_param="Harmonic.sample_rate=44100" \
--gin_param="FilteredNoise.n_samples=176400" \
--gin_param="Harmonic.n_samples=176400" \
--gin_param='F0LoudnessPreprocessor.time_steps=1008' \
--gin_param='F0LoudnessPreprocessor.frame_rate=252' \
--gin_param='F0LoudnessPreprocessor.sample_rate=44100' \
--gin_param="TFRecordProvider.frame_rate=252"

ddsp_run \
--mode=train \
--run_name=vanilla_ae_13B_sr44k_fr250_loss4096 \
--gin_file=/root/ddsp/ddsp/training/gin/models/ae.gin \
--gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
--gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
--gin_param="TFRecordProvider.file_pattern='/root/tfrecord_13B_sr44k_fr250/train.tfrecord*'" \
--gin_param="batch_size=8" \
--alsologtostderr \
--gin_param="TFRecordProvider.sample_rate=44000" \
--gin_param="Harmonic.sample_rate=44000" \
--gin_param="FilteredNoise.n_samples=176000" \
--gin_param="Harmonic.n_samples=176000" \
--gin_param='F0LoudnessPreprocessor.time_steps=1000' \
--gin_param='F0LoudnessPreprocessor.frame_rate=250' \
--gin_param='F0LoudnessPreprocessor.sample_rate=44000' \  --gin_param='oscillator_bank.use_angular_cumsum=True' \
--gin_param="TFRecordProvider.frame_rate=250" >> ~/logs/ddsp_run-vanilla_ae_13B_sr44k_fr250_loss4096_$(date +%Y%m%d_%H%M%S).log 2>&1 &





# new GCP account. f0 from di stuff
DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-ddsp2-data/monophonic-passagebased/*wav' \
--output_tfrecord_path=/root/tfrecord/train.tfrecord \
--chunk_secs=0.0 \
--num_shards=10 \
--f0_from_di=True \
--alsologtostderr >> ~/logs/data_prep_16k_f0_from_di_$(date +%Y%m%d_%H%M%S).log 2>&1 &

ddsp_prepare_tfrecord \
--input_audio_filepatterns='/work/gk77/k77021/data/timbre/monophonic-4secchunks/*wav' \
--output_tfrecord_path=/work/gk77/k77021/data/ddsp/monophonic/train.tfrecord \
--num_shards=10 \
--sample_rate=44100 \
--alsologtostderr



DEBUG=1 
ddsp_run \
--mode=train \
--run_name=ae-f0_di \
--gin_file=/root/ddsp/ddsp/training/gin/models/ae.gin \
--gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
--gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
--gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
--gin_param="batch_size=16" \
--gin_param="train.steps_per_save=1000" \
--alsologtostderr >> ~/logs/ddsp_run-ae-f0_di_$(date +%Y%m%d_%H%M%S).log 2>&1 &


DEBUG=1 
ddsp_run \
--mode=train \
--run_name=ae_last-f0_di \
--gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
--gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
--gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
--gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
--gin_param="batch_size=16" \
--gin_param="train.steps_per_save=1000" \
--alsologtostderr >> ~/logs/ddsp_run-ae_last-f0_di_$(date +%Y%m%d_%H%M%S).log 2>&1 &


DEBUG=1 ddsp_run \
--mode=train \
--run_name=ae_mean-f0_di \
--gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_mean.gin \
--gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
--gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
--gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
--gin_param="batch_size=16" \
--gin_param="train.steps_per_save=1000" \
--alsologtostderr >> ~/logs/ddsp_run-ae_last-f0_di_$(date +%Y%m%d_%H%M%S).log 2>&1 &



`
# 1. Rsync repo
cd /Users/pratik/repos/ddsp
watch -d -n5 "rsync -av --exclude-from=\".rsyncignore_upload\" \"/Users/pratik/repos/ddsp\" w:/work/gk77/k77021/repos"

nohup watch -d -n5 rsync -av --exclude-from=".rsyncignore_upload" "/Users/pratik/repos/ddsp" w:/work/gk77/k77021/repos 0<&- &> /dev/null &

# 2. Rsync data
cd /Users/pratik/data/A_sharp_3
rsync -avz "/Users/pratik/data/A_sharp_3" w:/work/gk77/k77021/data
rsync -avz "/Users/pratik/data/single_note_distorted" w:/work/gk77/k77021/data
rsync -avz "/Users/pratik/data/di_1_one_clip" w:/work/gk77/k77021/data

# from wisteria
rsync -av w:/work/gk77/k77021/data/A_sharp_3 "/Users/pratik/Downloads"


gsutil -u nws-gb cp -r gs://pratik-timbre-data-gb ~/buckets


DEBUG=1 python /root/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
--input_audio_filepatterns='/root/buckets/pratik-timbre-data-gb/monophonic-4secchunks/*wav' \
--output_tfrecord_path=/root/data/ddsp/tfrecord_final_sr16k/train.tfrecord \
--chunk_secs=0.0 \
--frame_rate=250 \
--num_shards=10 \
--example_secs=4 \
--f0_from_di=True \
--sample_rate=16000 \
--alsologtostderr >> ~/logs/ddsp_data-final_sr16k_$(date +%Y%m%d_%H%M%S).log 2>&1 &


DEBUG=1 ddsp_run \
--mode=train \
--run_name=ismir_ae_mean \
--gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_mean.gin \
--gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
--gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
--gin_param="train.steps_per_save=1000" \
--gin_param="TFRecordProvider.file_pattern='/root/data/ddsp/tfrecord_final_sr16k/train.tfrecord*'" \
--gin_param="batch_size=8" \
--alsologtostderr >> ~/logs/ddsp_run-ismir_ae_mean_$(date +%Y%m%d_%H%M%S).log 2>&1 &
