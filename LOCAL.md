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

git clone <ddsp>

apt-get install git
apt-get install wget
# install miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3.....
source /root/.bashrc

sudo apt-get install libsndfile-dev
pip install tensorflow==2.11.0rc0
pip install apache-beam
pip install --upgrade pip
pip install --upgrade ddsp

gcsfuse --> https://medium.com/google-cloud/scheduled-mirror-sync-sftp-to-gcs-b167d0eb487a
gsutil -> https://hartwigmedical.github.io/documentation/accessing-hartwig-data-through-gcp.html#accessing-data
mkdir ~/bucket ~/bucket-tfrecord
gcsfuse pratik-ddsp-data ~/bucket
gcsfuse pratik-ddsp-tfrecord ~/bucket-tfrecord
OR
gsutil -u ddsp-366504 cp -r gs://pratik-ddsp-data ~/buckets


sudo apt install ffmpeg

# install docker. IFF not already installed

```

## data prep


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

## ddsp_run on AI Platform

### Image build and push
cd /root/ddsp/ddsp/training/docker

export PROJECT_ID=ddsp-366504
export IMAGE_REPO_NAME=ddsp_train
export IMAGE_TAG=ai_platform
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

[//]: #(https://stackoverflow.com/questions/55446787/permission-issues-while-docker-push)
[comment]: <> (https://docs.docker.com/engine/install/debian/#install-using-the-repository)

docker build -f Dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI

### Submit job

export SAVE_DIR=gs://pratik-ddsp-models
export FILE_PATTERN=gs://pratik-ddsp-tfrecord/train.tfrecord*
export REGION=us-east1

cd /root/ddsp/ddsp/training/docker

export JOB_NAME=ddsp_container_job_$(date +%Y%m%d_%H%M%S)
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --config config_single_vm.yaml \
  --master-image-uri $IMAGE_URI \
  -- \
  --save_dir=$SAVE_DIR \
  --file_pattern=$FILE_PATTERN \
  --batch_size=16 \
  --learning_rate=0.0001 \
  --num_steps=40000 \
  --early_stop_loss_value=5.0


gcloud ai-platform jobs list

[comment]: <> (https://console.cloud.google.com/ai-platform/jobs?authuser=1&project=ddsp-366504)


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






```shell
I1031 09:49:25.585731 139929154720000 ddsp_run.py:179] Restore Dir: /root/save_dir_tmp
I1031 09:49:25.586658 139929154720000 ddsp_run.py:180] Save Dir: /root/save_dir_tmp
I1031 09:49:25.587098 139929154720000 resource_reader.py:50] system_path_file_exists:optimization/base.gin
E1031 09:49:25.587472 139929154720000 resource_reader.py:55] Path not found: optimization/base.gin
I1031 09:49:25.590388 139929154720000 resource_reader.py:50] system_path_file_exists:eval/basic.gin
E1031 09:49:25.590616 139929154720000 resource_reader.py:55] Path not found: eval/basic.gin
I1031 09:49:25.593094 139929154720000 ddsp_run.py:152] Operative config not found in /root/save_dir_tmp
I1031 09:49:25.597534 139929154720000 resource_reader.py:50] system_path_file_exists:datasets/base.gin
E1031 09:49:25.597746 139929154720000 resource_reader.py:55] Path not found: datasets/base.gin
I1031 09:49:25.605808 139929154720000 ddsp_run.py:184] Operative Gin Config:
import ddsp
import ddsp.training as ddsp2

# Macros:
# ==============================================================================
batch_size = 16
evaluators = [@BasicEvaluator, @F0LdEvaluator]
learning_rate = 0.0003

# Parameters for processors.Add:
# ==============================================================================
processors.Add.name = 'add'

# Parameters for Autoencoder:
# ==============================================================================
Autoencoder.decoder = @decoders.RnnFcDecoder()
Autoencoder.encoder = @encoders.MfccTimeDistributedRnnEncoder()
Autoencoder.losses = [@losses.SpectralLoss()]
Autoencoder.preprocessor = @preprocessing.F0LoudnessPreprocessor()
Autoencoder.processor_group = @processors.ProcessorGroup()

# Parameters for evaluate:
# ==============================================================================
evaluate.batch_size = 32
evaluate.data_provider = @data.TFRecordProvider()
evaluate.evaluator_classes = %evaluators
evaluate.num_batches = 5

# Parameters for F0LoudnessPreprocessor:
# ==============================================================================
F0LoudnessPreprocessor.time_steps = 1000

# Parameters for FilteredNoise:
# ==============================================================================
FilteredNoise.n_samples = 64000
FilteredNoise.name = 'filtered_noise'
FilteredNoise.scale_fn = @core.exp_sigmoid
FilteredNoise.window_size = 0

# Parameters for get_model:
# ==============================================================================
get_model.model = @models.Autoencoder()

# Parameters for Harmonic:
# ==============================================================================
Harmonic.n_samples = 64000
Harmonic.name = 'harmonic'
Harmonic.normalize_below_nyquist = True
Harmonic.sample_rate = 16000
Harmonic.scale_fn = @core.exp_sigmoid

# Parameters for MfccTimeDistributedRnnEncoder:
# ==============================================================================
MfccTimeDistributedRnnEncoder.rnn_channels = 512
MfccTimeDistributedRnnEncoder.rnn_type = 'gru'
MfccTimeDistributedRnnEncoder.z_dims = 16
MfccTimeDistributedRnnEncoder.z_time_steps = 125

# Parameters for ProcessorGroup:
# ==============================================================================
ProcessorGroup.dag = \
    [(@synths.Harmonic(), ['amps', 'harmonic_distribution', 'f0_hz']),
     (@synths.FilteredNoise(), ['noise_magnitudes']),
     (@processors.Add(), ['filtered_noise/signal', 'harmonic/signal'])]

# Parameters for RnnFcDecoder:
# ==============================================================================
RnnFcDecoder.ch = 512
RnnFcDecoder.input_keys = ('ld_scaled', 'f0_scaled', 'z')
RnnFcDecoder.layers_per_stack = 3
RnnFcDecoder.output_splits = \
    (('amps', 1), ('harmonic_distribution', 100), ('noise_magnitudes', 65))
RnnFcDecoder.rnn_channels = 512
RnnFcDecoder.rnn_type = 'gru'

# Parameters for sample:
# ==============================================================================
sample.batch_size = 16
sample.ckpt_delay_secs = 300
sample.data_provider = @data.TFRecordProvider()
sample.evaluator_classes = %evaluators
sample.num_batches = 1

# Parameters for SpectralLoss:
# ==============================================================================
SpectralLoss.logmag_weight = 1.0
SpectralLoss.loss_type = 'L1'
SpectralLoss.mag_weight = 1.0

# Parameters for TFRecordProvider:
# ==============================================================================
TFRecordProvider.file_pattern = '/root/tfrecord/train.tfrecord*'

# Parameters for train:
# ==============================================================================
train.batch_size = %batch_size
train.data_provider = @data.TFRecordProvider()
train.num_steps = 1000000
train.steps_per_save = 300
train.steps_per_summary = 300

# Parameters for Trainer:
# ==============================================================================
Trainer.checkpoints_to_keep = 100
Trainer.grad_clip_norm = 3.0
Trainer.learning_rate = %learning_rate
Trainer.lr_decay_rate = 0.98
Trainer.lr_decay_steps = 10000
```

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
ddsp_run \
  --mode=train \
  --run_name=rnn_last \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr

# LSTM
ddsp_run \
  --mode=train \
  --run_name=rnn_lstm_last \
  --gin_file=/root/ddsp/ddsp/training/gin/models/ae_mfccRnnEncoder_last.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr >> ~/logs/ddsp_run_rnn_lstm_last_$(date +%Y%m%d_%H%M%S).log 2>&1 &
