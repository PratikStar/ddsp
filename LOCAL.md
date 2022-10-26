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
# install miniconda
sudo apt-get install libsndfile-dev
pip install --upgrade pip
pip install --upgrade ddsp

gcsfuse --> https://medium.com/google-cloud/scheduled-mirror-sync-sftp-to-gcs-b167d0eb487a
gcsfuse pratik-ddsp-data ~/bucket
gcsfuse pratik-ddsp-tfrecord ~/bucket-tfrecord

pip install tensorflow==2.11.0rc0
pip install apache-beam

sudo apt install ffmpeg

# data prep
ddsp_prepare_tfrecord \
--input_audio_filepatterns='/root/bucket/*wav' \
--output_tfrecord_path=/root/tfrecord/train.tfrecord \
--num_shards=10 \
--alsologtostderr

# just download tfrecords from wisteria to test ddsp_run
rsync -av w:/work/gk77/k77021/data/ddsp/monophonic "/Users/pratik/data/ddsp_tfrecords"


ddsp_run \
  --mode=train \
  --save_dir=/root/ddsp/save_dir \
  --gin_file=/root/ddsp/ddsp/training/gin/models/solo_instrument.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/datasets/tfrecord.gin \
  --gin_file=/root/ddsp/ddsp/training/gin/eval/basic_f0_ld.gin \
  --gin_param="TFRecordProvider.file_pattern='/root/bucket-tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr
  
gcloud compute ssh instance-1 --zone us-east1-b --command "sudo expand-root.sh /dev/sda 1 ext4"

wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb
```


## Image build and push
cd /root/ddsp/ddsp/training/docker

export PROJECT_ID=ddsp-366504
export IMAGE_REPO_NAME=ddsp_train
export IMAGE_TAG=ai_platform
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

[//]: #(https://stackoverflow.com/questions/55446787/permission-issues-while-docker-push) 


## Submit job

export SAVE_DIR=gs://pratik-ddsp-models
export FILE_PATTERN=gs://pratik-ddsp-tfrecord/train.tfrecord*
export REGION=us-east1
export JOB_NAME=ddsp_container_job_$(date +%Y%m%d_%H%M%S)

cd /root/ddsp/ddsp/training/docker

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