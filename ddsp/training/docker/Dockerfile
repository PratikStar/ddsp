# Uses one of AI Platform base images.
# You can try using different images however only this one has been tested.
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-2

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Installs sndfile library for reading and writing audio files.
RUN apt-get update && \
    apt-get install --no-install-recommends -y libsndfile-dev

# Error: NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/opt/conda/lib/python3.7/site-packages/tensorflow_io/core/python/ops/libtensorflow_io.so']
# https://stackoverflow.com/questions/65623468/unable-to-open-file-libtensorflow-io-so-caused-by-undefined-symbol
RUN pip uninstall -y tensorflow
RUN pip uninstall -y tensorflow-io

# Upgrades Tensorflow and Tensorflow Probability
# Newer version of Tensorflow is needed for multiple VMs training
RUN pip install --upgrade pip && \
    pip install --upgrade tensorflow==2.11.0rc0 tensorflow-probability

RUN pip install --no-deps tensorflow-io


# Installs cloudml-hypertune package needed for hyperparameter tuning
RUN pip install cloudml-hypertune

WORKDIR /root

RUN pip install ddsp
# Copies running script.
COPY task.py task.py

ENTRYPOINT ["python", "task.py"]
