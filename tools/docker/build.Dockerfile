ARG PYTHON_VERSION
ARG TENSORFLOW_VERSION
FROM tensorflow/build:${TENSORFLOW_VERSION}-python$PYTHON_VERSION

ARG PYTHON_VERSION
ARG TENSORFLOW_VERSION
ARG BAZEL_OPTIMIZATION

ADD . /opt/io
WORKDIR /opt/io

RUN python${PYTHON_VERSION} -m pip install tensorflow==${TENSORFLOW_VERSION}

RUN python$PYTHON_VERSION -m pip uninstall -y tensorflow-io-gcs-filesystem

RUN python$PYTHON_VERSION tools/build/configure.py

RUN cat .bazelrc

RUN TF_PYTHON_VERSION=${PYTHON_VERSION} bazel build --copt="-fPIC" --crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.8-cudnn8.6-tensorrt8.4_config_cuda//crosstool:toolchain --noshow_progress --verbose_failures ${BAZEL_OPTIMIZATION} -- //tensorflow_io/...  //tensorflow_io_gcs_filesystem/...

