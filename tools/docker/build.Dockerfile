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

RUN TF_PYTHON_VERSION=${PYTHON_VERSION} bazel build --noshow_progress --verbose_failures ${BAZEL_OPTIMIZATION} -- //tensorflow_io/...  //tensorflow_io_gcs_filesystem/...

