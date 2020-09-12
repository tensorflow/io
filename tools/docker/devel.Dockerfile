FROM tensorflow/tensorflow:custom-op-ubuntu16

RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list && apt-get update && \
    apt-get install -y \
    git \
    gcc \
    g++ \
    gdb \
    make \
    patch \
    curl \
    nano \
    unzip \
    ffmpeg \
    dnsutils

ARG BAZEL_VERSION=3.1.0
ARG BAZEL_OS=linux

RUN curl -sL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${BAZEL_OS}-x86_64.sh -o bazel-install.sh && \
    bash -x bazel-install.sh && \
    rm bazel-install.sh

ARG CONDA_OS=Linux

# Miniconda - Python 3.6, 64-bit, x86, latest
RUN curl -sL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o mconda-install.sh && \
    bash -x mconda-install.sh -b -p miniconda && \
    rm mconda-install.sh

ENV PATH="/miniconda/bin:$PATH"

ARG CONDA_ADD_PACKAGES=""

RUN conda create -y -q -n tfio-dev python=3.6 ${CONDA_ADD_PACKAGES}

ARG ARROW_VERSION=0.16.0

RUN echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "source activate tfio-dev" >> ~/.bashrc

ARG PIP_ADD_PACKAGES=""

RUN /bin/bash -c "source activate tfio-dev && python -m pip install \
    avro-python3 \
    pytest \
    pytest-benchmark \
    pylint \
    boto3 \
    google-cloud-pubsub==0.39.1 \
    pyarrow==${ARROW_VERSION} \
    pandas \
    fastavro \
    gast==0.2.2 \
    ${PIP_ADD_PACKAGES} \
    "

ENV TFIO_DATAPATH=bazel-bin
