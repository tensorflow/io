#/bin/bash
set -x -e

PYTHON_VERSION=${1}
shift

TENSORFLOW_VERSION=1.12.0
if [[ ! -z ${1} ]]; then
  TENSORFLOW_VERSION=${1}
  shift
fi

apt-get -y -qq update
apt-get -y -qq install lsb-core > /dev/null
if [[ $(lsb_release -r | awk '{ print $2 }') == "16.04" ]]; then
  if [[ ${PYTHON_VERSION} == "2.7" ]]; then
    echo Python 2.7 Ubuntu 16.04
    apt-get -y -qq install ffmpeg python-pip > /dev/null
    pip install -q tensorflow==${TENSORFLOW_VERSION}
    pip install -q artifacts/tensorflow_io*-cp27-*.whl
    pip install -q pytest
    # Pin more-itertools==5.0.0 as otherwise pytest will fail
    pip install -q more-itertools==5.0.0
    (cd tests && python -m pytest -s .)
    echo Success
    exit 0
  elif [[ ${PYTHON_VERSION} == "3.5" ]]; then
    echo Python 3.5 Ubuntu 16.04
    apt-get -y -qq install ffmpeg python3-pip > /dev/null
    pip3 install -q tensorflow==${TENSORFLOW_VERSION}
    pip3 install -q artifacts/tensorflow_io*-cp35-*.whl
    pip3 install -q pytest
    # Pin more-itertools==5.0.0 as otherwise pytest will fail
    pip3 install -q more-itertools==5.0.0
    (cd tests && python3 -m pytest -s .)
    echo Success
    exit 0
  fi
elif [[ $(lsb_release -r | awk '{ print $2 }') == "18.04" ]]; then
  apt-get -y -qq install ffmpeg > /dev/null
  if [[ ${PYTHON_VERSION} == "2.7" ]]; then
    echo Python 2.7 Ubuntu 18.04
    apt-get -y -qq install ffmpeg python-pip > /dev/null
    pip install -q tensorflow==${TENSORFLOW_VERSION}
    pip install -q artifacts/tensorflow_io*-cp27-*.whl
    pip install -q pytest
    # Pin more-itertools==5.0.0 as otherwise pytest will fail
    pip install -q more-itertools==5.0.0
    (cd tests && python -m pytest -s .)
    echo Success
    exit 0
  elif [[ ${PYTHON_VERSION} == "3.6" ]]; then
    echo Python 3.6 Ubuntu 18.04
    apt-get -y -qq install ffmpeg python3-pip > /dev/null
    pip3 install -q tensorflow==${TENSORFLOW_VERSION}
    pip3 install -q artifacts/tensorflow_io*-cp36-*.whl
    pip3 install -q pytest
    # Pin more-itertools==5.0.0 as otherwise pytest will fail
    pip3 install -q more-itertools==5.0.0
    (cd tests && python3 -m pytest -s .)
    echo Success
    exit 0
  fi
fi
echo Python ${1} on Ubuntu $(lsb_release -r | awk '{ print $2 }') not tested
