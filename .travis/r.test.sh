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
    echo R with Python 2.7 Ubuntu 16.04
    .travis/r.install.sh
    apt-get -y -qq install ffmpeg python-pip > /dev/null
    pip install -q tensorflow==${TENSORFLOW_VERSION}
    pip install -q artifacts/tensorflow_io-*-cp27-*.whl
    (cd R-package && R -e 'stopifnot(all(data.frame(devtools::test())$failed == 0L))')
    echo Success
    exit 0
  fi
elif [[ $(lsb_release -r | awk '{ print $2 }') == "18.04" ]]; then
  if [[ ${PYTHON_VERSION} == "2.7" ]]; then
    echo R with Python 2.7 Ubuntu 18.04
    .travis/r.install.sh
    apt-get -y -qq install ffmpeg python-pip > /dev/null
    pip install -q tensorflow==${TENSORFLOW_VERSION}
    pip install -q artifacts/tensorflow_io-*-cp27-*.whl
    (cd R-package && R -e 'stopifnot(all(data.frame(devtools::test())$failed == 0L))')
    echo Success
    exit 0
  fi
fi
echo R with Python ${1} on Ubuntu $(lsb_release -r | awk '{ print $2 }') not tested
