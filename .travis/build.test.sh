set -e

export TENSORFLOW_INSTALL="${1}"

# Path to shared libraries for running pytest
export TFIO_DATAPATH="bazel-bin"

if [[ $(uname) == "Linux" ]]; then
  apt-get -y -qq update
  apt-get -y -qq install python python3 ffmpeg > /dev/null
  curl -sSOL https://bootstrap.pypa.io/get-pip.py
  for entry in python3 python ; do
    $entry get-pip.py -q
    $entry -m pip --version
    $entry -m pip install -q "${TENSORFLOW_INSTALL}"
    $entry -m pip install -q pytest boto3 pyarrow==0.11.1 pandas==0.19.2
    $entry -m pytest tests
  done
fi
