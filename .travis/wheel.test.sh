set -e

export TF_AZURE_USE_DEV_STORAGE=1

run_test() {
  entry=$1
  CPYTHON_VERSION=$($entry -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
  (cd wheelhouse && $entry -m pip install tensorflow_io-*-cp${CPYTHON_VERSION}-*.whl)
  $entry -m pip install -q pytest boto3 google-cloud-pubsub==0.39.1 pyarrow==0.14.1 pandas==0.24.2 scikit-learn==0.20.4 gast==0.2.2
  (cd tests && $entry -m pytest -v --import-mode=append $(find . -type f \( -iname "test_*.py" ! \( -iname "test_*_eager.py" -o -iname "test_grpc.py" -o -iname "test_gcs_config_ops.py" \) \)))
  (cd tests && $entry -m pytest -v --import-mode=append $(find . -type f \( -iname "test_*_eager.py" \)))
  (cd tests && $entry -m pytest -v --import-mode=append $(find . -type f \( -iname "test_grpc.py" \)))
  (cd tests && $entry -m pytest -v --import-mode=append $(find . -type f \( -iname "test_gcs_config_ops.py" \)))
}

PYTHON_VERSION=python
if [[ "$#" -gt 0 ]]; then
  PYTHON_VERSION="${1}"
  shift
fi
if [[ $(uname) == "Linux" ]]; then
  apt-get -y -qq update
  apt-get -y -qq install $PYTHON_VERSION ffmpeg  dnsutils
  curl -sSOL https://bootstrap.pypa.io/get-pip.py
  $PYTHON_VERSION get-pip.py -q
fi
run_test $PYTHON_VERSION
