set -e

export TF_AZURE_USE_DEV_STORAGE=1

run_test() {
  entry=$1
  CPYTHON_VERSION=$($entry -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
  (cd wheelhouse && $entry -m pip install tensorflow_io-*-cp${CPYTHON_VERSION}-*.whl)
  $entry -m pip install -q pytest pytest-benchmark boto3 python-dateutil==2.8.0 google-cloud-pubsub==0.39.1 pyarrow==0.14.1 pandas==0.24.2 scikit-learn==0.20.4 gast==0.2.2 google-cloud-bigquery-storage==0.7.0 google-cloud-bigquery==1.22.0 fastavro
  (cd tests && $entry -m pytest --benchmark-disable -v --import-mode=append $(find . -type f \( -iname "test_*.py" ! \( -iname "test_*_eager.py" -o -iname "test_gcs_config_ops.py" \) \)))
  (cd tests && $entry -m pytest --benchmark-disable -v --import-mode=append $(find . -type f \( -iname "test_*_eager.py" ! \( -iname "test_bigquery_eager.py" \) \)))
  # GRPC and test_bigquery_eager tests have to be executed separately because of https://github.com/grpc/grpc/issues/20034
  (cd tests && $entry -m pytest --benchmark-disable -v --import-mode=append $(find . -type f \( -iname "test_bigquery_eager.py" \)))
  (cd tests && $entry -m pytest --benchmark-disable -v --import-mode=append $(find . -type f \( -iname "test_gcs_config_ops.py" \)))
}

PYTHON_VERSION=python
if [[ "$#" -gt 0 ]]; then
  PYTHON_VERSION="${1}"
  shift
fi
if [[ $(uname) == "Linux" ]]; then
  apt-get -y -qq update
  if [[ "${PYTHON_VERSION}" == "python3.7" ]]; then
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get -y -qq update
  fi
  apt-get -y -qq install $PYTHON_VERSION ffmpeg  dnsutils
  curl -sSOL https://bootstrap.pypa.io/get-pip.py
  $PYTHON_VERSION get-pip.py -q
fi
run_test $PYTHON_VERSION
