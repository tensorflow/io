set -e

export TF_AZURE_USE_DEV_STORAGE=1
export TF_USE_MODULAR_FILESYSTEM=1

run_test() {
  entry=$1
  CPYTHON_VERSION=$($entry -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
  TF_VERSION=$(/usr/bin/grep tensorflow tensorflow_io/python/ops/version_ops.py | /usr/bin/cut -d '"' -f 2)
  $entry -m pip install $TF_VERSION pytest pytest-benchmark pytest-xdist==2.5.0 boto3 fastavro avro-python3 scikit-image pandas pyarrow==3.0.0 google-cloud-pubsub==2.1.0 google-cloud-bigtable==1.6.0 google-cloud-bigquery-storage==1.1.0 google-cloud-bigquery==2.3.1 google-cloud-storage==1.32.0 PyYAML==5.3.1 azure-storage-blob==12.8.1 azure-cli==2.29.0
  $entry -m pip uninstall -y tensorflow_io_gcs_filesystem
  (cd wheelhouse && $entry -m pip install tensorflow_io_gcs_filesystem-*-cp${CPYTHON_VERSION}-*.whl)
  (cd wheelhouse && $entry -m pip install tensorflow_io-*-cp${CPYTHON_VERSION}-*.whl)
  $entry -m pip freeze
  (cd tests && $entry -m pytest --benchmark-disable -v --import-mode=append --forked --numprocesses=auto --dist loadfile $(find . -type f \( -iname "test_*.py" ! \( -iname "test_standalone_*.py" \) \)))
  (cd tests && $entry -m pytest --benchmark-disable -v --import-mode=append $(find . -type f \( -iname "test_standalone_*.py" \)))
}

PYTHON_VERSION=python
if [[ "$#" -gt 0 ]]; then
  PYTHON_VERSION="${1}"
  shift
fi

if [[ $(uname) == "Linux" ]]; then
  apt-get -y -qq update
  if [[ "${PYTHON_VERSION}" == "python3.9" ]]; then
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get -y -qq update
    apt-get -y -qq install python3.9-distutils
  fi
  apt-get -y -qq install $PYTHON_VERSION ffmpeg  dnsutils libmp3lame0
  curl -sSOL https://bootstrap.pypa.io/get-pip.py
  $PYTHON_VERSION get-pip.py -q

  # Install Java
  apt-get -y -qq install openjdk-8-jdk
  update-alternatives --config java
  export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

  # Install Hadoop
  curl -OL https://archive.apache.org/dist/hadoop/common/hadoop-2.7.0/hadoop-2.7.0.tar.gz
  tar -xzf hadoop-2.7.0.tar.gz -C /usr/local
  export HADOOP_HOME=/usr/local/hadoop-2.7.0

  # Update environmental variable
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native
  export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
  export
elif [[ $(uname) == "Darwin" ]]; then
  brew install ffmpeg
fi
run_test $PYTHON_VERSION
