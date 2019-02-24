set -e

run_test() {
  entry=$1
  CPYTHON_VERSION=$($entry -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
  (cd wheelhouse && $entry -m pip install *-cp${CPYTHON_VERSION}-*.whl)
  $entry -m pip install -q pytest boto3 pyarrow==0.11.1 pandas==0.19.2
  (cd tests && $entry -m pytest --import-mode=append .)
}

# If Linux just assume testing both python and python3
if [[ $(uname) == "Linux" ]]; then
  apt-get -y -qq update
  apt-get -y -qq install python python3 ffmpeg >/dev/null
  curl -sSOL https://bootstrap.pypa.io/get-pip.py
  python3 get-pip.py -q
  run_test python3
  python get-pip.py -q
fi

run_test python
