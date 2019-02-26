set -e

python --version
python -m pip --version
if [[ $(uname) == "Darwin" ]]; then
  python -m pip install -q delocate
  delocate-wheel --version
else
  apt-get -y -qq update
  apt-get -y -qq install software-properties-common apt-transport-https
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get -y -qq update
  apt-get -y -qq install python3 python3.5 python3.6

  # Install patchelf
  curl -sSOL https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.bz2
  tar xfa patchelf-0.9.tar.bz2
  bash -c -e '(cd patchelf-0.9 && ./configure -q --prefix=/usr && make -s && make -s install)'
  rm -rf patchelf-0.9*

  # Install the latest version of pip
  curl -sSOL https://bootstrap.pypa.io/get-pip.py
  python3.6 get-pip.py -q
  python3.5 get-pip.py -q
  python3 get-pip.py -q

  # Install auditwheel
  python3 -m pip install -q auditwheel==1.5.0
  python3 -m pip install -q wheel==0.31.1
  auditwheel --version
fi
