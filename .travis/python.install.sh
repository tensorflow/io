#/bin/bash
set -x -e

apt-get -y -qq update
apt-get -y -qq install lsb-core > /dev/null
if [[ $(lsb_release -r | awk '{ print $2 }') == "14.04" ]]; then
  apt-get -y -qq install libav-tools > /dev/null
  if [[ ${1} == "2.7" ]]; then
    pip install boto3
    echo "Python 2.7"
    pip install pyarrow==0.11.1
  elif [[ ${1} == "3.4" ]]; then
    rm -f /usr/local/bin/pip
    ln -s /usr/local/bin/pip3.4 /usr/local/bin/pip
    rm -f /usr/bin/python
    ln -s /usr/bin/python3.4 /usr/bin/python
    pip install boto3
  elif [[ ${1} == "3.5" ]]; then
    curl -sOL https://raw.githubusercontent.com/tensorflow/tensorflow/v1.12.0/tensorflow/tools/ci_build/install/install_python3.5_pip_packages.sh
    sed -i 's/apt-get update/apt-get -y -qq update/g' install_python3.5_pip_packages.sh
    sed -i 's/apt-get install/apt-get -y -qq install/g' install_python3.5_pip_packages.sh
    sed -i 's/pip3.5 install/pip3.5 -q install/g' install_python3.5_pip_packages.sh
    bash install_python3.5_pip_packages.sh
    rm -f install_python3.5_pip_packages.sh 
    rm -f /usr/bin/python
    ln -s /usr/bin/python3.5 /usr/bin/python
    rm -f /usr/local/bin/pip
    ln -s /usr/local/bin/pip3.5 /usr/local/bin/pip
    pip install pyarrow==0.11.1
    pip install boto3
  elif [[ ${1} == "3.6" ]]; then
    rm -f /usr/local/bin/pip3
    curl -sOL https://raw.githubusercontent.com/tensorflow/tensorflow/v1.12.0/tensorflow/tools/ci_build/install/install_python3.6_pip_packages.sh
    sed -i 's/apt-get update/apt-get -y -qq update/g' install_python3.6_pip_packages.sh
    sed -i 's/apt-get install/apt-get -y -qq install/g' install_python3.6_pip_packages.sh
    sed -i 's/apt-get upgrade/apt-get -y -qq upgrade/g' install_python3.6_pip_packages.sh
    sed -i 's/pip3 install/pip3 -q install/g' install_python3.6_pip_packages.sh
    sed -i 's/tar xvf/tar xf/g' install_python3.6_pip_packages.sh
    sed -i 's/configure/configure -q/g' install_python3.6_pip_packages.sh
    sed -i 's/make altinstall/make altinstall>make.log/g' install_python3.6_pip_packages.sh
    sed -i 's/wget /wget -q /g' install_python3.6_pip_packages.sh
    bash install_python3.6_pip_packages.sh
    rm -f install_python3.6_pip_packages.sh make.log
    rm -rf Python-3.6.1*
    rm -f /usr/bin/python
    ln -s /usr/local/bin/python3.6 /usr/bin/python
    rm -f /usr/local/bin/pip
    ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip
    pip install pyarrow==0.11.1
    pip install boto3
  else
    echo Python ${1} not supported!
    exit 1
  fi
elif [[ $(lsb_release -r | awk '{ print $2 }') == "16.04" ]]; then
  if [[ ${1} == "2.7" ]]; then
    echo "Python 2.7"
    apt-get -y -qq install ffmpeg python-pip python3-pip patchelf > /dev/null
    pip3 install -q auditwheel==1.5.0
    # Pin wheel==0.31.1 to work around issue
    # https://github.com/pypa/auditwheel/issues/102
    pip3 install -q wheel==0.31.1
    pip install pyarrow==0.11.1 pandas==0.19.2
    pip install boto3
  elif [[ ${1} == "3.5" ]]; then
    echo "Python 3.5"
    apt-get -y -qq install ffmpeg python3-pip patchelf > /dev/null
    pip3 install -q auditwheel==1.5.0
    # Pin wheel==0.31.1 to work around issue
    # https://github.com/pypa/auditwheel/issues/102
    pip3 install wheel==0.31.1
    rm -f /usr/bin/python
    ln -s /usr/bin/python3 /usr/bin/python
    rm -f /usr/bin/pip
    ln -s /usr/bin/pip3 /usr/bin/pip
    pip install pyarrow==0.11.1 pandas==0.19.2
    pip install boto3
  else
    echo Platform $(lsb_release -r | awk '{ print $2 }') not supported!
    exit 1
  fi
elif [[ $(lsb_release -r | awk '{ print $2 }') == "18.04" ]]; then
  if [[ ${1} == "2.7" ]]; then
    echo "Python 2.7"
    apt-get -y -qq install ffmpeg python-pip python3-pip python3-wheel patchelf > /dev/null
    pip3 install -q auditwheel==1.5.0
    # Pin wheel==0.31.1 to work around issue
    # https://github.com/pypa/auditwheel/issues/102
    pip3 install wheel==0.31.1
    pip install pyarrow==0.11.1 pandas==0.19.2
    pip install boto3
  elif [[ ${1} == "3.6" ]]; then
    echo "Python 3.6"
    apt-get -y -qq install ffmpeg python3-pip python3-wheel patchelf > /dev/null
    pip3 install -q auditwheel==1.5.0
    # Pin wheel==0.31.1 to work around issue
    # https://github.com/pypa/auditwheel/issues/102
    pip3 install wheel==0.31.1
    rm -f /usr/bin/python
    ln -s /usr/bin/python3 /usr/bin/python
    rm -f /usr/bin/pip
    ln -s /usr/bin/pip3 /usr/bin/pip
    pip install pyarrow==0.11.1 pandas==0.19.2
    pip install boto3
  else
    echo Platform $(lsb_release -r | awk '{ print $2 }') not supported!
    exit 1
  fi
else
  echo Platform $(lsb_release -r | awk '{ print $2 }') not supported!
  exit 1
fi
python --version
pip --version
pip freeze
