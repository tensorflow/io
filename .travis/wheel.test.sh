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

if [[ ( $(uname) == "Darwin" ) || ( $(python -c 'import tensorflow as tf; print(tf.version.VERSION)') == "2.0"* ) ]]; then
  # Skip macOS or preview build
  exit 0
fi

DEBIAN_FRONTEND=noninteractive apt-get -y -qq install software-properties-common apt-transport-https > /dev/null
DEBIAN_FRONTEND=noninteractive apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
DEBIAN_FRONTEND=noninteractive add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran35/"
DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
# Note: libpython-dev libpython3-dev are needed to for reticulate to see python binding (/usr/lib/libpython2.7.so)
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install libpython-dev libpython3-dev > /dev/null
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install r-base > /dev/null
echo "options(repos = c(CRAN='http://cran.rstudio.com'))" >> ~/.Rprofile
R -e 'install.packages(c("tensorflow"), quiet = TRUE)'
R -e 'install.packages(c("testthat", "devtools"), quiet = TRUE)'
R -e 'install.packages(c("forge"), quiet = TRUE)'
R -e 'library("devtools"); install_github("rstudio/tfdatasets", ref="c6fc59b", quiet = TRUE)'

(cd R-package && R -e 'v <- data.frame(devtools::test()); stopifnot(all(!v$failed && !v$skipped && !v$error))')
