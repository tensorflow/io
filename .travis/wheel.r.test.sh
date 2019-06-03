set -e

apt-get -y -qq update
apt-get -y -qq install python ffmpeg
curl -sSOL https://bootstrap.pypa.io/get-pip.py
python get-pip.py -q

CPYTHON_VERSION=$(python -c 'import sys; print(str(sys.version_info[0])+str(sys.version_info[1]))')
(cd wheelhouse && python -m pip install *-cp${CPYTHON_VERSION}-*.whl)
python -m pip install -q pytest boto3 google-cloud-pubsub==0.39.1 pyarrow==0.11.1 pandas==0.19.2

set +e
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install software-properties-common apt-transport-https
DEBIAN_FRONTEND=noninteractive apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
DEBIAN_FRONTEND=noninteractive add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran35/"
DEBIAN_FRONTEND=noninteractive apt-get -y -qq update
# Note: libpython-dev libpython3-dev are needed to for reticulate to see python binding (/usr/lib/libpython2.7.so)
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install libpython-dev libpython3-dev
DEBIAN_FRONTEND=noninteractive apt-get -y -qq install r-base
set -e
echo "options(repos = c(CRAN='http://cran.rstudio.com'))" >> ~/.Rprofile
R -e 'install.packages(c("tensorflow"), quiet = TRUE)'
R -e 'install.packages(c("testthat", "devtools"), quiet = TRUE)'
R -e 'install.packages(c("forge"), quiet = TRUE)'
R -e 'library("devtools"); install_github("rstudio/tfdatasets", ref="c6fc59b", quiet = TRUE)'

(cd R-package && R -e 'v <- data.frame(devtools::test()); stopifnot(all(!v$failed && !v$skipped && !v$error))')
