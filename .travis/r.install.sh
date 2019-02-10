#/bin/bash
set -x -e

# Install needed repo
apt-get -y -qq update
apt-get -y -qq install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev > /dev/null
apt-get -y -qq install apt-transport-https ca-certificates curl gnupg2 software-properties-common > /dev/null
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran35/"
apt-get -y -qq update
DEBIAN_FRONTEND=nonexteractive apt-get -y -qqq install r-base > /dev/null
echo "options(repos = c(CRAN='http://cran.rstudio.com'))" >> ~/.Rprofile
R -e 'install.packages(c("Rcpp", "reticulate", "knitr", "tensorflow", "tfdatasets", "forge", "tidyselect", "testthat", "devtools"), quiet = TRUE)'
