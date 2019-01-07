FROM r-base
COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python-dev \
      python-setuptools \
      python-pip && \
    rm -rf /var/lib/apt/lists/*

# Dependencies
RUN pip install tensorflow-io
RUN Rscript -e 'install.packages(c("Rcpp", "reticulate", "knitr", "tensorflow", "tfdatasets", "forge", "tidyselect"))'

# tfio package installation
RUN R CMD build R-package/
RUN R CMD INSTALL tfio_*.gz
