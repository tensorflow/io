# DAOS-TensorFlow IO GUIDE

## Table Of Content

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Building](#building)
- [Testing](#testing)
- [Example](#example)

## Features

* Providing a plugin utilizing the DAOS DFS layer to provide efficient utilization for Intel's filesystem.

## Prerequisites

* A valid DAOS installation, currently based on [version v1.3.106](https://github.com/daos-stack/daos/releases/tag/v1.3.106-tb)
  * An installation guide and steps can be accessed from [here](https://docs.daos.io/admin/installation/)

## Environment Setup

Assuming you are in a terminal in the repository root directory:

* Install latest versions of the following dependencies by running
  * Centos 8
    ```
    $ yum install -y python3 python3-devel gcc gcc-c++ git unzip which make
    ```
  *  Ubuntu 20.04
     ```
     $ sudo apt-get -y -qq update 
     $ sudo apt-get -y -qq install gcc g++ git unzip curl python3-pip
     ```
* Download the Bazel installer
  ```
    $ curl -sSOL https://github.com/bazelbuild/bazel/releases/download/\$(cat .bazelversion)/bazel-\$(cat .bazelversion)-installer-linux-x86_64.sh
  ```
* Install Bazel
  ```
  $ bash -x -e bazel-$(cat .bazelversion)-installer-linux-x86_64.sh
  ```
* Update Pip and install pytest
  ```
  $ python3 -m pip install -U pip
  $ python3 -m pip install pytest
  ```
  
## Building

Assuming you are in a terminal in the repository root directory:

* Configure and install tensorflow (the current version should be tensorflow2.6.2)
  ```
  $ ./configure.sh
  ## Set python3 as default.
  $ ln -s /usr/bin/python3 /usr/bin/python
  ```

* At this point, all libraries and dependencies should be installed.  
  * Make sure the environment variable **LD_LIBRARY_PATH** includes the paths to:
    * All daos libraries
    * The tensorflow framework (libtensorflow and libtensorflow_framework)
  * If not, find the required libraries and add their paths to the environment variable
    ```
    export LD_LIBRARY_PATH="<path-to-library>:$LD_LIBARY_PATH"
    ```
  * Make sure the environment variable **CPLUS_INCLUDE_PATH** and **C_INCLUDE_PATH** includes the paths to:
    * The tensorflow headers (usually in /usr/local/lib64/python3.6/site-packages/tensorflow/include)
  * If not, find the required headers and add their paths to the environment variable
    ```
    export CPLUS_INCLUDE_PATH="<path-to-headers>:$CPLUS_INCLUDE_PATH"
    export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$C_INCLUDE_PATH
    ```

* Build the project using bazel
  ```
  bazel build -s --verbose_failures //tensorflow_io/... //tensorflow_io_gcs_filesystem/...
  ```
  This should take a few minutes. Note that sandboxing may result in build failures when using Docker Containers for DAOS due to mounting issues, if that’s the case, add **--spawn_strategy=standalone** to the above build command to bypass sandboxing. (When disabling sandbox, an error may be thrown for an undefined type z_crc_t due to a conflict in header files. Please find the crypt.h file in the bazel cache in subdirectory /external/zlib/contrib/minizip/crypt.h and add the following line to the file **typedef unsigned long z_crc_t;** then re-build)



## Testing
Assuming you are in a terminal in the repository root directory:

* Run the following command for the simple serial test to validate building. Note that any tests need to be run with the TFIO_DATAPATH flag to specify the location of the binaries.
  ```
  $ TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py

  ```

* Run the following commands to run the dfs plugin test:
  ```
  # To create the required pool and container and export required env variables for the dfs tests.
  $ source tests/test_dfs/dfs_init.sh
  # To run dfs tests
  $ TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_dfs.py
  # For Cleanup, deletes pools and containers created for test.
  $ bash ./tests/test_dfs/dfs_cleanup.sh
  ```

## Example

Please refer to [the DAOS notebook example in the tutorials folder in docs folder.](tutorials/daos.ipynb)

