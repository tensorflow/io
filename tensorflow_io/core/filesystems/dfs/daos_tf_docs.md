## DAOS-TF BUILD GUIDE

This assumes a running DAOS engine v1.3.106 (to be updated to 2.x). Please follow the following steps with the specified order, to build the tensorflow-io project and run the tests written for the DFS Plugin



* Install latest versions of the following dependencies by running “**yum install -y python3 python3-devel gcc gcc-c++ git unzip which make**” (assuming Centos 8) or “**sudo apt-get -y -qq update**” and “**sudo apt-get -y -qq install gcc g++ git unzip curl python3-pip**” for Ubuntu 20.04

* **“curl -sSOL https://github.com/bazelbuild/bazel/releases/download/\$(cat .bazelversion)/bazel-\$(cat .bazelversion)-installer-linux-x86_64.sh”** to download the Bazel installer

* “**bash -x -e bazel-$(cat .bazelversion)-installer-linux-x86_64.sh**” to install Bazel itself

* “**python3 -m pip install -U pip**”

* “**./configure.sh**” to configure and install tensorflow (the current version should be tensorflow2.6.2)

* “**ln -s /usr/bin/python3 /usr/bin/python**”

* At this point, all libraries and dependencies should be installed. Make sure the environment variable“$LD_LIBRARY_PATH” includes the paths to all the daos libraries ( should be at /opt/daos/lib and /opt/daos/lib64) and the tensorflow framework (libtensorflow and libtensorflow_framework)(usually at /usr/local/lib or /usr/local/lib64/python3.6/site-packages). If not, find the required libraries and add their paths to the variable “**export LD_LIBRARY_PATH=&lt;path-to-lib>:$LD_LIBARY_PATH**”

* **“bazel build -s --verbose_failures //tensorflow_io/... //tensorflow_io_gcs_filesystem/...”**. This should take a few minutes. (P.S Sandboxing may result in build failures when using Docker Containers for DAOS due to mounting issues, if that’s the case, add **--spawn_strategy=standalone** to the above build command to bypass sandboxing.

* “**python3 -m pip install pytest**” and then “**TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py**” . If the build is completed successfully, you can run the serialization test located under the tests directory to make sure everything is working fine. Note that any tests need to be run with the TFIO_DATAPATH flag to specify the location of the binaries

* **“source tests/test_dfs/dfs_init.sh”** to create the required pool and container and export required env variables for the dfs tests.

* “**TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_dfs.py”** to run dfs tests. For Cleanup, run the script labelled “**tests/test_dfs/dfs_cleanup.sh”**