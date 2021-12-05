## DAOS-TF BUILD GUIDE



* Before building, make sure to install latest versions of the following dependencies by running “**yum install -y python3 python3-devel gcc gcc-c++ git unzip which make**” (assuming Centos 8) or “**sudo apt-get -y -qq update**” and “**sudo apt-get -y -qq install gcc g++ git unzip curl python3-pip**” for Ubuntu 20.04

* Run “**curl -sSOL https://github.com/bazelbuild/bazel/releases/download/$(cat .bazelversion)/bazel-$(cat .bazelversion)-installer-linux-x86_64.sh**” to download the Bazel installer

* Run the installed bash script to install Bazel itself “**bash -x -e bazel-$(cat .bazelversion)-installer-linux-x86_64.sh**”

*  Make sure pip is upgraded “**python3 -m pip install -U pip**”

* Run the configuration script “**./configure.sh**” to configure and install tensorflow (the current version should be tensorflow2.6.2)

* Alias the python3 “**ln -s /usr/bin/python3 /usr/bin/python**”

* At this point, all libraries and dependencies should be installed. Make sure the environment variable“$LD_LIBRARY_PATH” includes the paths to all the daos libraries ( should be at /opt/daos/lib and /opt/daos/lib64) and the tensorflow framework (usually at /usr/local/lib or /usr/local/lib64/python3.6/site-packages). If not, find the required libraries and add their paths to the variable “**export LD_LIBRARY_PATH=&lt;path-to-lib>:$LD_LIBARY_PATH**”

* Build the tensorflow io project “**bazel build -s --verbose_failures //tensorflow_io/… //tensorflow_io_gcs_filesystem/...” **. This should take a few minutes. (P.S Sandboxing may result in build failures when using Docker Containers for DAOS due to mounting issues, if that’s the case, add **--spawn_strategy=standalone **to the above build command to bypass sandboxing.

* If the build is completed successfully, you can run the serialization test located under the tests directory to make sure everything is working fine. “**python3 -m pip install pytest**” and then “**TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py**” . Note that any tests need to be run with the TFIO_DATAPATH flag to specify the location of the binaries

* To run the dfs tests, first run the bash script in “**tests/test_dfs/dfs_test.sh” **to create the required pool and container, then similarly to the serialization tests, please run “**TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_dfs.py”**