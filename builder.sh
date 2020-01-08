# Start/restart docker
#systemctl restart docker
# Build and run the Docker image
docker build -f tools/dev/Dockerfile -t tfio-dev .
# Run the docker image
docker run --rm --net=host -v ${PWD}:/v -w /v tfio-dev
# Inside the container configure install
./configure.sh
# Build TensorFlow I/O. More info on flags https://www.tensorflow.org/install/source#configuration_options
bazel build -c opt --copt=-fPIC -s --verbose_failures //tensorflow_io/core/...
# Run c++ tests
bazel test --verbose_failures //tensorflow_io/avro/... --action_env=LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# Build the TensorFlow I/O package
python setup.py bdist_wheel
# TODO Install the wheel
pip install --upgrade --force-reinstall dist/tensorflow_io-*.whl
# Run tests
pytest -s -v tensorflow_io/avro/python/tests
