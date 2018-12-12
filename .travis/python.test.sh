gcc -v
python --version
bazel test -s --verbose_failures //tensorflow_io/hadoop:hadoop_py_test
bazel test -s --verbose_failures //tensorflow_io/kafka:kafka_py_test
bazel test -s --verbose_failures //tensorflow_io/ignite:ignite_py_test
bazel test -s --verbose_failures //tensorflow_io/ignite:igfs_py_test
