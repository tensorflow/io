# Tensorflow-IO Dockerfiles

This directory maintains the Dockerfiles needed to build the tensorflow-io images.

## Building

To build a `tensorflow-io` image with CPU support:

```bash
$ docker build -f ./cpu.Dockerfile -t tfio-cpu .
```

**NOTE:** Each `.Dockerfile` has its own set of available `--build-arg`s which are documented
in the file itself.

## Running Locally Built Images

**Note for new Docker users:** the `-v` and `-u` flags share directories and
permissions between the Docker container and your machine. Without `-v`, your
work will be wiped once the container quits, and without `-u`, files created by
the container will have the wrong file permissions on your host machine. Check
out the
[Docker run documentation](https://docs.docker.com/engine/reference/run/) for
more info.

```sh
# Mount $PWD into the container and make it as the current working directory.
$ docker run -it --rm -v ${PWD}:/v -w /v tfio-cpu
```