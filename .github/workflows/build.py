import subprocess, sys, os

path = os.getcwd()
with subprocess.Popen(
        [
            "docker",
            "run",
            "-i",
            "-e",
            "TF_PYTHON_VERSION=3.9",
            "-v",
            f"{path}:/v",
            "-w",
            "/v",
            "--net=host",
            "--entrypoint=/bin/bash",
            "gcr.io/tensorflow-testing/nosla-cuda12.0.1-cudnn8.8-ubuntu20.04-manylinux2014-multipython",
            "-x",
            "-e",
            ".github/workflows/build.bazel.sh",
            "python3.9",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        for line in process.stdout:
            print(line.decode().translate(dict.fromkeys(range(32))).strip())
