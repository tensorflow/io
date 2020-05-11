# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""

import os
import sys
import shutil
import tempfile
import fnmatch
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# read package and version from:
# tensorflow_io/core/python/ops/version_ops.py
with open(os.path.join(here, "tensorflow_io/core/python/ops/version_ops.py")) as f:
    entries = [e.strip() for e in f.readlines() if not e.startswith("#")]
    assert sum(e.startswith("package = ") for e in entries) == 1
    assert sum(e.startswith("version = ") for e in entries) == 1
    package = list([e[10:] for e in entries if e.startswith("package = ")])[0].strip(
        '"'
    )
    version = list([e[10:] for e in entries if e.startswith("version = ")])[0].strip(
        '"'
    )
    assert package != ""
    assert version != ""

if "--package-version" in sys.argv:
    print(package)
    sys.exit(0)

project = "tensorflow-io"
if "--nightly" in sys.argv:
    nightly_idx = sys.argv.index("--nightly")
    version = version + ".dev" + sys.argv[nightly_idx + 1]
    project = "tensorflow-io-nightly"
    sys.argv.remove("--nightly")
    sys.argv.pop(nightly_idx)

datapath = None
if "--data" in sys.argv:
    data_idx = sys.argv.index("--data")
    datapath = sys.argv[data_idx + 1]
    sys.argv.remove("--data")
    sys.argv.pop(data_idx)
else:
    datapath = os.environ.get("TFIO_DATAPATH")

if (datapath is not None) and ("bdist_wheel" in sys.argv):
    rootpath = tempfile.mkdtemp()
    print("setup.py - create {} and copy tensorflow_io data files".format(rootpath))
    for rootname, _, filenames in os.walk(os.path.join(datapath, "tensorflow_io")):
        if not fnmatch.fnmatch(rootname, "*test*") and not fnmatch.fnmatch(
            rootname, "*runfiles*"
        ):
            for filename in [
                f
                for f in filenames
                if fnmatch.fnmatch(f, "*.so") or fnmatch.fnmatch(f, "*.py")
            ]:
                # NOTE:
                # cc_grpc_library will generate a lib<name>_cc_grpc.so
                # proto_library will generate a lib<name>_proto.so
                # both .so files are not needed in final wheel.
                # The cc_grpc_library only need to pass `linkstatic = True`
                # to the underlying native.cc_library. However it is not
                # exposed. proto_library is a native library in bazel which
                # we could not patch easily as well.
                # For that reason we skip lib<name>_cc_grpc.so and lib<name>_proto.so:
                if filename.endswith("_cc_grpc.so") or filename.endswith("_proto.so"):
                    continue
                src = os.path.join(rootname, filename)
                dst = os.path.join(
                    rootpath,
                    os.path.relpath(os.path.join(rootname, filename), datapath),
                )
                print("setup.py - copy {} to {}".format(src, dst))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
    sys.argv.append("--bdist-dir")
    sys.argv.append(rootpath)

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


class BinaryDistribution(setuptools.dist.Distribution):
    def has_ext_modules(self):
        return True


setuptools.setup(
    name=project,
    version=version,
    description="TensorFlow IO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorflow/io",
    download_url="https://github.com/tensorflow/io/tags",
    author="Google Inc.",
    author_email="opensource@google.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="tensorflow io machine learning",
    packages=setuptools.find_packages(where=".", exclude=["tests"]),
    python_requires=">=3.5, <3.9",
    install_requires=[package],
    package_data={".": ["*.so"],},
    project_urls={
        "Source": "https://github.com/tensorflow/io",
        "Bug Reports": "https://github.com/tensorflow/io/issues",
        "Documentation": "https://tensorflow.org/io",
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
