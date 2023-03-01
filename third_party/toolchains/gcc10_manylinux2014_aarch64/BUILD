# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This becomes the BUILD file for @local_config_cc// under non-FreeBSD unixes.

package(default_visibility = ["//visibility:public"])

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "malloc",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "compiler_deps",
    srcs = glob(["extra_tools/**"]) + [":empty"],
)

# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# the "${CPU}" entry in the toolchains attribute.
cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-aarch64",
        "aarch64": ":cc-compiler-aarch64",
    },
)

cc_toolchain(
    name = "cc-compiler-aarch64",
    all_files = ":compiler_deps",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":compiler_deps",
    dwp_files = ":empty",
    linker_files = ":compiler_deps",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":linux_gnu_aarch64",
    toolchain_identifier = "linux_gnu_aarch64",
)

cc_toolchain_config(
    name = "linux_gnu_aarch64",
    compiler = "/dt10/usr/bin/gcc",
    cpu = "aarch64",
)

toolchain(
    name = "cc-toolchain-aarch64",
    exec_compatible_with = [
        # TODO(katre): add autodiscovered constraints for host CPU and OS.
    ],
    target_compatible_with = [
        # TODO(katre): add autodiscovered constraints for host CPU and OS.
    ],
    toolchain = ":cc-compiler-aarch64",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
