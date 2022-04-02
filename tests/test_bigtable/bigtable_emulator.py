# Copyright 2021 Google LLC
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

# disable module docstring for tests
# pylint: disable=C0114
# disable class docstring for tests
# pylint: disable=C0115
# disable warning for access to protected members
# pylint: disable=W0212
# disable warning for not using 'with' for resource-allocating operations
# pylint: disable=R1732
import os
import subprocess
import re
import tensorflow as tf
from threading import Thread
from typing import List

CBT_EMULATOR_SEARCH_PATHS = [
    "/usr/lib/google-cloud-sdk/platform/bigtable-emulator/cbtemulator",
    "/usr/local/google-cloud-sdk/platform/bigtable-emulator/cbtemulator",
    "/v/google-cloud-sdk/platform/bigtable-emulator/cbtemulator",
    "cbtemulator",
]

CBT_CLI_SEARCH_PATHS = [
    "/usr/local/google-cloud-sdk/bin/cbt",
    "/usr/bin/cbt",
    "/v/google-cloud-sdk/bin/cbt",
    "cbt",
]

CBT_EMULATOR_PATH_ENV_VAR = "CBT_EMULATOR_PATH"
CBT_CLI_PATH_ENV_VAR = "CBT_CLI_PATH"


def _get_cbt_binary_path(env_var_name, search_paths, description):
    res = os.environ.get(env_var_name)
    if res is not None:
        if not os.path.isfile(res):
            raise OSError(
                f"{description} specified in the {env_var_name} "
                "environment variable does not exist"
            )
        return res
    for candidate in search_paths:
        if os.path.isfile(candidate):
            return candidate
    raise OSError(f"Could not find {description}")


def _get_cbt_emulator_path():
    return _get_cbt_binary_path(
        CBT_EMULATOR_PATH_ENV_VAR, CBT_EMULATOR_SEARCH_PATHS, "cbt emulator"
    )


def _get_cbt_cli_path():
    return _get_cbt_binary_path(CBT_CLI_PATH_ENV_VAR, CBT_CLI_SEARCH_PATHS, "cbt cli")


def _extract_emulator_addr_from_output(emulator_output):
    while True:
        line = emulator_output.readline().decode()
        if not line:
            raise RuntimeError("CBT emulator stopped producing output")
        if "running on" in line:
            words = line.split()
            for word in words:
                if re.fullmatch("[a-z.0-9]+:[0-9]+", word):
                    return word
            raise RuntimeError(f"Failed to find CBT emulator in the line {line}")


class BigtableEmulator:
    def __init__(self):
        emulator_path = _get_cbt_emulator_path()
        self._emulator = subprocess.Popen(
            [emulator_path, "-port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        out = self._emulator.stdout
        self._emulator_addr = _extract_emulator_addr_from_output(out)
        self._output_reading_thread = Thread(target=out.read)
        self._output_reading_thread.start()

    def get_addr(self):
        return self._emulator_addr

    def create_table(
        self, project_id, instance_id, table_id, column_families, splits=None
    ):
        cli_path = _get_cbt_cli_path()
        cmd = [
            cli_path,
            "-project",
            project_id,
            "-instance",
            instance_id,
            "createtable",
            table_id,
            "families=" + ",".join([f"{fam}:never" for fam in column_families]),
        ]
        if splits:
            cmd.append("splits=" + ",".join(splits))
        subprocess.check_output(cmd)

    def write_tensor(
        self,
        project_id,
        instance_id,
        table_id,
        tensor: tf.Tensor,
        rows: List[str],
        columns: List[str],
    ):
        assert len(tensor.shape) == 2
        assert len(rows) == tensor.shape[0]
        assert len(columns) == tensor.shape[1]
        cli_path = _get_cbt_cli_path()
        for i, row in enumerate(tensor):
            for j, value in enumerate(row):
                cmd = [
                    cli_path,
                    "-project",
                    project_id,
                    "-instance",
                    instance_id,
                    "set",
                    table_id,
                    rows[i],
                    f"{columns[j]}={value.numpy().decode()}",
                ]
                subprocess.check_output(cmd)

    def stop(self):
        self._emulator.terminate()
        self._output_reading_thread.join()
        self._emulator.stdout.close()
        self._emulator.wait()
