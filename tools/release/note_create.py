# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
import re
import pathlib

note = pathlib.Path("RELEASE.md").read_text()

release = re.findall(r"^# Release [\d\.]+", note)[0][len("# Release ") :].rstrip()

exec(pathlib.Path("tensorflow_io/python/ops/version_ops.py").read_text())

# the `version` variable is loaded from the `exec`` above
if tuple(map(int, (release.split(".")))) < tuple(map(int, (version.split(".")))):
    print(f"[Create {release}]\n")
    pathlib.Path("RELEASE.md").write_text(
        f"""# Release {version}

## Major Features and Bug Fixes
* <TODO>

## Thanks to our Contributors

This release contains contributions from many people:

<TODO>

We are also grateful to all who filed issues or helped resolve them, asked and
answered questions, and were part of inspiring discussions.

{note}"""
    )
print("\n")
print("[README.md]:\n")
print(pathlib.Path("RELEASE.md").read_text())
