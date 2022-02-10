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
import sys
import pathlib
import subprocess
import textwrap

exec(pathlib.Path("tensorflow_io/python/ops/version_ops.py").read_text())

# the `version` variable is loaded from the `exec`` above
if f"v{version}" != sys.argv[1]:
    print(f"[Version {version} in version_ops.py does not match {sys.argv[1]}]")
    sys.exit(1)
version = sys.argv[1]

note = pathlib.Path("RELEASE.md").read_text()

entries = []
for index, line in enumerate(note.split("\n")):
    if re.match(r"^# Release [\d\.]+", line):
        entries.append((line[len("# Release ") :].rstrip(), index))

tag = (
    subprocess.run(
        ["git", "tag", "-l", f"v{entries[0][0]}"],
        capture_output=True,
        check=True,
    )
    .stdout.decode("utf-8")
    .strip()
)
if tag == f"v{entries[0][0]}":
    print(
        "[The latest version in RELEASE.md {} has already been released]".format(
            entries[0][0]
        )
    )
    sys.exit(0)


if version != f"v{entries[0][0]}":
    print(
        "[The latest version in RELEASE.md {} has does not match {}]".format(
            entries[0][0], version
        )
    )
    sys.exit(0)

# Cut note
current = "\n".join(note.split("\n")[: entries[1][1]])
pathlib.Path("CURRENT.md").write_text(current)
