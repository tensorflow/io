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
    print("[The latest version in RELEASE.md {} has already been released. No Update]")
    sys.exit(0)

# Find current version
curr = tuple(map(int, (entries[0][0].split("."))))


# Find last version, padded to patch version of 0
last = tuple(map(int, (entries[1][0].split("."))))

print(
    "[Authors {} => {}]:\n".format(".".join(map(str, last)), ".".join(map(str, curr)))
)
logs = subprocess.run(
    ["git", "shortlog", "v{}..HEAD".format(".".join(map(str, last))), "-s"],
    capture_output=True,
    check=True,
).stdout.decode("utf-8")

authors = [e.split("\t")[1].strip() for e in logs.rstrip().split("\n")]
authors = list(filter(lambda e: e != "github-actions[bot]", authors))
# Replace " " with "@" to allow text wrap without break names, then replace back
authors = textwrap.fill(", ".join([e.replace(" ", "@") for e in authors]), 80).replace(
    "@", " "
)

print(authors)

print("\n")

current_note = note.split("\n")[: entries[1][1]]
for index, line in enumerate(current_note):
    if re.match(r"^This release contains contributions from many people:", line):
        current_author_index_start = index + 2
    if re.match(r"^We are also grateful to all who ", line):
        current_author_index_final = index - 1

note = "\n".join(
    note.split("\n")[:current_author_index_start]
    + [authors]
    + note.split("\n")[current_author_index_final:]
)
print("[RELEASE.md]:\n")
print(note)

pathlib.Path("RELEASE.md").write_text(note)
