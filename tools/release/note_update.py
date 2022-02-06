import re
import pathlib
import subprocess
import textwrap

note = pathlib.Path("RELEASE.md").read_text()

entries = []
for index, line in enumerate(note.split("\n")):
    if re.match(r"^# Release [\d\.]+", line):
        entries.append((line[len("# Release ") :].rstrip(), index))

# Find current version and last version
curr = tuple(map(int, (entries[0][0].split("."))))
# Find last minor version
last = tuple([curr[0], curr[1] - 1, curr[2]])

print(
    "[Authors {} => {}]:\n".format(".".join(map(str, last)), ".".join(map(str, curr)))
)
logs = subprocess.run(
    ["git", "shortlog", "v{}..HEAD".format(".".join(map(str, last))), "-s"],
    capture_output=True,
    check=False,
).stdout.decode("utf-8")

authors = [e.split("\t")[1].strip() for e in logs.rstrip().split("\n")]
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
