import re
import pathlib


note = pathlib.Path("RELEASE.md").read_text()

release = re.findall(r"^# Release [\d\.]+", note)[0][len("# Release ") :].rstrip()

exec(pathlib.Path("tensorflow_io/python/ops/version_ops.py").read_text())

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
