import sys

sudo = False
if sys.argv[1].startswith("--sudo="):
    sudo = sys.argv[1][7:].lower() == "true"

source = sys.argv[len(sys.argv) - 2]
section = sys.argv[len(sys.argv) - 1]
with open(source, "r") as f:
    lines = [line.rstrip() for line in list(f)]

# Remove lines before section title
lines = lines[lines.index(section) :]

# Remove lines outside (including) "```sh" and "```"
lines = lines[lines.index("```sh") + 1 : lines.index("```")]

# Remove sudo
if not sudo:
    lines = [
        (line[len("sudo ") :] if line.startswith("sudo ") else line) for line in lines
    ]

print("\n".join(lines))
