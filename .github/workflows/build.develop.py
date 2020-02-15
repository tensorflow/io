import sys

source = sys.argv[1]
section = sys.argv[2]
with open (source, "r") as f:
    lines = [line.rstrip() for line in list(f)]

# Remove lines before section title
lines = lines[lines.index(section):]

# Remove lines outside (including) "```sh" and "```"
lines = lines[lines.index("```sh")+1:lines.index("```")]

# Remove sudo
lines = [(line[len("sudo "):] if line.startswith("sudo ") else line) for line in lines]

print("\n".join(lines))
