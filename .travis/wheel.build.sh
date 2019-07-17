set -e

if [[ "$1" == "--"* ]]; then
  VERSION_CHOICE=$1
  VERSION_NUMBER=$2
  shift
  shift
fi

for entry in "$@" ; do

  if [[ $(uname) == "Darwin" && $entry == *"3"* ]]; then
    # If on macOS and building python3 packages
    minor=$(echo $entry | cut -d. -f2)
    patch=$(echo $entry | cut -d. -f3)
    # Specify version number for local use
    pyenv local "3.${minor}.${patch}"
    # Drop patch number from executable command
    entry=$(pyenv which python)
  fi

  $entry --version
  $entry -m pip --version
  # Let's also build a release candidate if it is nightly build
  if [[ "$VERSION_CHOICE" == "--nightly" ]]; then
    $entry setup.py --data bazel-bin -q bdist_wheel
  fi
  $entry setup.py --data bazel-bin -q bdist_wheel $VERSION_CHOICE $VERSION_NUMBER
done
ls dist/*
for f in dist/*.whl; do
  if [[ $(uname) == "Darwin" ]]; then
    delocate-wheel -w wheelhouse  $f
  else
    auditwheel repair $f
  fi
done
ls wheelhouse/*
