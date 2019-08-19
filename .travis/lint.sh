set -x -e

if [[ "$#" -gt 0 && "$1" == "python" ]]; then
  python -c 'import urllib; urllib.urlretrieve("https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc", ".pylint")'
  python -m pip install pylint
  find tensorflow_io -name \*.py | xargs python -m pylint --rcfile=.pylint
  find tests -name \*.py | xargs python -m pylint --rcfile=.pylint
  exit 0
fi

docker run -i -t --rm -v $PWD:/v -w /v --net=host python:2.7-slim bash -x -e .travis/lint.sh python

docker run -i -t --rm -v $PWD:/v -w /v --net=host golang:1.12 bash -x -e -c 'go get github.com/bazelbuild/buildtools/buildifier && buildifier --mode=diff $(find . -type f \( -name WORKSPACE -or -name BUILD -or -name *.BUILD \))'

