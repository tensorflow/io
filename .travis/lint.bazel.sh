set -x -e

docker run -i -t --rm -v $PWD:/v -w /v --net=host golang:1.12 bash -x -e -c 'go get github.com/bazelbuild/buildtools/buildifier && buildifier --mode=diff $(find . -type f \( -iname WORKSPACE -or -iname BUILD -or -iname *.BUILD \))'
