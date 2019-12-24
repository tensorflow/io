#! /usr/bin/env bash

MODE=@@MODE@@

PYLINT_PATH=@@PYLINT_PATH@@
BUILDIFIER_PATH=@@BUILDIFIER_PATH@@
CLANG_FORMAT_PATH=@@CLANG_FORMAT_PATH@@

mode="$MODE"

pylint_path=$(readlink "$PYLINT_PATH")
buildifier_path=$(readlink "$BUILDIFIER_PATH")
clang_format_path=$(readlink "$CLANG_FORMAT_PATH")

echo "mode:" $mode
echo "pylint:" $pylint_path
echo "buildifier:" $buildifier_path
echo "clang-format:" $clang_format_path

if [[ "$mode" == "lint" ]]; then
    echo 
    echo "WARN: pylint does not have lint mode, use check instead"
    echo 
fi


pylint_func() {
  echo $1 $2
  return # TODO: enable after python 2 deprecation
  $pylint_path $2
}

buildifier_func() {
  echo $1 $2
  if [[ "$1" == "lint" ]]; then
    $buildifier_path -mode=fix $2
  else
    $buildifier_path -mode=check $2
  fi
}

clang_format_func() {
  echo $1 $2
  return # TODO: enable after TF 2.1
  if [[ "$1" == "lint" ]]; then
    $clang_format_path --style=google -i $2
  else
    diff -u <(cat $2) <($clang_format_path --style=google $2)
  fi
}

set -e

( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in \
        $( \
            find -f tensorflow_io tests -type f \
                \( -name '*.py' \) \
        ) ; do \
        pylint_func $mode "$i" ; \
    done \
)

( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in \
        $( \
            find . -type f \
                \( \
                    -name '*.bzl' \
                    -o -name '*.sky' \
                    -o -name '*.BUILD' \
                    -o -name 'BUILD.*.bazel' \
                    -o -name 'BUILD.*.oss' \
                    -o -name 'WORKSPACE.*.bazel' \
                    -o -name 'WORKSPACE.*.oss' \
                    -o -name BUILD.bazel \
                    -o -name BUILD \
                    -o -name WORKSPACE \
                    -o -name WORKSPACE.bazel \
                    -o -name WORKSPACE.oss \
                \) \
        ) ; do \
        buildifier_func $mode "$i" ; \
    done \
)

( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in \
        $( \
            find tensorflow_io -type f \
                \( \
                    -name '*.cc' \
                    -o -name '*.h' \
                \) \
        ) ; do \
        clang_format_func $mode "$i" ; \
    done \
)

exit 0
