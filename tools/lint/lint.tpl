#! /usr/bin/env bash

MODE=@@MODE@@

echo "$MODE: " $@


RUN_BAZEL=no
RUN_BLACK=no
RUN_CLANG=no
RUN_PYUPGRADE=no
RUN_ENTRIES=all
if [[ $# -eq 0 ]]; then
  RUN_BAZEL=true
  RUN_BLACK=true
  RUN_CLANG=true
  RUN_PYUPGRADE=true
else
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "bazel" ]]; then
      shift
      echo "$MODE: " bazel
      RUN_BAZEL=true
    elif [[ "$1" == "black" ]]; then
      shift
      echo "$MODE: " black
      RUN_BLACK=true
    elif [[ "$1" == "clang" ]]; then
      shift
      echo "$MODE: " clang
      RUN_CLANG=true
    elif [[ "$1" == "pyupgrade" ]]; then
      shift
      echo "$MODE: " pyupgrade
      RUN_PYUPGRADE=true
    elif [[ "$1" == "--" ]]; then
      shift
      echo "$MODE: " "--"
      RUN_ENTRIES="--"
      break
    else
      echo "unknown command: $i"
      exit 1
    fi
  done
fi

echo "Selected: Bazel=$RUN_BAZEL Black=$RUN_BLACK Clang=$RUN_CLANG Pyupgrade=$RUN_PYUPGRADE [Entries]: $RUN_ENTRIES"

BLACK_PATH=@@BLACK_PATH@@
BUILDIFIER_PATH=@@BUILDIFIER_PATH@@
CLANG_FORMAT_PATH=@@CLANG_FORMAT_PATH@@
PYUPGRADE_PATH=@@PYUPGRADE_PATH@@

mode="$MODE"

black_path=$(readlink "$BLACK_PATH")
buildifier_path=$(readlink "$BUILDIFIER_PATH")
clang_format_path=$(readlink "$CLANG_FORMAT_PATH")
pyupgrade_path=$(readlink "$PYUPGRADE_PATH")

echo "mode:" $mode
echo "black:" $black_path
echo "buildifier:" $buildifier_path
echo "clang-format:" $clang_format_path
echo "pyupgrade:" $pyupgrade_path

black_func() {
  echo $1 $2
  if [[ "$1" == "lint" ]]; then
    $black_path $2
  else
    $black_path --check $2
  fi
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
  if [[ "$1" == "lint" ]]; then
    $clang_format_path --style=google -i $2
  else
    diff -u <(cat $2) <($clang_format_path --style=google $2)
  fi
}

pyupgrade_func() {
  echo $1 $2
  if [[ "$1" == "lint" ]]; then
    $pyupgrade_path --exit-zero-even-if-changed --py37-plus $2
  else
    $pyupgrade_path --py37-plus $2
  fi
}

set -e

if [[ "$RUN_BLACK" == "true" ]]; then
echo "Run Black"

if [[ "$RUN_ENTRIES" == "--" ]]; then
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in $@ ; do \
        black_func $mode "$i" ; \
    done \
)
else
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in \
        $( \
            find tensorflow_io tensorflow_io_gcs_filesystem tests -type f \
                \( -name '*.py' \) \
        ) ; do \
        black_func $mode "$i" ; \
    done \
)
fi

fi


if [[ "$RUN_BAZEL" == "true" ]]; then
echo "Run Bazel Buildifier"

if [[ "$RUN_ENTRIES" == "--" ]]; then
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in $@ ; do \
        buildifier_func $mode "$i" ; \
    done \
)
else
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
fi

fi

if [[ "$RUN_CLANG" == "true" ]]; then
echo "Run Clang Format"

if [[ "$RUN_ENTRIES" == "--" ]]; then
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in $@ ; do \
        clang_format_func $mode "$i" ; \
    done \
)
else
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in \
        $( \
            find tensorflow_io tensorflow_io_gcs_filesystem -type f \
                \( \
                    -name '*.cc' \
                    -o -name '*.h' \
                \) \
        ) ; do \
        clang_format_func $mode "$i" ; \
    done \
)
fi

fi

if [[ "$RUN_PYUPGRADE" == "true" ]]; then
echo "Run Pyupgrade"

if [[ "$RUN_ENTRIES" == "--" ]]; then
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in $@ ; do \
        pyupgrade_func $mode "$i" ; \
    done \
)
else
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in \
        $( \
            find tensorflow_io tensorflow_io_gcs_filesystem tests -type f \
                \( -name '*.py' \) \
        ) ; do \
        pyupgrade_func $mode "$i" ; \
    done \
)
fi

fi
exit 0
