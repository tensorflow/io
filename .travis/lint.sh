set -x -e

if [[ "$#" -gt 0 && "$1" == "python" ]]; then
  python -c 'import urllib; urllib.urlretrieve("https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc", ".pylint")'
  python -m pip install pylint
  find tensorflow_io -name \*.py | xargs python -m pylint --rcfile=.pylint
  find tests -name \*.py | xargs python -m pylint --rcfile=.pylint
  exit 0
fi

# Note: clang-format is not enabled yet
if [[ "$#" -gt 0 && "$1" == "clang" ]]; then
  for f in $(find tensorflow_io/ -name '*.cc' -o -name '*.h'); do
    diff -u <(cat $f) <(bazel run --noshow_progress --noshow_loading_progress --verbose_failures --test_output=errors @llvm_toolchain//:bin/clang-format -- --style=google $f)
  done
fi
if [[ "$#" -gt 0 && "$1" == "clang-git-diff" ]]; then
  for f in $(git diff --name-status master -- 'tensorflow_io/*.cc' 'tensorflow_io/*.h' | awk '{print $2}'); do
    diff -u <(cat $f) <(bazel run --noshow_progress --noshow_loading_progress --verbose_failures --test_output=errors @llvm_toolchain//:bin/clang-format -- --style=google $f)
  done
fi

# Note: `bazel run @com_github_bazelbuild_buildtools//:buildifier` fixes the lint
# while below checks the lint
bazel run --noshow_progress --noshow_loading_progress --verbose_failures --test_output=errors --run_under="cd $PWD && " @com_github_bazelbuild_buildtools//buildifier:buildifier -- --mode=diff $(find . -type f \( -name WORKSPACE -or -name BUILD -or -name '*.BUILD' \))

docker run -i --rm -v $PWD:/v -w /v --net=host python:2.7-slim bash -x -e .travis/lint.sh python


