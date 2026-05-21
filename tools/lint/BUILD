load("defs.bzl", "lint")

package(default_visibility = ["//visibility:public"])

lint(
    name = "lint",
    mode = "lint",
)

lint(
    name = "check",
    mode = "check",
)

exports_files(["lint.tpl"])

py_binary(
    name = "pyupgrade_py",
    srcs = ["pyupgrade_python.py"],
    main = "pyupgrade_python.py",
    deps = [
    ],
)

genrule(
    name = "pyupgrade",
    srcs = [],
    outs = ["pyupgrade"],
    cmd = "echo 'pyupgrade \"$$@\"' > $@",
    executable = True,
)

py_binary(
    name = "black_py",
    srcs = ["black_python.py"],
    main = "black_python.py",
    deps = [
    ],
)

genrule(
    name = "black",
    srcs = [],
    outs = ["black"],
    cmd = "echo 'black \"$$@\"' > $@",
    executable = True,
)

genrule(
    name = "clang_format",
    srcs = ["@llvm_toolchain//:bin/clang-format"],
    outs = ["clang-format"],
    cmd = "cat $< > $@",
    executable = True,
)
