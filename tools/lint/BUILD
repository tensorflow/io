load("defs.bzl", "lint")
load("@lint_dependencies//:requirements.bzl", "requirement")

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
    name = "pylint_py",
    srcs = ["pylint_python.py"],
    main = "pylint_python.py",
    deps = [
        requirement("pylint"),
        requirement("astroid"),
        requirement("isort"),
        requirement("lazy_object_proxy"),
    ],
)

filegroup(
    name = "pylint_rc",
    srcs = [
        "pylint_python.rc",
    ],
)

genrule(
    name = "pylint",
    srcs = [],
    outs = ["pylint"],
    cmd = "echo '$(location :pylint_py) --rcfile=$(location :pylint_rc) \"$$@\"' > $@",
    executable = True,
    tools = [
        ":pylint_py",
        ":pylint_rc",
    ],
)

genrule(
    name = "clang_format",
    srcs = ["@llvm_toolchain//:bin/clang-format"],
    outs = ["clang-format"],
    cmd = "cat $< > $@",
    executable = True,
)
