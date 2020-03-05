load("@bazel_skylib//lib:shell.bzl", "shell")

def _lint_impl(ctx):
    bash_file = ctx.actions.declare_file(ctx.label.name + ".bash")
    substitutions = {
        "@@MODE@@": shell.quote(ctx.attr.mode),
        "@@PYLINT_PATH@@": shell.quote(ctx.executable._pylint.short_path),
        "@@BUILDIFIER_PATH@@": shell.quote(ctx.executable._buildifier.short_path),
        "@@CLANG_FORMAT_PATH@@": shell.quote(ctx.executable._clang_format.short_path),
    }
    ctx.actions.expand_template(
        template = ctx.file._runner,
        output = bash_file,
        substitutions = substitutions,
        is_executable = True,
    )
    runfiles = ctx.runfiles(files = [ctx.executable._buildifier, ctx.executable._clang_format, ctx.executable._pylint])
    return [DefaultInfo(
        files = depset([bash_file]),
        runfiles = runfiles,
        executable = bash_file,
    )]

_lint = rule(
    implementation = _lint_impl,
    attrs = {
        "mode": attr.string(
            default = "lint",
            doc = "Formatting mode",
            values = ["lint", "check"],
        ),
        "_pylint": attr.label(
            default = "//tools/lint:pylint",
            cfg = "host",
            executable = True,
        ),
        "_buildifier": attr.label(
            default = "@com_github_bazelbuild_buildtools//buildifier",
            cfg = "host",
            executable = True,
        ),
        "_clang_format": attr.label(
            default = "//tools/lint:clang_format",
            cfg = "host",
            executable = True,
        ),
        "_runner": attr.label(
            default = "//tools/lint:lint.tpl",
            allow_single_file = True,
        ),
    },
    executable = True,
)

def lint(**kwargs):
    tags = kwargs.get("tags", [])
    if "manual" not in tags:
        tags.append("manual")
        kwargs["tags"] = tags
    _lint(**kwargs)
