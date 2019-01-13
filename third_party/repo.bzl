# cc_import_library is a replacement of cc_import in bazel,
# The purposes are:
# - Allows the specification of soname e.g., libavformat.so.57
# - Restrict to only allow linking against for GPL licenses libraries,
#   so that no srcs files are used.
def cc_import_library(
        name,
        hdrs,
        libraries,
        **kwargs):
    for library in libraries:
        native.genrule(
            name = "stub-" + library,
            outs = [library],
            cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
        )
    native.cc_library(
        name = name,
        srcs = [],
        hdrs = hdrs,
        copts = [],
        defines = [],
        includes = [],
	linkopts = ["-L$(GENDIR)/external/" + native.repository_name()[1:]] + ["-l:" + x for x in libraries],
        visibility = ["//visibility:public"],
        data = libraries,
        **kwargs
    )
